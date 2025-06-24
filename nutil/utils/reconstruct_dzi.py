import cv2
import numpy as np
import os
import zipfile
import xmltodict
import logging
import sys
import re
from typing import Optional, Tuple

# Set up module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add console handler if none exist (avoid duplicates)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def reconstruct_dzi(zip_file_path: str):
    """
    Reconstructs a Deep Zoom Image from a zip file containing DZI tiles.
    
    Args:
        zip_file_path: Path to the zip file with DZI tiles
        
    Returns:
        np.ndarray: Reconstructed image as BGR numpy array
        
    Raises:
        FileNotFoundError: Zip file doesn't exist
        ValueError: Invalid DZI format or no tiles found
        RuntimeError: Reconstruction failed
    """
    return reconstruct_dzi_opencv(zip_file_path)


def reconstruct_dzi_opencv(zip_file_path: str):
    """
    Rebuilds a DZI image from its zip file using OpenCV.
    
    Args:
        zip_file_path: Path to zip containing DZI tiles
        
    Returns:
        np.ndarray: Complete image as BGR array
    """
    logger.info(f"Reconstructing DZI: {zip_file_path}")

    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"DZI zip not found: {zip_file_path}")

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            # Find the XML descriptor
            dzi_file = next((name for name in zip_file.namelist() if name.endswith(".dzi")), None)
            if not dzi_file:
                raise ValueError(f"No .dzi file found in {zip_file_path}")

            # Parse the DZI metadata
            with zip_file.open(dzi_file) as f:
                dzi_data = xmltodict.parse(f.read())
            
            tile_size = int(dzi_data["Image"]["@TileSize"])
            width = int(dzi_data["Image"]["Size"]["@Width"])
            height = int(dzi_data["Image"]["Size"]["@Height"])
            img_format = dzi_data["Image"]["@Format"]
            
            logger.info(f"Image: {width}x{height}, tiles: {tile_size}px, format: {img_format}")

            # Find highest resolution level (biggest number = highest detail)
            files_dir = os.path.splitext(dzi_file)[0] + "_files"
            max_level = 0
            
            for item in zip_file.namelist():
                if item.startswith(files_dir + "/"):
                    parts = item.split("/")
                    if len(parts) >= 3:
                        try:
                            level = int(parts[1])
                            max_level = max(max_level, level)
                        except ValueError:
                            continue

            # Get all tiles from the highest level
            level_dir = f"{files_dir}/{max_level}"
            tile_files = [f for f in zip_file.namelist() 
                         if f.startswith(level_dir + "/") and f.endswith(f".{img_format}")]

            if not tile_files:
                # Try flat structure (no level folders)
                tile_files = [f for f in zip_file.namelist() 
                             if f.startswith(files_dir + "/") and f.endswith(f".{img_format}")]
                
                if not tile_files:
                    raise ValueError(f"No tiles found in {zip_file_path}")

            logger.info(f"Found {len(tile_files)} tiles at level {max_level}")

            # Create blank canvas
            image = np.zeros((height, width, 3), dtype=np.uint8)
            # dtype=np.bool for a mask 

            # Place each tile in the right spot
            for tile_path in tile_files:
                try:
                    # Load tile data
                    tile_data = zip_file.read(tile_path)
                    tile_img = cv2.imdecode(np.frombuffer(tile_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    if tile_img is None:
                        logger.warning(f"Couldn't decode tile: {tile_path}")
                        continue

                    # Extract position from filename (e.g., "3_2.png" = column 3, row 2)
                    filename = os.path.basename(tile_path)
                    coords = os.path.splitext(filename)[0]
                    col, row = map(int, coords.split("_"))

                    # Calculate where this tile goes
                    x_start = col * tile_size
                    y_start = row * tile_size
                    x_end = min(x_start + tile_size, width)
                    y_end = min(y_start + tile_size, height)

                    # Handle edge tiles that might be smaller
                    tile_h, tile_w = tile_img.shape[:2]
                    actual_h = y_end - y_start
                    actual_w = x_end - x_start

                    # Place the tile (or part of it) in the image
                    image[y_start:y_end, x_start:x_end] = tile_img[:actual_h, :actual_w]

                except Exception as e:
                    logger.warning(f"Skipping problematic tile {tile_path}: {e}")
                    continue

            logger.info(f"Reconstruction complete: {image.shape}")
            return image

    except zipfile.BadZipFile as e:
        raise ValueError(f"Corrupted zip file: {zip_file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to reconstruct DZI: {e}")


# Example usage
if __name__ == "__main__":
    # Set up basic logging for standalone testing
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    logger.info("DZI reconstruction module loaded")
    # To test: uncomment and provide a real DZI zip file path
    # img = reconstruct_dzi("path/to/your/file.zip")
    # cv2.imwrite("reconstructed.png", img)
