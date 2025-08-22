"""
Section visualization utilities for creating colored atlas slice images.
"""

import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ..core.generate_target_slice import generate_target_slice
from ..core.transformations import image_to_atlas_space
from .read_and_write import load_segmentation


def create_colored_atlas_slice(
    slice_dict: Dict,
    atlas_volume: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_path: str,
    segmentation_path: Optional[str] = None,
    objects_data: Optional[List[Dict]] = None,
    scale_factor: float = 1.0,
) -> None:
    """
    Create a colored atlas slice image showing regions with their atlas colors
    and optionally overlay detected objects with region IDs.

    Args:
        slice_dict: Dictionary containing slice information including anchoring vector
        atlas_volume: 3D atlas volume
        atlas_labels: DataFrame containing atlas region information with colors
        output_path: Path to save the output image
        segmentation_path: Optional path to segmentation image for overlay
        objects_data: Optional list of object dictionaries with coordinates and region IDs
        scale_factor: Factor to scale the output image size
    """
    # Generate the atlas slice using the anchoring vector
    atlas_slice = generate_target_slice(slice_dict["anchoring"], atlas_volume)

    # Create color mapping from atlas labels
    color_map = create_atlas_color_map(atlas_labels)

    # Create colored image from atlas slice
    colored_slice = create_colored_image_from_slice(atlas_slice, color_map)

    # Scale if requested
    if scale_factor != 1.0:
        new_height = int(colored_slice.shape[0] * scale_factor)
        new_width = int(colored_slice.shape[1] * scale_factor)
        colored_slice = cv2.resize(
            colored_slice, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

    # Convert to PIL for text overlay
    pil_image = Image.fromarray(colored_slice)

    # Overlay segmentation if provided
    if segmentation_path and os.path.exists(segmentation_path):
        overlay_segmentation(pil_image, segmentation_path, slice_dict, scale_factor)

    # Overlay object locations and region IDs if provided
    if objects_data:
        overlay_objects_with_region_ids(
            pil_image, objects_data, slice_dict, scale_factor
        )

    # Save the image
    pil_image.save(output_path)


def create_atlas_color_map(
    atlas_labels: pd.DataFrame,
) -> Dict[int, Tuple[int, int, int]]:
    """
    Create a color mapping from atlas labels DataFrame.

    Args:
        atlas_labels: DataFrame containing 'idx', 'r', 'g', 'b' columns

    Returns:
        Dictionary mapping region IDs to RGB colors
    """
    color_map = {0: (0, 0, 0)}  # Background

    for _, row in atlas_labels.iterrows():
        if "idx" in row and "r" in row and "g" in row and "b" in row:
            region_id = int(row["idx"])
            r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
            color_map[region_id] = (r, g, b)

    return color_map


def create_colored_image_from_slice(
    atlas_slice: np.ndarray, color_map: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Create a colored RGB image from an atlas slice using the color mapping.

    Args:
        atlas_slice: 2D array with region IDs
        color_map: Dictionary mapping region IDs to RGB colors

    Returns:
        RGB image array
    """
    height, width = atlas_slice.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    for region_id, color in color_map.items():
        mask = atlas_slice == region_id
        colored_image[mask] = color

    # For unmapped regions, use a default gray color
    unmapped_mask = np.isin(atlas_slice, list(color_map.keys()), invert=True)
    colored_image[unmapped_mask] = (128, 128, 128)

    return colored_image


def overlay_segmentation(
    pil_image: Image.Image,
    segmentation_path: str,
    slice_dict: Dict,
    scale_factor: float = 1.0,
    alpha: float = 0.3,
) -> None:
    """
    Overlay segmentation contours on the atlas slice image.

    Args:
        pil_image: PIL Image to overlay on
        segmentation_path: Path to segmentation image
        slice_dict: Slice dictionary with transformation info
        scale_factor: Scale factor applied to the image
        alpha: Transparency for the overlay
    """
    try:
        # Load segmentation
        segmentation = load_segmentation(segmentation_path)

        # Scale segmentation to match atlas slice if needed
        atlas_height, atlas_width = slice_dict["height"], slice_dict["width"]
        seg_height, seg_width = segmentation.shape[:2]

        if seg_height != atlas_height or seg_width != atlas_width:
            segmentation = cv2.resize(
                segmentation,
                (atlas_width, atlas_height),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply scale factor
        if scale_factor != 1.0:
            new_height = int(segmentation.shape[0] * scale_factor)
            new_width = int(segmentation.shape[1] * scale_factor)
            segmentation = cv2.resize(
                segmentation, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            )

        # Convert to grayscale if needed
        if len(segmentation.shape) == 3:
            segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)

        # Find contours
        contours, _ = cv2.findContours(
            segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create overlay image
        overlay = Image.fromarray(np.zeros_like(np.array(pil_image)))
        draw = ImageDraw.Draw(overlay)

        # Draw contours
        for contour in contours:
            points = [(int(point[0][0]), int(point[0][1])) for point in contour]
            if len(points) > 2:
                draw.polygon(points, outline=(255, 255, 0), width=2)

        # Blend with original image
        pil_image.paste(Image.blend(pil_image, overlay, alpha))

    except Exception as e:
        print(f"Warning: Could not overlay segmentation from {segmentation_path}: {e}")


def overlay_objects_with_region_ids(
    pil_image: Image.Image,
    objects_data: List[Dict],
    slice_dict: Dict,
    scale_factor: float = 1.0,
    font_size: int = 12,
) -> None:
    """
    Overlay object locations with their region IDs on the image.

    Args:
        pil_image: PIL Image to overlay on
        objects_data: List of dictionaries with object information
        slice_dict: Slice dictionary with transformation info
        scale_factor: Scale factor applied to the image
        font_size: Font size for region ID labels
    """
    draw = ImageDraw.Draw(pil_image)

    try:
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", int(font_size * scale_factor))
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
    except:
        font = None

    for obj_data in objects_data:
        if "triplets" in obj_data and "idx" in obj_data and "name" in obj_data:
            triplets = obj_data["triplets"]
            region_id = obj_data["idx"]
            region_name = obj_data["name"]

            # Convert atlas coordinates back to image coordinates
            if len(triplets) >= 3:
                # Take the first coordinate (centroid)
                atlas_coords = np.array([triplets[0], triplets[1], triplets[2]])

                # Transform from atlas space back to image space
                # This is a simplified transformation - might need refinement
                image_coords = transform_atlas_to_image_coords(
                    atlas_coords, slice_dict, scale_factor
                )

                if image_coords is not None:
                    x, y = int(image_coords[0]), int(image_coords[1])

                    # Ensure coordinates are within image bounds
                    img_width, img_height = pil_image.size
                    if 0 <= x < img_width and 0 <= y < img_height:
                        # Draw a circle at the object location
                        circle_radius = max(2, int(3 * scale_factor))
                        draw.ellipse(
                            [
                                x - circle_radius,
                                y - circle_radius,
                                x + circle_radius,
                                y + circle_radius,
                            ],
                            fill=(255, 255, 255),
                            outline=(0, 0, 0),
                            width=1,
                        )

                        # Draw region ID
                        text = str(region_id)
                        if font:
                            # Get text size
                            bbox = draw.textbbox((0, 0), text, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        else:
                            text_width, text_height = len(text) * 6, 11

                        # Position text near the circle
                        text_x = x + circle_radius + 2
                        text_y = y - text_height // 2

                        # Ensure text is within bounds
                        if text_x + text_width > img_width:
                            text_x = x - circle_radius - text_width - 2
                        if text_y < 0:
                            text_y = 0
                        elif text_y + text_height > img_height:
                            text_y = img_height - text_height

                        # Draw text background
                        draw.rectangle(
                            [
                                text_x - 1,
                                text_y - 1,
                                text_x + text_width + 1,
                                text_y + text_height + 1,
                            ],
                            fill=(255, 255, 255, 200),
                            outline=(0, 0, 0),
                        )

                        # Draw text
                        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)


def transform_atlas_to_image_coords(
    atlas_coords: np.ndarray, slice_dict: Dict, scale_factor: float = 1.0
) -> Optional[Tuple[float, float]]:
    """
    Transform atlas coordinates back to image coordinates.
    This is a simplified inverse transformation.

    Args:
        atlas_coords: 3D atlas coordinates [x, y, z]
        slice_dict: Slice dictionary with anchoring vector
        scale_factor: Scale factor applied to the image

    Returns:
        2D image coordinates or None if transformation fails
    """
    try:
        # Extract anchoring vector
        anchoring = slice_dict["anchoring"]
        ox, oy, oz = anchoring[0:3]
        ux, uy, uz = anchoring[3:6]
        vx, vy, vz = anchoring[6:9]

        # This is a simplified inverse transformation
        # In practice, this might need more sophisticated calculation

        # Get image dimensions
        reg_height = slice_dict["height"]
        reg_width = slice_dict["width"]

        # Calculate approximate image coordinates
        # This assumes a linear relationship which may not be entirely accurate
        o = np.array([ox, oy, oz])
        u = np.array([ux, uy, uz])
        v = np.array([vx, vy, vz])

        # Solve for s and t such that atlas_coords ≈ o + s*u + t*v
        # This is a simplified approximation
        diff = atlas_coords - o

        # Project onto u and v vectors
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)

        if u_norm > 0 and v_norm > 0:
            s = np.dot(diff, u) / (u_norm * u_norm)
            t = np.dot(diff, v) / (v_norm * v_norm)

            # Convert to image coordinates
            image_x = s * reg_width * scale_factor
            image_y = t * reg_height * scale_factor

            return (image_x, image_y)

    except Exception as e:
        print(f"Warning: Could not transform coordinates: {e}")

    return None


def create_section_visualizations(
    segmentation_folder: str,
    alignment_json: Dict,
    atlas_volume: np.ndarray,
    atlas_labels: pd.DataFrame,
    output_folder: str,
    objects_per_section: Optional[List[List[Dict]]] = None,
    scale_factor: float = 0.5,
) -> None:
    """
    Create visualization images for all sections in the analysis.

    Args:
        segmentation_folder: Path to folder containing segmentation images
        alignment_json: Alignment JSON data
        atlas_volume: 3D atlas volume
        atlas_labels: DataFrame with atlas region information
        output_folder: Output folder for visualizations
        objects_per_section: Optional list of object data per section
        scale_factor: Scale factor for output images
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_folder, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Get list of slices from alignment JSON
    slices = alignment_json.get("slices", [])

    for i, slice_dict in enumerate(slices):
        try:
            # Find corresponding segmentation file
            filename = slice_dict.get("filename", "")
            if filename:
                # Look for segmentation file
                base_name = os.path.splitext(filename)[0]
                seg_files = [
                    f"{base_name}.png",
                    f"{base_name}_Seg.png",
                    f"{base_name}_Simple_Seg.png",
                    f"{base_name}_resize_Simple_Seg.png",
                ]

                segmentation_path = None
                for seg_file in seg_files:
                    potential_path = os.path.join(segmentation_folder, seg_file)
                    if os.path.exists(potential_path):
                        segmentation_path = potential_path
                        break

                # Get objects data for this section
                section_objects = None
                if objects_per_section and i < len(objects_per_section):
                    section_objects = objects_per_section[i]

                # Create output filename
                output_filename = f"section_{slice_dict.get('nr', i):03d}_{base_name}_atlas_colored.png"
                output_path = os.path.join(viz_dir, output_filename)

                # Create the colored slice visualization
                create_colored_atlas_slice(
                    slice_dict,
                    atlas_volume,
                    atlas_labels,
                    output_path,
                    segmentation_path,
                    section_objects,
                    scale_factor,
                )

                print(f"Created visualization: {output_filename}")

        except Exception as e:
            print(f"Error creating visualization for slice {i}: {e}")
