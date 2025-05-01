import json
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from ..utils.atlas_loader import load_custom_atlas
from .data_analysis import quantify_labeled_points
from ..utils.file_operations import save_analysis_output
from .coordinate_extraction import folder_to_atlas_space


class Neutil:
    """
    A class to perform brain-wide quantification and spatial analysis of serial section images.

    Methods
    -------
    constructor(...)
        Initialize the Neutil class with segmentation, alignment, and custom atlas settings.
    get_coordinates(...)
        Extract and transform pixel coordinates from segmentation files.
    quantify_coordinates()
        Quantify pixel and centroid counts by atlas regions.
    save_analysis(output_folder)
        Save the analysis output to the specified directory.
    """

    def __init__(
        self,
        segmentation_folder=None,
        alignment_json=None,
        colour=None,
        atlas_path=None,
        label_path=None,
        hemi_path=None,
        custom_region_path=None,
    ):
        """
        Initializes the Neutil class with the given parameters.

        Parameters
        ----------
        segmentation_folder : str, optional
            The folder containing the segmentation files (default is None).
        alignment_json : str, optional
            The path to the alignment JSON file (default is None).
        colour : list, optional
            The RGB colour of the object to be quantified in the segmentation (default is None).
        atlas_path : str, optional
            The path to the custom atlas volume file (required).
        label_path : str, optional
            The path to the custom atlas label file (required).
        hemi_path : str, optional
            The path to the hemisphere map file (optional).
        custom_region_path : str, optional
            The path to a custom region id file (optional).

        Raises
        ------
        ValueError
            If required atlas files are missing or cannot be loaded.
        """
        try:
            # Store basic parameters
            self.segmentation_folder = segmentation_folder
            self.alignment_json = alignment_json
            self.colour = colour
            self.custom_region_path = custom_region_path

            # Validate and store atlas parameters
            self._validate_atlas_params(atlas_path, label_path)
            self.atlas_path = atlas_path
            self.label_path = label_path
            self.hemi_path = hemi_path

            # Load custom atlas
            self.atlas_volume, self.hemi_map, self.atlas_labels = load_custom_atlas(
                atlas_path, hemi_path, label_path
            )

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading atlas files: {e}")
        except Exception as e:
            raise ValueError(f"Initialization error: {e}")

    def _validate_atlas_params(self, atlas_path, label_path):
        """Validate that required atlas files are provided."""
        if not atlas_path:
            raise ValueError("The atlas_path parameter is required.")
        if not label_path:
            raise ValueError("The label_path parameter is required.")

    def get_coordinates(
        self, non_linear=True, object_cutoff=0, use_flat=False, apply_damage_mask=True
    ):
        """
        Retrieves pixel and centroid coordinates from segmentation data,
        applies atlas-space transformations, and optionally uses a damage
        mask if specified.

        Parameters
        ----------
        non_linear : bool, optional
            Enable non-linear transformation (default is True).
        object_cutoff : int, optional
            Minimum object size (default is 0).
        use_flat : bool, optional
            Use flat maps if True (default is False).
        apply_damage_mask : bool, optional
            Apply damage mask if True (default is True).

        Returns
        -------
        None
            Results are stored in class attributes.

        Raises
        ------
        ValueError
            If coordinate extraction fails.
        """
        try:
            (
                self.pixel_points,
                self.centroids,
                self.points_labels,
                self.centroids_labels,
                self.points_hemi_labels,
                self.centroids_hemi_labels,
                self.region_areas_list,
                self.points_len,
                self.centroids_len,
                self.segmentation_filenames,
                self.per_point_undamaged,
                self.per_centroid_undamaged,
            ) = folder_to_atlas_space(
                self.segmentation_folder,
                self.alignment_json,
                self.atlas_labels,
                self.colour,
                non_linear,
                object_cutoff,
                self.atlas_volume,
                self.hemi_map,
                use_flat,
                apply_damage_mask,
            )
            self.apply_damage_mask = apply_damage_mask

        except Exception as e:
            raise ValueError(f"Error extracting coordinates: {e}")

    def quantify_coordinates(self):
        """
        Quantifies and summarizes pixel and centroid coordinates by atlas region,
        storing the aggregated results in class attributes.

        Attributes
        ----------
        label_df : pd.DataFrame
            Contains aggregated label information.
        per_section_df : list of pd.DataFrame
            DataFrames with section-wise statistics.

        Raises
        ------
        ValueError
            If required attributes are missing or computation fails.

        Returns
        -------
        None
        """
        if not hasattr(self, "pixel_points") or not hasattr(self, "centroids"):
            raise ValueError(
                "Please run get_coordinates before running quantify_coordinates."
            )

        try:
            (self.label_df, self.per_section_df) = quantify_labeled_points(
                self.points_len,
                self.centroids_len,
                self.region_areas_list,
                self.points_labels,
                self.centroids_labels,
                self.atlas_labels,
                self.points_hemi_labels,
                self.centroids_hemi_labels,
                self.per_point_undamaged,
                self.per_centroid_undamaged,
                self.apply_damage_mask,
            )

        except Exception as e:
            raise ValueError(f"Error quantifying coordinates: {e}")

    def save_analysis(self, output_folder):
        """
        Saves the analysis results to different files in the specified output folder.

        Parameters
        ----------
        output_folder : str
            The folder where the analysis output will be saved.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If saving fails.
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_folder).mkdir(parents=True, exist_ok=True)

            save_analysis_output(
                self.pixel_points,
                self.centroids,
                self.label_df,
                self.per_section_df,
                self.points_labels,
                self.centroids_labels,
                self.points_hemi_labels,
                self.centroids_hemi_labels,
                self.points_len,
                self.centroids_len,
                self.segmentation_filenames,
                self.atlas_labels,
                output_folder,
                segmentation_folder=self.segmentation_folder,
                alignment_json=self.alignment_json,
                colour=self.colour,
                custom_region_path=self.custom_region_path,
                atlas_path=self.atlas_path,
                label_path=self.label_path,
                settings_file=getattr(self, "settings_file", None),
                prepend="",
            )

            print(f"Analysis results saved to: {output_folder}")
        except Exception as e:
            raise ValueError(f"Error saving analysis: {e}")

    def get_region_summary(self):
        """
        Get a summary of detected objects by brain region.

        Returns
        -------
        pd.DataFrame
            Summary of quantification results by brain region

        Raises
        ------
        ValueError
            If quantification hasn't been run yet
        """
        if not hasattr(self, "label_df"):
            raise ValueError(
                "Please run quantify_coordinates before getting region summary."
            )

        return self.label_df
