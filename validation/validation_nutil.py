import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from nutil.core.nutil import Nutil


class TestNutilInitialization:
    """Test class for Nutil initialization, validation, and constructor behavior."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_atlas_files(self, temp_dir):
        """Create mock atlas files for testing."""
        atlas_path = Path(temp_dir) / "test_atlas.nrrd"
        label_path = Path(temp_dir) / "test_labels.csv"
        hemi_path = Path(temp_dir) / "test_hemi.nrrd"
        
        # Create dummy files
        atlas_path.touch()
        hemi_path.touch()
        
        # Create a minimal CSV for labels
        label_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Region1', 'Region2', 'Region3'],
            'acronym': ['R1', 'R2', 'R3']
        })
        label_data.to_csv(label_path, index=False)
        
        return str(atlas_path), str(label_path), str(hemi_path)
    
    @pytest.fixture
    def valid_atlas_paths(self):
        """Return paths to actual atlas files for integration testing."""
        base_path = Path("atlases/pynutil-allen_atlases")
        atlas_path = base_path / "allen_annotation_2017_25um.nrrd"
        label_path = base_path / "allen_labels.csv"
        
        if atlas_path.exists() and label_path.exists():
            return str(atlas_path), str(label_path), None
        else:
            pytest.skip("Atlas files not found for integration testing")

    def test_init_missing_atlas_path_raises_error(self):
        """Test that missing atlas_path raises ValueError."""
        with pytest.raises(ValueError, match="The atlas_path parameter is required"):
            Nutil(label_path="some_path.csv")
    
    def test_init_missing_label_path_raises_error(self):
        """Test that missing label_path raises ValueError."""
        with pytest.raises(ValueError, match="The label_path parameter is required"):
            Nutil(atlas_path="some_path.nrrd")
    
    def test_init_both_atlas_and_label_missing_raises_error(self):
        """Test that missing both atlas_path and label_path raises ValueError."""
        with pytest.raises(ValueError, match="The atlas_path parameter is required"):
            Nutil()
    
    def test_init_nonexistent_atlas_files_raises_error(self):
        """Test that nonexistent atlas files raise ValueError."""
        with pytest.raises(ValueError, match="Error loading atlas files"):
            Nutil(
                atlas_path="nonexistent_atlas.nrrd",
                label_path="nonexistent_labels.csv"
            )
    
    @patch('nutil.core.nutil.load_custom_atlas')
    def test_init_atlas_loading_error_propagation(self, mock_load_atlas):
        """Test that atlas loading errors are properly propagated."""
        mock_load_atlas.side_effect = FileNotFoundError("Atlas file not found")
        
        with pytest.raises(ValueError, match="Error loading atlas files"):
            Nutil(
                atlas_path="test_atlas.nrrd",
                label_path="test_labels.csv"
            )
    
    @patch('nutil.core.nutil.load_custom_atlas')
    def test_init_json_decode_error_propagation(self, mock_load_atlas):
        """Test that JSON decode errors are properly propagated."""
        mock_load_atlas.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        with pytest.raises(ValueError, match="Error decoding JSON from atlas files"):
            Nutil(
                atlas_path="test_atlas.nrrd",
                label_path="test_labels.csv"
            )
    
    @patch('nutil.core.nutil.load_custom_atlas')
    def test_init_unexpected_error_propagation(self, mock_load_atlas):
        """Test that unexpected errors are properly wrapped."""
        mock_load_atlas.side_effect = RuntimeError("Unexpected error")
        
        with pytest.raises(ValueError, match="Unexpected initialization error"):
            Nutil(
                atlas_path="test_atlas.nrrd",
                label_path="test_labels.csv"
            )
    
    @patch('nutil.core.nutil.load_custom_atlas')
    def test_init_successful_with_minimal_params(self, mock_load_atlas):
        """Test successful initialization with minimal required parameters."""
        # Mock the atlas loading to return expected values
        mock_atlas_volume = np.zeros((10, 10, 10))
        mock_hemi_map = np.ones((10, 10, 10))
        mock_atlas_labels = pd.DataFrame({
            'id': [1, 2], 
            'name': ['Region1', 'Region2']
        })
        mock_load_atlas.return_value = (mock_atlas_volume, mock_hemi_map, mock_atlas_labels)
        
        nutil = Nutil(
            atlas_path="test_atlas.nrrd",
            label_path="test_labels.csv"
        )
        
        # Verify attributes are set correctly
        assert nutil.atlas_path == "test_atlas.nrrd"
        assert nutil.label_path == "test_labels.csv"
        assert nutil.segmentation_folder is None
        assert nutil.alignment_json is None
        assert nutil.colour is None
        assert nutil.hemi_path is None
        assert nutil.custom_region_path is None
        
        # Verify atlas data is loaded
        assert nutil.atlas_volume is not None
        assert nutil.hemi_map is not None
        assert isinstance(nutil.atlas_labels, pd.DataFrame)
    
    @patch('nutil.core.nutil.load_custom_atlas')
    def test_init_successful_with_all_params(self, mock_load_atlas):
        """Test successful initialization with all parameters."""
        # Mock the atlas loading
        mock_atlas_volume = np.zeros((10, 10, 10))
        mock_hemi_map = np.ones((10, 10, 10))
        mock_atlas_labels = pd.DataFrame({
            'id': [1, 2], 
            'name': ['Region1', 'Region2']
        })
        mock_load_atlas.return_value = (mock_atlas_volume, mock_hemi_map, mock_atlas_labels)
        
        nutil = Nutil(
            segmentation_folder="/path/to/segmentations",
            alignment_json="/path/to/alignment.json",
            colour=[255, 0, 0],
            atlas_path="test_atlas.nrrd",
            label_path="test_labels.csv",
            hemi_path="test_hemi.nrrd",
            custom_region_path="/path/to/custom_regions.json"
        )
        
        # Verify all attributes are set correctly
        assert nutil.segmentation_folder == "/path/to/segmentations"
        assert nutil.alignment_json == "/path/to/alignment.json"
        assert nutil.colour == [255, 0, 0]
        assert nutil.atlas_path == "test_atlas.nrrd"
        assert nutil.label_path == "test_labels.csv"
        assert nutil.hemi_path == "test_hemi.nrrd"
        assert nutil.custom_region_path == "/path/to/custom_regions.json"
    
    def test_validate_atlas_params_with_valid_inputs(self):
        """Test _validate_atlas_params with valid inputs."""
        with patch('nutil.core.nutil.load_custom_atlas'):
            nutil = Nutil.__new__(Nutil)  # Create instance without calling __init__
            # This should not raise an exception
            nutil._validate_atlas_params("valid_atlas.nrrd", "valid_labels.csv")
    
    def test_validate_atlas_params_missing_atlas_path(self):
        """Test _validate_atlas_params with missing atlas_path."""
        with patch('nutil.core.nutil.load_custom_atlas'):
            nutil = Nutil.__new__(Nutil)  # Create instance without calling __init__
            with pytest.raises(ValueError, match="The atlas_path parameter is required"):
                nutil._validate_atlas_params(None, "valid_labels.csv")
    
    def test_validate_atlas_params_empty_atlas_path(self):
        """Test _validate_atlas_params with empty atlas_path."""
        with patch('nutil.core.nutil.load_custom_atlas'):
            nutil = Nutil.__new__(Nutil)  # Create instance without calling __init__
            with pytest.raises(ValueError, match="The atlas_path parameter is required"):
                nutil._validate_atlas_params("", "valid_labels.csv")
    
    def test_validate_atlas_params_missing_label_path(self):
        """Test _validate_atlas_params with missing label_path."""
        with patch('nutil.core.nutil.load_custom_atlas'):
            nutil = Nutil.__new__(Nutil)  # Create instance without calling __init__
            with pytest.raises(ValueError, match="The label_path parameter is required"):
                nutil._validate_atlas_params("valid_atlas.nrrd", None)
    
    def test_validate_atlas_params_empty_label_path(self):
        """Test _validate_atlas_params with empty label_path."""
        with patch('nutil.core.nutil.load_custom_atlas'):
            nutil = Nutil.__new__(Nutil)  # Create instance without calling __init__
            with pytest.raises(ValueError, match="The label_path parameter is required"):
                nutil._validate_atlas_params("valid_atlas.nrrd", "")


class TestNutilMethodValidation:
    """Test class for Nutil method validation and error handling."""
    
    def setup_method(self):
        """Set up a valid Nutil instance for method testing."""
        mock_atlas_volume = np.zeros((10, 10, 10))
        mock_hemi_map = np.ones((10, 10, 10))
        mock_atlas_labels = pd.DataFrame({
            'id': [1, 2], 
            'name': ['Region1', 'Region2']
        })
        
        with patch('nutil.core.nutil.load_custom_atlas') as mock_load:
            mock_load.return_value = (mock_atlas_volume, mock_hemi_map, mock_atlas_labels)
            self.nutil = Nutil(
                atlas_path="test_atlas.nrrd",
                label_path="test_labels.csv"
            )
    
    def test_get_coordinates_missing_segmentation_folder(self):
        """Test get_coordinates raises error when segmentation_folder is None."""
        with pytest.raises(ValueError, match="Segmentation folder must be provided"):
            self.nutil.get_coordinates()
    
    def test_get_coordinates_missing_alignment_json(self):
        """Test get_coordinates raises error when alignment_json is None."""
        self.nutil.segmentation_folder = "/path/to/segmentations"
        with pytest.raises(ValueError, match="Alignment JSON must be provided"):
            self.nutil.get_coordinates()
    
    def test_get_coordinates_missing_colour(self):
        """Test get_coordinates raises error when colour is None."""
        self.nutil.segmentation_folder = "/path/to/segmentations"
        self.nutil.alignment_json = "/path/to/alignment.json"
        with pytest.raises(ValueError, match="Colour must be provided"):
            self.nutil.get_coordinates()
    
    def test_quantify_coordinates_before_get_coordinates(self):
        """Test quantify_coordinates raises error when called before get_coordinates."""
        with pytest.raises(ValueError, match="Please run get_coordinates before running quantify_coordinates"):
            self.nutil.quantify_coordinates()
    
    def test_save_analysis_before_quantify_coordinates(self):
        """Test save_analysis raises error when called before quantify_coordinates."""
        with pytest.raises(ValueError, match="Please run get_coordinates before saving"):
            self.nutil.save_analysis("/output/path")
    
    def test_save_analysis_after_coordinates_before_quantify(self):
        """Test save_analysis raises error when called after get_coordinates but before quantify_coordinates."""
        # Mock that get_coordinates has been run by adding pixel_points attribute
        self.nutil.pixel_points = np.array([[1, 2, 3]])
        
        with pytest.raises(ValueError, match="Please run quantify_coordinates before saving"):
            self.nutil.save_analysis("/output/path")
    
    def test_get_region_summary_before_quantify_coordinates(self):
        """Test get_region_summary raises error when called before quantify_coordinates."""
        with pytest.raises(ValueError, match="Please run quantify_coordinates before getting region summary"):
            self.nutil.get_region_summary()


class TestNutilEdgeCases:
    """Test class for edge cases and boundary conditions."""
    
    @patch('nutil.core.nutil.load_custom_atlas')
    def test_init_with_none_values(self, mock_load_atlas):
        """Test initialization with explicit None values."""
        mock_atlas_volume = np.zeros((10, 10, 10))
        mock_hemi_map = np.ones((10, 10, 10))
        mock_atlas_labels = pd.DataFrame({'id': [1], 'name': ['Region1']})
        mock_load_atlas.return_value = (mock_atlas_volume, mock_hemi_map, mock_atlas_labels)
        
        nutil = Nutil(
            segmentation_folder=None,
            alignment_json=None,
            colour=None,
            atlas_path="test_atlas.nrrd",
            label_path="test_labels.csv",
            hemi_path=None,
            custom_region_path=None
        )
        
        # Verify None values are preserved
        assert nutil.segmentation_folder is None
        assert nutil.alignment_json is None
        assert nutil.colour is None
        assert nutil.hemi_path is None
        assert nutil.custom_region_path is None
    
    @patch('nutil.core.nutil.load_custom_atlas')
    def test_colour_as_empty_list(self, mock_load_atlas):
        """Test initialization with empty colour list."""
        mock_atlas_volume = np.zeros((10, 10, 10))
        mock_hemi_map = np.ones((10, 10, 10))
        mock_atlas_labels = pd.DataFrame({'id': [1], 'name': ['Region1']})
        mock_load_atlas.return_value = (mock_atlas_volume, mock_hemi_map, mock_atlas_labels)
        
        nutil = Nutil(
            atlas_path="test_atlas.nrrd",
            label_path="test_labels.csv",
            colour=[]
        )
        
        assert nutil.colour == []
    
    @patch('nutil.core.nutil.load_custom_atlas')
    def test_colour_as_single_value_list(self, mock_load_atlas):
        """Test initialization with single value colour list."""
        mock_atlas_volume = np.zeros((10, 10, 10))
        mock_hemi_map = np.ones((10, 10, 10))
        mock_atlas_labels = pd.DataFrame({'id': [1], 'name': ['Region1']})
        mock_load_atlas.return_value = (mock_atlas_volume, mock_hemi_map, mock_atlas_labels)
        
        nutil = Nutil(
            atlas_path="test_atlas.nrrd",
            label_path="test_labels.csv",
            colour=[255]
        )
        
        assert nutil.colour == [255]

