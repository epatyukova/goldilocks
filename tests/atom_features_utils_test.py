import sys
import os
import pytest
import json
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
from models.atom_features_utils import (  
    load_atom_features,
    atom_features_from_structure,
    atomic_soap_features,
    atomic_soap_features_for_composition
)


@pytest.fixture
def sample_structure():
    """Create a sample structure for testing"""
    # Create a simple SiO2 structure
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice, species=species, coords=coords)


@pytest.fixture
def mock_atom_features_file(tmp_path):
    """Create a temporary atom features JSON file"""
    # Create mock embeddings for common elements
    atom_features_dict = {
        "1": [float(i) for i in range(64)],  # H
        "8": [float(i + 1) for i in range(64)],  # O
        "14": [float(i + 2) for i in range(64)],  # Si
    }
    
    file_path = tmp_path / "atom_init_test.json"
    with open(file_path, 'w') as f:
        json.dump(atom_features_dict, f)
    
    return str(file_path)


@pytest.fixture
def mock_atom_features_file_86(tmp_path):
    """Create a temporary atom features JSON file with 86 features"""
    # Create mock embeddings with 86 features (ALIGNN standard)
    atom_features_dict = {
        "1": [float(i) for i in range(86)],  # H
        "8": [float(i + 1) for i in range(86)],  # O
        "14": [float(i + 2) for i in range(86)],  # Si
    }
    
    file_path = tmp_path / "atom_init_test_86.json"
    with open(file_path, 'w') as f:
        json.dump(atom_features_dict, f)
    
    return str(file_path)


def test_load_atom_features(mock_atom_features_file):
    """Test loading atom features from JSON file"""
    features_dict = load_atom_features(mock_atom_features_file)
    
    assert isinstance(features_dict, dict)
    assert "1" in features_dict  # H
    assert "8" in features_dict  # O
    assert "14" in features_dict  # Si
    assert len(features_dict["1"]) == 64
    assert isinstance(features_dict["1"], list)


def test_load_atom_features_nonexistent_file():
    """Test loading from non-existent file raises error"""
    with pytest.raises(FileNotFoundError):
        load_atom_features("nonexistent_file.json")


def test_atom_features_from_structure_basic(sample_structure, mock_atom_features_file):
    """Test extracting atom features from structure"""
    atomic_features = {
        'atom_feature_strategy': {
            'atom_feature_file': mock_atom_features_file,
            'soap_atomic': False
        }
    }
    
    features = atom_features_from_structure(sample_structure, atomic_features)
    
    assert len(features) == 3  # 3 atoms
    assert len(features[0]) == 64  # Si features
    assert len(features[1]) == 64  # O features
    assert len(features[2]) == 64  # O features
    assert isinstance(features[0], (list, np.ndarray))


def test_atom_features_from_structure_86_features(sample_structure, mock_atom_features_file_86):
    """Test extracting atom features with 86 features (ALIGNN standard)"""
    atomic_features = {
        'atom_feature_strategy': {
            'atom_feature_file': mock_atom_features_file_86,
            'soap_atomic': False
        }
    }
    
    features = atom_features_from_structure(sample_structure, atomic_features)
    
    assert len(features) == 3  # 3 atoms
    assert len(features[0]) == 86  # Si features
    assert len(features[1]) == 86  # O features
    assert len(features[2]) == 86  # O features


def test_atom_features_from_structure_missing_element(sample_structure, mock_atom_features_file):
    """Test that missing element raises ValueError"""
    # Create structure with element not in features file
    structure = Structure(
        lattice=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        species=['Fe'],  # Fe not in mock file
        coords=[[0, 0, 0]]
    )
    
    atomic_features = {
        'atom_feature_strategy': {
            'atom_feature_file': mock_atom_features_file,
            'soap_atomic': False
        }
    }
    
    with pytest.raises(ValueError, match="Atomic feature not found"):
        atom_features_from_structure(structure, atomic_features)


def test_atom_features_from_structure_inconsistent_dimensions(tmp_path):
    """Test that inconsistent feature dimensions raise ValueError"""
    # Create features file with inconsistent dimensions
    atom_features_dict = {
        "8": [float(i) for i in range(64)],  # O - 64 features
        "14": [float(i) for i in range(86)],  # Si - 86 features (inconsistent!)
    }
    
    file_path = tmp_path / "inconsistent_features.json"
    with open(file_path, 'w') as f:
        json.dump(atom_features_dict, f)
    
    atomic_features = {
        'atom_feature_strategy': {
            'atom_feature_file': str(file_path),
            'soap_atomic': False
        }
    }
    
    # Create structure with both Si and O to trigger the inconsistency check
    structure = Structure(
        lattice=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        species=['Si', 'O'],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    
    # The function should raise ValueError when it detects inconsistent dimensions
    with pytest.raises(ValueError, match="Inconsistent feature dimension"):
        atom_features_from_structure(structure, atomic_features)


def test_atomic_soap_features(sample_structure):
    """Test SOAP feature generation"""
    soap_params = {
        'r_cut': 5.0,
        'n_max': 4,
        'l_max': 3,
        'sigma': 1.0
    }
    
    soap_features = atomic_soap_features(sample_structure, soap_params)
    
    assert isinstance(soap_features, np.ndarray)
    assert soap_features.shape[0] == len(sample_structure)  # One feature vector per atom
    assert soap_features.shape[1] > 0  # Should have some features


def test_atomic_soap_features_for_composition(sample_structure):
    """Test SOAP feature generation for composition"""
    
    soap_params = {
        'r_cut': 5.0,
        'n_max': 4,
        'l_max': 3,
        'sigma': 1.0
    }
    
    soap_features = atomic_soap_features_for_composition(sample_structure, soap_params)
    assert isinstance(soap_features, np.ndarray)
    # Should have one feature vector per unique element
    comp = Composition(sample_structure.formula)
    assert soap_features.shape[0] == len(comp)  # One per unique element
    assert soap_features.shape[1] > 0  # Should have some features


def test_atom_features_from_structure_with_soap(sample_structure, mock_atom_features_file):
    """Test atom features with SOAP features enabled"""
    
    atomic_features = {
        'atom_feature_strategy': {
            'atom_feature_file': mock_atom_features_file,
            'soap_atomic': True
        },
        'soap_params': {
            'r_cut': 5.0,
            'n_max': 4,
            'l_max': 3,
            'sigma': 1.0
        }
    }
    
    features = atom_features_from_structure(sample_structure, atomic_features)
    
    assert len(features) == 3  # 3 atoms
    # Features should be longer than base features due to SOAP concatenation
    assert len(features[0]) > 64  # Base features (64) + SOAP features


def test_atom_features_from_structure_empty_structure(tmp_path):
    """Test with empty structure (should handle gracefully)"""
    # Create empty structure
    structure = Structure(
        lattice=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        species=[],
        coords=[]
    )
    
    atom_features_dict = {"8": [float(i) for i in range(64)]}
    file_path = tmp_path / "empty_test.json"
    with open(file_path, 'w') as f:
        json.dump(atom_features_dict, f)
    
    atomic_features = {
        'atom_feature_strategy': {
            'atom_feature_file': str(file_path),
            'soap_atomic': False
        }
    }
    
    features = atom_features_from_structure(structure, atomic_features)
    assert len(features) == 0  # No atoms, no features


def test_atom_features_from_structure_different_elements(sample_structure, mock_atom_features_file):
    """Test with structure containing multiple different elements"""
    atomic_features = {
        'atom_feature_strategy': {
            'atom_feature_file': mock_atom_features_file,
            'soap_atomic': False
        }
    }
    
    features = atom_features_from_structure(sample_structure, atomic_features)
    
    # Should have features for all atoms
    assert len(features) == 3
    # All features should have same dimension
    assert all(len(f) == len(features[0]) for f in features)
    # Si and O features should be different (different embeddings)
    # Si is element 14, O is element 8
    assert not np.array_equal(features[0], features[1])  # Si != O

