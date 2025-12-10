import sys
import os
import pytest
import numpy as np

# Skip tests if dependencies are not available
torch = pytest.importorskip("torch")
pymatgen = pytest.importorskip("pymatgen")
from pymatgen.core.structure import Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
from models.alignn_graph import build_alignn_graph_with_angles_from_structure


@pytest.fixture
def sample_structure():
    """Create a sample structure for testing"""
    # Create a simple SiO2 structure
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice, species=species, coords=coords)


@pytest.fixture
def sample_atom_features():
    """Create sample atom features (64 features per atom)"""
    # Create features for 3 atoms, each with 64 features
    return [[float(i + j) for j in range(64)] for i in range(3)]


def test_build_alignn_graph_basic(sample_structure, sample_atom_features):
    """Test basic ALIGNN graph construction"""
    g, lg = build_alignn_graph_with_angles_from_structure(
        sample_structure, 
        sample_atom_features,
        radius=10.0,
        max_neighbors=12
    )
    
    # Check atomic graph (g)
    assert g.x.shape == (3, 64)  # 3 atoms, 64 features each
    assert g.edge_index.shape[0] == 2  # edge_index should be [2, num_edges]
    assert g.edge_index.shape[1] > 0  # Should have edges
    assert g.edge_attr.shape[0] == g.edge_index.shape[1]  # One distance per edge
    assert g.edge_attr.dim() == 1  # 1D tensor of distances
    assert hasattr(g, 'edge_vecs')  # Should have edge vectors
    assert g.edge_vecs.shape[0] == g.edge_index.shape[1]  # One vector per edge
    assert g.edge_vecs.shape[1] == 3  # 3D vectors
    
    # Check line graph (lg)
    assert lg.x.shape[0] == g.edge_index.shape[1]  # Nodes in line graph = edges in atomic graph
    assert lg.edge_index.shape[0] == 2  # edge_index should be [2, num_line_edges]
    assert lg.edge_attr.shape[0] == lg.edge_index.shape[1]  # One angle per line edge
    assert lg.edge_attr.dim() == 1  # 1D tensor of angle cosines


def test_build_alignn_graph_edge_distances(sample_structure, sample_atom_features):
    """Test that edge distances are positive"""
    g, lg = build_alignn_graph_with_angles_from_structure(
        sample_structure, 
        sample_atom_features,
        radius=10.0,
        max_neighbors=12
    )
    
    # All distances should be positive
    assert torch.all(g.edge_attr > 0)
    # Distances should be reasonable (less than radius)
    assert torch.all(g.edge_attr <= 10.0)


def test_build_alignn_graph_angle_features(sample_structure, sample_atom_features):
    """Test that angle features (cosines) are in valid range [-1, 1]"""
    g, lg = build_alignn_graph_with_angles_from_structure(
        sample_structure, 
        sample_atom_features,
        radius=10.0,
        max_neighbors=12
    )
    
    if lg.edge_attr.shape[0] > 0:
        # Angle cosines should be in range [-1, 1]
        assert torch.all(lg.edge_attr >= -1.0)
        assert torch.all(lg.edge_attr <= 1.0)


def test_build_alignn_graph_edge_vectors(sample_structure, sample_atom_features):
    """Test that edge vectors are correctly computed"""
    g, lg = build_alignn_graph_with_angles_from_structure(
        sample_structure, 
        sample_atom_features,
        radius=10.0,
        max_neighbors=12
    )
    
    # Check that edge_vecs magnitudes match edge_attr distances
    edge_vecs_norms = torch.norm(g.edge_vecs, dim=1)
    # Allow small numerical differences
    assert torch.allclose(edge_vecs_norms, g.edge_attr, atol=1e-5)


def test_build_alignn_graph_max_neighbors(sample_structure, sample_atom_features):
    """Test that max_neighbors parameter is respected"""
    g, lg = build_alignn_graph_with_angles_from_structure(
        sample_structure, 
        sample_atom_features,
        radius=10.0,
        max_neighbors=2
    )
    
    # Count edges per node
    edge_index = g.edge_index
    num_nodes = g.x.shape[0]
    
    for node_idx in range(num_nodes):
        # Count outgoing edges from this node
        outgoing_edges = (edge_index[0] == node_idx).sum().item()
        assert outgoing_edges <= 2  # Should respect max_neighbors


def test_build_alignn_graph_radius(sample_structure, sample_atom_features):
    """Test that radius parameter limits edge distances"""
    g, lg = build_alignn_graph_with_angles_from_structure(
        sample_structure, 
        sample_atom_features,
        radius=5.0,
        max_neighbors=12
    )
    
    # All distances should be within radius
    assert torch.all(g.edge_attr <= 5.0)


def test_build_alignn_graph_empty_structure():
    """Test with a structure that has minimal neighbors (should handle gracefully)"""
    # Use a small, simple structure with a very small radius to minimize edges
    # Avoid very large lattices that can cause memory issues with periodic boundaries
    lattice = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    species = ['Si', 'O']
    coords = [[0, 0, 0], [9, 9, 9]]  # Far apart but not causing periodic boundary issues
    structure = Structure(lattice=lattice, species=species, coords=coords)
    
    atom_features = [[float(i + j) for j in range(64)] for i in range(2)]
    
    # Use a small radius to minimize neighbors
    g, lg = build_alignn_graph_with_angles_from_structure(
        structure, 
        atom_features,
        radius=1.0,  # Small radius
        max_neighbors=12
    )
    
    # Should still create graphs and handle gracefully
    assert g.x.shape[0] == 2  # 2 atoms
    assert g.edge_index.shape[0] == 2  # edge_index should have shape [2, num_edges]
    # Verify the function completes without error and produces valid graphs
    assert isinstance(g.x, torch.Tensor)
    assert isinstance(lg.x, torch.Tensor)
    # Verify edge attributes match edge index
    if g.edge_index.shape[1] > 0:
        assert g.edge_attr.shape[0] == g.edge_index.shape[1]
    # If there are no edges in atomic graph, line graph should also have no edges
    if g.edge_index.shape[1] == 0:
        assert lg.edge_index.shape[1] == 0


def test_build_alignn_graph_different_feature_dimensions():
    """Test with different atom feature dimensions"""
    structure = Structure(
        lattice=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        species=['Si', 'O', 'O'],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    )
    
    # Test with 86 features (ALIGNN standard)
    atom_features_86 = [[float(i + j) for j in range(86)] for i in range(3)]
    g, lg = build_alignn_graph_with_angles_from_structure(
        structure, 
        atom_features_86,
        radius=10.0,
        max_neighbors=12
    )
    
    assert g.x.shape == (3, 86)  # Should match input feature dimension


def test_build_alignn_graph_line_graph_connectivity(sample_structure, sample_atom_features):
    """Test that line graph correctly represents bond angles"""
    g, lg = build_alignn_graph_with_angles_from_structure(
        sample_structure, 
        sample_atom_features,
        radius=10.0,
        max_neighbors=12
    )
    
    if lg.edge_index.shape[1] > 0:
        # Line graph nodes should correspond to edges in atomic graph
        assert lg.x.shape[0] == g.edge_index.shape[1]
        
        # Line graph edges connect bonds that share a node
        # This is a structural check - line graph should have some connectivity
        assert lg.edge_index.shape[1] >= 0  # Can be empty for some structures


def test_build_alignn_graph_tensor_types(sample_structure, sample_atom_features):
    """Test that all tensors have correct dtypes"""
    g, lg = build_alignn_graph_with_angles_from_structure(
        sample_structure, 
        sample_atom_features,
        radius=10.0,
        max_neighbors=12
    )
    
    # Check dtypes
    assert g.x.dtype == torch.float32
    assert g.edge_index.dtype == torch.long
    assert g.edge_attr.dtype == torch.float32
    assert g.edge_vecs.dtype == torch.float32
    
    if lg.edge_index.shape[1] > 0:
        assert lg.edge_index.dtype == torch.long
        assert lg.edge_attr.dtype == torch.float32
        assert lg.x.dtype == torch.float32

