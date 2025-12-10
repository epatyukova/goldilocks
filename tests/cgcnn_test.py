import sys
import os
import pytest
import torch
import numpy as np

# Skip tests if dependencies are not available
torch = pytest.importorskip("torch")
from torch_geometric.data import Data, Batch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
from models.cgcnn import (
    Standardize,
    RBFExpansion,
    CGCNNConv,
    CGCNN_PyG
)


@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing"""
    # Create a simple graph with 3 nodes and 4 edges
    x = torch.randn(3, 64)  # 3 nodes, 64 features each
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # 4 edges
    edge_attr = torch.randn(4, 1)  # 4 edges, 1 feature each (distance)
    batch = torch.zeros(3, dtype=torch.long)  # All nodes in same batch
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


@pytest.fixture
def sample_batch_data():
    """Create batched graph data for testing"""
    # Create two graphs in a batch
    x = torch.randn(5, 64)  # 5 nodes total
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 2, 4]], dtype=torch.long)
    edge_attr = torch.randn(5, 1)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)  # First 3 nodes in graph 0, last 2 in graph 1
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def test_standardize(sample_graph_data):
    """Test Standardize module"""
    mean = torch.mean(sample_graph_data.x, dim=0)
    std = torch.std(sample_graph_data.x, dim=0) + 1e-8
    
    standardize = Standardize(mean, std)
    standardized_data = standardize(sample_graph_data)
    
    # Check that data is cloned (not modified in place)
    assert standardized_data is not sample_graph_data
    
    # Check that mean is approximately 0
    assert torch.allclose(torch.mean(standardized_data.x, dim=0), torch.zeros_like(mean), atol=1e-5)
    
    # Check that std is approximately 1
    assert torch.allclose(torch.std(standardized_data.x, dim=0), torch.ones_like(std), atol=1e-5)


def test_rbf_expansion():
    """Test RBFExpansion module"""
    rbf = RBFExpansion(vmin=0, vmax=8, bins=40)
    
    # Test with scalar distances
    distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expanded = rbf(distances)
    
    assert expanded.shape == (4, 40)  # 4 distances, 40 bins
    assert torch.all(expanded >= 0)  # RBF values should be non-negative
    assert torch.all(expanded <= 1)  # RBF values should be <= 1 (exp of negative)


def test_rbf_expansion_custom_lengthscale():
    """Test RBFExpansion with custom lengthscale"""
    rbf = RBFExpansion(vmin=0, vmax=8, bins=40, lengthscale=0.5)
    
    distances = torch.tensor([1.0, 2.0])
    expanded = rbf(distances)
    
    assert expanded.shape == (2, 40)
    assert isinstance(rbf.gamma, float)


def test_cgcnn_conv(sample_graph_data):
    """Test CGCNNConv layer"""
    node_dim = 64
    edge_dim = 64  # After RBF expansion
    out_dim = 64
    
    conv = CGCNNConv(node_dim, edge_dim, out_dim)
    
    # Expand edge attributes using RBF (as done in the model)
    rbf = RBFExpansion(vmin=0, vmax=8, bins=edge_dim)
    edge_attr_expanded = rbf(sample_graph_data.edge_attr.view(-1))
    
    # Forward pass
    output = conv(sample_graph_data.x, sample_graph_data.edge_index, edge_attr_expanded)
    
    assert output.shape == (3, out_dim)  # Same number of nodes, out_dim features
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_cgcnn_pyg_regression(sample_graph_data):
    """Test CGCNN_PyG model for regression"""
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128,
        robust_regression=False,
        classification=False,
        quantile_regression=False
    )
    
    model.eval()
    with torch.no_grad():
        output = model(sample_graph_data)
    
    assert output.shape == (1, 1)  # Single graph, single output
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_cgcnn_pyg_classification(sample_graph_data):
    """Test CGCNN_PyG model for classification"""
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128,
        classification=True,
        num_classes=3
    )
    
    model.eval()
    with torch.no_grad():
        output = model(sample_graph_data)
    
    assert output.shape == (1, 3)  # Single graph, 3 classes
    assert not torch.isnan(output).any()


def test_cgcnn_pyg_quantile_regression(sample_graph_data):
    """Test CGCNN_PyG model for quantile regression"""
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128,
        quantile_regression=True,
        num_quantiles=3
    )
    
    model.eval()
    with torch.no_grad():
        output = model(sample_graph_data)
    
    assert output.shape == (1, 3)  # Single graph, 3 quantiles
    assert not torch.isnan(output).any()


def test_cgcnn_pyg_robust_regression(sample_graph_data):
    """Test CGCNN_PyG model for robust regression"""
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128,
        robust_regression=True
    )
    
    model.eval()
    with torch.no_grad():
        output = model(sample_graph_data)
    
    assert output.shape == (1, 2)  # Single graph, mean and log_std
    assert not torch.isnan(output).any()


def test_cgcnn_pyg_with_additional_features(sample_graph_data):
    """Test CGCNN_PyG model with additional compound features"""
    add_feat_len = 128
    sample_graph_data.additional_compound_features = torch.randn(1, add_feat_len)
    
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128,
        additional_compound_features=True,
        add_feat_len=add_feat_len
    )
    
    model.eval()
    with torch.no_grad():
        output = model(sample_graph_data)
    
    assert output.shape == (1, 1)  # Single graph, single output
    assert not torch.isnan(output).any()


def test_cgcnn_pyg_batch(sample_batch_data):
    """Test CGCNN_PyG model with batched graphs"""
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128
    )
    
    model.eval()
    with torch.no_grad():
        output = model(sample_batch_data)
    
    assert output.shape == (2, 1)  # Two graphs, single output each
    assert not torch.isnan(output).any()


def test_cgcnn_pyg_extract_crystal_repr(sample_graph_data):
    """Test extract_crystal_repr method"""
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128
    )
    
    model.eval()
    with torch.no_grad():
        repr = model.extract_crystal_repr(sample_graph_data)
    
    assert repr.shape == (1, 64)  # Single graph, atom_fea_len features
    assert not torch.isnan(repr).any()


def test_cgcnn_pyg_extract_crystal_repr_with_features(sample_graph_data):
    """Test extract_crystal_repr with additional features"""
    add_feat_len = 128
    sample_graph_data.additional_compound_features = torch.randn(1, add_feat_len)
    
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128,
        additional_compound_features=True,
        add_feat_len=add_feat_len
    )
    
    model.eval()
    with torch.no_grad():
        repr = model.extract_crystal_repr(sample_graph_data)
    
    assert repr.shape == (1, 64)  # Single graph, atom_fea_len features
    assert not torch.isnan(repr).any()


def test_cgcnn_pyg_different_n_conv(sample_graph_data):
    """Test CGCNN_PyG with different number of convolutional layers"""
    for n_conv in [1, 2, 3]:
        model = CGCNN_PyG(
            orig_atom_fea_len=64,
            edge_feat_dim=64,
            atom_fea_len=64,
            n_conv=n_conv,
            n_h=1,
            h_fea_len=128
        )
        
        model.eval()
        with torch.no_grad():
            output = model(sample_graph_data)
        
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()


def test_cgcnn_pyg_different_n_h(sample_graph_data):
    """Test CGCNN_PyG with different number of hidden layers"""
    for n_h in [1, 2, 3]:
        model = CGCNN_PyG(
            orig_atom_fea_len=64,
            edge_feat_dim=64,
            atom_fea_len=64,
            n_conv=2,
            n_h=n_h,
            h_fea_len=128
        )
        
        model.eval()
        with torch.no_grad():
            output = model(sample_graph_data)
        
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()


def test_cgcnn_pyg_different_feature_dims():
    """Test CGCNN_PyG with different feature dimensions"""
    # Create data with different feature dimensions
    x = torch.randn(3, 86)  # 86 features (ALIGNN standard)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 1)
    batch = torch.zeros(3, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    model = CGCNN_PyG(
        orig_atom_fea_len=86,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data)
    
    assert output.shape == (1, 1)
    assert not torch.isnan(output).any()


def test_cgcnn_conv_gradient_flow(sample_graph_data):
    """Test that gradients flow through CGCNNConv"""
    conv = CGCNNConv(node_dim=64, edge_dim=64, out_dim=64)
    rbf = RBFExpansion(vmin=0, vmax=8, bins=64)
    edge_attr_expanded = rbf(sample_graph_data.edge_attr.view(-1))
    
    x = sample_graph_data.x.clone().requires_grad_(True)
    output = conv(x, sample_graph_data.edge_index, edge_attr_expanded)
    
    # Compute gradient
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_cgcnn_pyg_gradient_flow(sample_graph_data):
    """Test that gradients flow through CGCNN_PyG"""
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128
    )
    
    sample_graph_data.x.requires_grad_(True)
    output = model(sample_graph_data)
    
    # Compute gradient
    loss = output.sum()
    loss.backward()
    
    assert sample_graph_data.x.grad is not None
    assert not torch.isnan(sample_graph_data.x.grad).any()


def test_cgcnn_pyg_empty_graph():
    """Test CGCNN_PyG with empty graph (should handle gracefully)"""
    # Create graph with no edges
    x = torch.randn(2, 64)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 1))
    batch = torch.zeros(2, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    model = CGCNN_PyG(
        orig_atom_fea_len=64,
        edge_feat_dim=64,
        atom_fea_len=64,
        n_conv=2,
        n_h=2,
        h_fea_len=128
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data)
    
    assert output.shape == (1, 1)  # Should still produce output
    assert not torch.isnan(output).any()

