import sys
import os
import pytest
import torch
import numpy as np

# Skip tests if dependencies are not available
torch = pytest.importorskip("torch")
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
from models.alignn import (
    EdgeGatedGraphConvPyG,
    ALIGNNConvPyG,
    RBFExpansion,
    Standardize,
    ALIGNN_PyG
)


@pytest.fixture
def sample_graph_data():
    """Create sample atomic graph data for testing"""
    x = torch.randn(3, 64)  # 3 nodes, 64 features each
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # 4 edges
    edge_attr = torch.randn(4, 64)  # 4 edges, 64 features each (matching node features)
    batch = torch.zeros(3, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


@pytest.fixture
def sample_line_graph_data():
    """Create sample line graph data for testing"""
    x = torch.randn(4, 64)  # 4 edges in atomic graph, 64 features each
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)  # Line graph edges
    edge_attr = torch.randn(3)  # Angle cosines (scalar for RBF expansion)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def sample_alignn_data(sample_graph_data, sample_line_graph_data):
    """Create sample data for ALIGNN model"""
    return sample_graph_data, sample_line_graph_data


def test_edge_gated_graph_conv_basic(sample_graph_data):
    """Test EdgeGatedGraphConvPyG basic functionality"""
    conv = EdgeGatedGraphConvPyG(in_channels=64, out_channels=64)
    
    output = conv(sample_graph_data.x, sample_graph_data.edge_index, sample_graph_data.edge_attr)
    
    assert output.shape == (3, 64)  # Same number of nodes, out_channels features
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_edge_gated_graph_conv_with_residual(sample_graph_data):
    """Test EdgeGatedGraphConvPyG with residual connection"""
    conv = EdgeGatedGraphConvPyG(in_channels=64, out_channels=64, residual=True)
    
    x_input = sample_graph_data.x.clone()
    output = conv(x_input, sample_graph_data.edge_index, sample_graph_data.edge_attr)
    
    assert output.shape == (3, 64)
    # With residual, output should be different from input (but not too different)
    assert not torch.allclose(output, x_input)


def test_edge_gated_graph_conv_without_residual(sample_graph_data):
    """Test EdgeGatedGraphConvPyG without residual connection"""
    conv = EdgeGatedGraphConvPyG(in_channels=64, out_channels=64, residual=False)
    
    output = conv(sample_graph_data.x, sample_graph_data.edge_index, sample_graph_data.edge_attr)
    
    assert output.shape == (3, 64)
    assert not torch.isnan(output).any()


def test_edge_gated_graph_conv_batch_norm(sample_graph_data):
    """Test EdgeGatedGraphConvPyG with BatchNorm instead of LayerNorm"""
    # BatchNorm1d requires at least 2 samples for stable statistics, so use more nodes
    x = torch.randn(5, 64)  # 5 nodes
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 2, 4]], dtype=torch.long)  # 5 edges
    edge_attr = torch.randn(5, 64)  # 5 edges, 64 features
    
    conv = EdgeGatedGraphConvPyG(in_channels=64, out_channels=64, use_layer_norm=False)
    
    output = conv(x, edge_index, edge_attr)
    
    assert output.shape == (5, 64)
    assert not torch.isnan(output).any()


def test_alignn_conv_pyg(sample_graph_data, sample_line_graph_data):
    """Test ALIGNNConvPyG layer"""
    alignn_conv = ALIGNNConvPyG(hidden_dim=64)
    
    # ALIGNNConvPyG expects:
    # - data_g.x: node features [num_nodes, hidden_dim]
    # - data_g.edge_attr: edge features [num_edges, hidden_dim] (already embedded)
    # - data_lg.x: line graph node features = edge features from atomic graph [num_edges, hidden_dim]
    # - data_lg.edge_attr: embedded angle features [num_line_edges, hidden_dim] (not scalar!)
    
    # Set data_lg.x to match data_g.edge_attr (as done in ALIGNN model)
    sample_line_graph_data.x = sample_graph_data.edge_attr.clone()
    # Set data_lg.edge_attr to have same feature dimension (embedded angles)
    sample_line_graph_data.edge_attr = torch.randn(3, 64)  # 3 line edges, 64 features
    
    x_out, edge_attr_out = alignn_conv(sample_graph_data, sample_line_graph_data)
    
    assert x_out.shape == (3, 64)  # Same number of nodes
    assert edge_attr_out.shape == (4, 64)  # Same number of edges
    assert not torch.isnan(x_out).any()
    assert not torch.isnan(edge_attr_out).any()


def test_rbf_expansion():
    """Test RBFExpansion module"""
    rbf = RBFExpansion(vmin=0, vmax=8, bins=40)
    
    distances = torch.tensor([1.0, 2.0, 3.0])
    expanded = rbf(distances)
    
    assert expanded.shape == (3, 40)  # 3 distances, 40 bins
    assert torch.all(expanded >= 0)  # RBF values should be non-negative
    assert torch.all(expanded <= 1)  # RBF values should be <= 1


def test_rbf_expansion_angle():
    """Test RBFExpansion for angles (cosines in [-1, 1])"""
    rbf = RBFExpansion(vmin=-1.0, vmax=1.0, bins=20)
    
    angles = torch.tensor([-0.5, 0.0, 0.5, 1.0])
    expanded = rbf(angles)
    
    assert expanded.shape == (4, 20)
    assert torch.all(expanded >= 0)
    assert torch.all(expanded <= 1)


def test_standardize(sample_graph_data):
    """Test Standardize module"""
    mean = torch.mean(sample_graph_data.x, dim=0)
    std = torch.std(sample_graph_data.x, dim=0) + 1e-8
    
    standardize = Standardize(mean, std)
    standardized_data = standardize(sample_graph_data)
    
    # Check that data is cloned
    assert standardized_data is not sample_graph_data
    
    # Check that mean is approximately 0
    assert torch.allclose(torch.mean(standardized_data.x, dim=0), torch.zeros_like(mean), atol=1e-5)


def test_alignn_pyg_regression(sample_alignn_data):
    """Test ALIGNN_PyG model for regression"""
    data_g, data_lg = sample_alignn_data
    # Fix edge_attr to be scalar distances (will be expanded by RBF)
    data_g.edge_attr = torch.randn(4)  # Scalar distances for 4 edges
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2,
        classification=False,
        robust_regression=False,
        quantile_regression=False
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (1,)  # Single graph, scalar output
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_alignn_pyg_classification(sample_alignn_data):
    """Test ALIGNN_PyG model for classification"""
    data_g, data_lg = sample_alignn_data
    data_g.edge_attr = torch.randn(4)  # Scalar distances
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2,
        classification=True,
        num_classes=3
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (1, 3)  # Single graph, 3 classes
    assert not torch.isnan(output).any()


def test_alignn_pyg_quantile_regression(sample_alignn_data):
    """Test ALIGNN_PyG model for quantile regression"""
    data_g, data_lg = sample_alignn_data
    data_g.edge_attr = torch.randn(4)  # Scalar distances
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2,
        quantile_regression=True,
        num_quantiles=3
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (1, 3)  # Single graph, 3 quantiles
    assert not torch.isnan(output).any()


def test_alignn_pyg_robust_regression(sample_alignn_data):
    """Test ALIGNN_PyG model for robust regression"""
    data_g, data_lg = sample_alignn_data
    data_g.edge_attr = torch.randn(4)  # Scalar distances
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2,
        robust_regression=True
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (1, 2)  # Single graph, mean and log_std
    assert not torch.isnan(output).any()


def test_alignn_pyg_with_additional_features(sample_alignn_data):
    """Test ALIGNN_PyG model with additional compound features"""
    data_g, data_lg = sample_alignn_data
    data_g.edge_attr = torch.randn(4)  # Scalar distances
    data_g.additional_compound_features = torch.randn(1, 231)  # Standard add_feat_len
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2,
        additional_compound_features=True,
        add_feat_len=231
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (1,)  # Single graph, scalar output
    assert not torch.isnan(output).any()


def test_alignn_pyg_86_features():
    """Test ALIGNN_PyG with 86 atom input features (ALIGNN standard)"""
    data_g = Data(
        x=torch.randn(3, 86),  # 86 features
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.randn(4),
        batch=torch.zeros(3, dtype=torch.long)
    )
    data_lg = Data(
        x=torch.randn(4, 64),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.randn(3)
    )
    
    model = ALIGNN_PyG(
        atom_input_features=86,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (1,)
    assert not torch.isnan(output).any()


def test_alignn_pyg_empty_line_graph():
    """Test ALIGNN_PyG with empty line graph (should handle gracefully)"""
    data_g = Data(
        x=torch.randn(2, 64),
        edge_index=torch.empty((2, 0), dtype=torch.long),  # No edges
        edge_attr=torch.empty(0),
        batch=torch.zeros(2, dtype=torch.long)
    )
    data_lg = Data(
        x=torch.empty(0, 64),
        edge_index=torch.empty((2, 0), dtype=torch.long),  # Empty line graph
        edge_attr=torch.empty(0)
    )
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (1,)  # Should still produce output
    assert not torch.isnan(output).any()


def test_alignn_pyg_different_layer_counts(sample_alignn_data):
    """Test ALIGNN_PyG with different numbers of layers"""
    data_g, data_lg = sample_alignn_data
    # Create fresh data for each iteration to avoid in-place modifications
    for alignn_layers in [1, 2, 4]:
        for gcn_layers in [1, 2, 4]:
            # Create fresh data
            data_g_fresh = Data(
                x=torch.randn(3, 64),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
                edge_attr=torch.randn(4),  # Scalar distances
                batch=torch.zeros(3, dtype=torch.long)
            )
            data_lg_fresh = Data(
                x=torch.randn(4, 64),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                edge_attr=torch.randn(3)  # Angle cosines
            )
            
            model = ALIGNN_PyG(
                atom_input_features=64,
                hidden_features=64,
                edge_input_features=40,
                triplet_input_features=20,
                alignn_layers=alignn_layers,
                gcn_layers=gcn_layers
            )
            
            model.eval()
            with torch.no_grad():
                output = model(data_g_fresh, data_lg_fresh)
            
            assert output.shape == (1,)
            assert not torch.isnan(output).any()


def test_alignn_pyg_batch():
    """Test ALIGNN_PyG with batched graphs"""
    # Create batched data
    data_g = Data(
        x=torch.randn(5, 64),  # 5 nodes total
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 2, 4]], dtype=torch.long),
        edge_attr=torch.randn(5),  # Scalar distances
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)  # 2 graphs
    )
    data_lg = Data(
        x=torch.randn(5, 64),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        edge_attr=torch.randn(4)  # Angle cosines
    )
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (2,)  # Two graphs
    assert not torch.isnan(output).any()


def test_edge_gated_graph_conv_gradient_flow(sample_graph_data):
    """Test that gradients flow through EdgeGatedGraphConvPyG"""
    conv = EdgeGatedGraphConvPyG(in_channels=64, out_channels=64)
    
    x = sample_graph_data.x.clone().requires_grad_(True)
    edge_attr = sample_graph_data.edge_attr.clone().requires_grad_(True)
    
    output = conv(x, sample_graph_data.edge_index, edge_attr)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert edge_attr.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(edge_attr.grad).any()


def test_alignn_pyg_gradient_flow(sample_alignn_data):
    """Test that gradients flow through ALIGNN_PyG"""
    data_g, data_lg = sample_alignn_data
    # Create fresh data with requires_grad to avoid in-place modification issues
    x = torch.randn(3, 64, requires_grad=True)
    edge_attr = torch.randn(4, requires_grad=False)  # Scalar distances, no grad needed
    
    data_g_fresh = Data(
        x=x,
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=edge_attr,
        batch=torch.zeros(3, dtype=torch.long)
    )
    data_lg_fresh = Data(
        x=torch.randn(4, 64),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.randn(3)
    )
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2
    )
    
    output = model(data_g_fresh, data_lg_fresh)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_alignn_conv_pyg_gradient_flow(sample_graph_data, sample_line_graph_data):
    """Test that gradients flow through ALIGNNConvPyG"""
    alignn_conv = ALIGNNConvPyG(hidden_dim=64)
    
    # Create fresh data with requires_grad to avoid in-place modification issues
    # Note: data_g.edge_attr is not used in ALIGNNConvPyG.forward(), only data_g.x is used
    x = torch.randn(3, 64, requires_grad=True)
    lg_x = torch.randn(4, 64, requires_grad=True)  # Line graph nodes = edge features
    lg_edge_attr = torch.randn(3, 64, requires_grad=True)  # Embedded angle features
    
    data_g = Data(
        x=x,
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.randn(4, 64),  # Not used in forward, so no grad needed
        batch=torch.zeros(3, dtype=torch.long)
    )
    data_lg = Data(
        x=lg_x,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=lg_edge_attr
    )
    
    x_out, edge_attr_out = alignn_conv(data_g, data_lg)
    loss = x_out.sum() + edge_attr_out.sum()
    loss.backward()
    
    # Only check gradients for tensors actually used in forward pass
    assert x.grad is not None, "Gradients should flow through data_g.x"
    assert lg_x.grad is not None, "Gradients should flow through data_lg.x"
    assert lg_edge_attr.grad is not None, "Gradients should flow through data_lg.edge_attr"
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(lg_x.grad).any()
    assert not torch.isnan(lg_edge_attr.grad).any()


def test_alignn_pyg_batch_norm_mode(sample_alignn_data):
    """Test ALIGNN_PyG with BatchNorm instead of LayerNorm"""
    data_g, data_lg = sample_alignn_data
    data_g.edge_attr = torch.randn(4)  # Scalar distances
    
    model = ALIGNN_PyG(
        atom_input_features=64,
        hidden_features=64,
        edge_input_features=40,
        triplet_input_features=20,
        alignn_layers=2,
        gcn_layers=2,
        use_layer_norm=False
    )
    
    model.eval()
    with torch.no_grad():
        output = model(data_g, data_lg)
    
    assert output.shape == (1,)
    assert not torch.isnan(output).any()


def test_rbf_expansion_custom_lengthscale():
    """Test RBFExpansion with custom lengthscale"""
    rbf = RBFExpansion(vmin=0, vmax=8, bins=40, lengthscale=0.5)
    
    distances = torch.tensor([1.0, 2.0])
    expanded = rbf(distances)
    
    assert expanded.shape == (2, 40)
    assert isinstance(rbf.gamma, float)
    assert rbf.gamma > 0


def test_edge_gated_graph_conv_different_channels(sample_graph_data):
    """Test EdgeGatedGraphConvPyG with different input/output channels"""
    conv = EdgeGatedGraphConvPyG(in_channels=64, out_channels=32)
    
    output = conv(sample_graph_data.x, sample_graph_data.edge_index, sample_graph_data.edge_attr)
    
    assert output.shape == (3, 32)  # Different output dimension
    assert not torch.isnan(output).any()

