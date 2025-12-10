import os
import sys
import pytest
import numpy as np

# Skip if heavy deps are missing in CI
pytest.importorskip("torch")
pytest.importorskip("pymatgen")

from pymatgen.core.structure import Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/qe_input")))
from models.cgcnn_graph import (
    build_radius_cgcnn_graph_from_structure,
    build_crystalnn_cgcnn_graph_from_structure,
)


@pytest.fixture
def sample_structure():
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ["Si", "O", "O"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice, species=species, coords=coords)


@pytest.fixture
def atom_features(sample_structure):
    # simple one-hot-ish features per atom: 3 atoms x 2 dims
    return np.array([[1, 0], [0, 1], [0, 1]])


def _basic_checks(data, n_atoms, feat_dim):
    # node features
    assert data.x.shape == (n_atoms, feat_dim)
    # edge_attr should have 1 column (distance)
    assert data.edge_attr.shape[1] == 1
    # edge_index should have 2 rows
    assert data.edge_index.shape[0] == 2
    # number of edges consistent
    assert data.edge_index.shape[1] == data.edge_attr.shape[0]


def test_build_radius_cgcnn_graph(sample_structure, atom_features):
    data = build_radius_cgcnn_graph_from_structure(
        sample_structure, atom_features, radius=10.0, max_neighbors=12
    )
    _basic_checks(data, n_atoms=3, feat_dim=2)
    # At least some edges should exist for this dense radius
    assert data.edge_index.shape[1] > 0


def test_build_crystalnn_cgcnn_graph(sample_structure, atom_features):
    data = build_crystalnn_cgcnn_graph_from_structure(
        sample_structure, atom_features, radius=10.0
    )
    _basic_checks(data, n_atoms=3, feat_dim=2)
    # CrystalNN should also produce edges
    assert data.edge_index.shape[1] > 0

