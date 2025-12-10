import os
import sys
import numpy as np
import pytest
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition

# Skip the whole module if heavy deps are unavailable in CI
matminer = pytest.importorskip("matminer")
pytest.importorskip("dscribe")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/qe_input")))
import models.compound_features_utils as cf  # noqa: E402


@pytest.fixture
def sample_structure():
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ["Si", "O", "O"]
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice, species=species, coords=coords)


@pytest.fixture
def sample_df(sample_structure):
    return cf.pd.DataFrame(
        {
            "formula": ["SiO2"],
            "structure": [sample_structure],
        }
    )


def test_normalize_formulas(sample_df):
    df = cf.normalize_formulas(sample_df.copy(), formula_column="formula")
    assert df.loc[0, "formula"] == Composition("SiO2").iupac_formula


def test_matminer_composition_features(sample_df):
    feats = cf.matminer_composition_features(
        sample_df.copy(), ["ElementProperty", "Stoichiometry", "ValenceOrbital"], formula_column="formula"
    )
    assert isinstance(feats, np.ndarray)
    assert feats.shape[0] == 1
    assert np.isfinite(feats).all()


def test_matminer_structure_features(sample_df):
    feats = cf.matminer_structure_features(
        sample_df.copy(), ["GlobalSymmetryFeatures", "DensityFeatures"], structure_column="structure"
    )
    assert isinstance(feats, np.ndarray)
    assert feats.shape[0] == 1
    assert np.isfinite(feats).all()


def test_lattice_features(sample_df):
    feats = cf.lattice_features(sample_df.copy(), structure_column="structure")
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (1, 15)
    assert np.isfinite(feats).all()


def test_soap_features(sample_df):
    feats = cf.soap_features(
        sample_df.copy(),
        soap_params={"r_cut": 5.0, "n_max": 2, "l_max": 2, "sigma": 0.5},
        structure_column="structure",
    )
    assert isinstance(feats, np.ndarray)
    assert feats.shape[0] == 1
    assert np.isfinite(feats).all()

