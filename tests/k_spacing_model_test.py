import sys
import os
import pytest

# Skip these tests entirely if optional heavy deps are missing in CI
pytest.importorskip("pymatgen")
pytest.importorskip("matminer")
pytest.importorskip("torch")

from pymatgen.core.structure import Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
import kspacing_model as km  # import module so we can monkeypatch its functions

@pytest.fixture
def sample_structure():
    """Create a sample structure for testing"""
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice,species=species,coords=coords)

def test_predict_kspacing(sample_structure, monkeypatch):
    """Tests the predict_kspacing function API with mocked output"""
    mock_out = (0.2, 0.25, 0.15)

    monkeypatch.setattr(km, 'predict_kspacing',
                        lambda structure, model_name, confidence_level=0.95: mock_out)

    prediction = km.predict_kspacing(sample_structure, 'RF')
    assert len(prediction) == 3
    assert prediction == mock_out


def test_predict_kspacing_with_invalid_structure(monkeypatch):
    """Tests the predict_kspacing function with None structure"""

    def _raise(structure, model_name, confidence_level=0.95):
        if structure is None:
            raise ValueError("structure is None")
        return (0.2, 0.25, 0.15)

    monkeypatch.setattr(km, 'predict_kspacing', _raise)

    with pytest.raises(ValueError):
        km.predict_kspacing(None, 'RF')


