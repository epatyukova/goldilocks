import sys
import os
import pytest
from pymatgen.core.structure import Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
from kspacing_model import predict_kspacing

@pytest.fixture
def sample_structure():
    """Create a sample structure for testing"""
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice,species=species,coords=coords)

def test_predict_kspacing(sample_structure):
    """Tests the predict_kspacing function"""
    prediction=predict_kspacing(sample_structure)
    assert len(prediction) == 2
    assert isinstance(prediction[0],float)
    assert isinstance(prediction[1],float)


def test_predict_kspacing_with_invalid_structure():
    """Tests the predict_kspacing function with None structure"""
    with pytest.raises(AttributeError):
        predict_kspacing(None)


