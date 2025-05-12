
import os
import pytest
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from bs4 import BeautifulSoup
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
from data_utils import StructureLookup

# from data_utils import jarvis_structure_lookup, mp_structure_lookup, mc3d_structure_lookup,oqmd_strucutre_lookup

@pytest.fixture
def sample_formula():
    return 'SiO2'

@pytest.fixture
def sample_structure():
    """Create a sample structure for testing"""
    # Create a simple SiO2 structure
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice,species=species,coords=coords)

@pytest.fixture
def structure_lookup():
    """Create a StructureLookup instance for testing"""
    return StructureLookup(mp_api_key='test_key')

@pytest.fixture
def mock_jarvis_dataframe():
    """Create a mock Jarvis DataFrame for testing"""
    mock_data = {
        'formula': ['O2 Si', 'Al2 O3'],
        'jid': ['JVASP-1234', 'JVASP-5678'],
        'formation_energy_peratom': [-5.0, -4.5],
        'spg_symbol': ['P1', 'P2'],
        'atoms': [
            {
                'lattice_mat': [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                'elements': ['Si', 'O', 'O'],
                'coords': [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
            },
            {
                'lattice_mat': [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
                'elements': ['Al', 'Al', 'O', 'O', 'O'],
                'coords': [[0, 0, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.25, 0.75, 0.75]]
            }
        ]
    }
    return pd.DataFrame(mock_data)

@pytest.fixture
def mock_mc3d_dataframe():
    """Create a mock MC3D DataFrame for testing"""
    mock_data = {
        'formula_hill': ['SiO2', 'Al2O3'],
        'id': ['mc3d-1234-pbe', 'mc3d-5678-pbe'],
        'spacegroup_int': [1, 2]
    }
    return pd.DataFrame(mock_data)

def test_get_jarvis_table(structure_lookup, mock_jarvis_dataframe):
    """Test Jarvis table retrieval"""
    with patch('pandas.read_pickle', return_value=mock_jarvis_dataframe):
        result = structure_lookup.get_jarvis_table('SiO2')
        
        # Check basic structure of result
        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]['formula'] == 'O2 Si'
        assert result.iloc[0]['form_energy_per_atom'] == -5.0

def test_get_jarvis_structure_by_id(structure_lookup, mock_jarvis_dataframe):
    """Test retrieving Jarvis structure by ID"""
    with patch('pandas.read_pickle', return_value=mock_jarvis_dataframe):
        structure = structure_lookup.get_jarvis_structure_by_id('JVASP-1234')
        
        # Check structure properties
        assert isinstance(structure, Structure)
        assert structure.formula == 'Si1 O2'
        assert len(structure.sites) == 3

# def test_get_mp_structure_table(structure_lookup):
#     """Test Materials Project structure table retrieval"""
#     mock_docs = MagicMock()
#     mock_docs.structure = Structure([[5,0,0],[0,5,0],[0,0,5]], ['Si', 'O', 'O'], [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0]])
#     mock_docs.formation_energy_per_atom = -4.5
#     mock_docs.material_id = 'mp-1234'
#     mock_docs.symmetry.symbol = 'P1'
    
#     with patch('mp_api.client.MPRester.materials.summary.search', return_value=[mock_docs]):
#         result = structure_lookup.get_mp_structure_table('SiO2')
        
#         # Check basic structure of result
#         assert not result.empty
#         assert len(result) == 1
#         assert result.iloc[0]['form_energy_per_atom'] == -4.5