
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pymatgen.core.structure import Structure
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
    return StructureLookup(mp_api_key="12345678901234567890123456789012")

@pytest.fixture
def mock_jarvis_df():
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
def mock_mc3d_df():
    """Create a mock MC3D DataFrame for testing"""
    mock_data = {
        'formula_hill': ['O2 Si', 'Al2 O3'],
        'id': ['mc3d-1234-pbe', 'mc3d-5678-pbe'],
        'spacegroup_int': [1, 2]
    }
    return pd.DataFrame(mock_data)

@pytest.fixture
def mock_oqmd_response():
    """Create a mock OQMD API response"""
    return {
        'data': [{
            'entry_id': 1234,
            'delta_e': 0.1,
            'spacegroup': 227,
            'unit_cell': [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            'sites': ['Si @ 0.0 0.0 0.0', 'Si @ 0.5 0.5 0.5']
        }]
    }

def test_get_jarvis_table(structure_lookup, mock_jarvis_df):
    """Test Jarvis table retrieval"""
    with patch('pandas.read_pickle', return_value=mock_jarvis_df):
        result = structure_lookup.get_jarvis_table('SiO2')
        
        # Check basic structure of result
        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]['formula'] == 'O2 Si'
        assert result.iloc[0]['form_energy_per_atom'] == -5.0
        assert result.iloc[0]['id'] == 'JVASP-1234'

def test_get_jarvis_structure_by_id(structure_lookup, mock_jarvis_df):
    """Test retrieving Jarvis structure by ID"""
    with patch('pandas.read_pickle', return_value=mock_jarvis_df):
        structure = structure_lookup.get_jarvis_structure_by_id('JVASP-1234')
        
        # Check structure properties
        assert isinstance(structure, Structure)
        assert structure.formula == 'Si1 O2'
        assert len(structure.sites) == 3

def test_get_mp_structure_table(structure_lookup):
    """Test Materials Project structure table retrieval"""
    mock_docs = MagicMock()
    mock_docs.structure = Structure([[5,0,0],[0,5,0],[0,0,5]], ['Si', 'O', 'O'], [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0]])
    mock_docs.formation_energy_per_atom = -5.0
    mock_docs.material_id = 'mp-1234'
    mock_docs.symmetry.symbol = 'P1'

    mock_mpr = MagicMock()
    mock_mpr.materials.summary.search.return_value = [mock_docs]

    with patch.object(structure_lookup, 'mp_request', return_value=[mock_docs]):
        results = structure_lookup.get_mp_structure_table("SiO2")

    results =  pd.DataFrame(results)
    assert len(results) == 1
    assert isinstance(results, pd.DataFrame)
    assert results.iloc[0]["id"] == "mp-1234"
    assert results.iloc[0]["formula"] == 'O2 Si'


def test_get_mp_structure_by_id(structure_lookup, sample_structure):
    """Test get_mp_structure_by_id"""
    mock_doc = MagicMock()
    mock_doc.structure = sample_structure
        
    mock_mpr = MagicMock()
    mock_mpr.materials.summary.search.return_value = mock_doc

    with patch.object(structure_lookup, 'mp_request_id', return_value=mock_doc):
        structure = structure_lookup.get_mp_structure_by_id("mp-1234")

    assert structure is not None
    assert structure.formula == 'Si1 O2'


def test_get_mc3d_structure_table(structure_lookup, sample_structure):
    """Test get_mc3d_structure_table"""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{
            "id": "mc3d-1234-pbe",
            "attributes": {
                "lattice_vectors": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                "cartesian_site_positions": [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
                "species_at_sites": ['Si', 'O', 'O'],
                "space_group_symbol": "P1"
            }
        }]
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch('requests.get', return_value=mock_response):
        result = structure_lookup.get_mc3d_structure_table('O2 Si')

    assert result is not None
    assert len(result) == 1
    assert result.iloc[0]['formula'] == 'O2 Si'
    assert result.iloc[0]['id'] == 'mc3d-1234-pbe'

def test_get_mc3d_structure_by_id(structure_lookup, sample_structure):
    """Test get_mc3d_structure_by_id"""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "attributes": {
                "lattice_vectors": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                "cartesian_site_positions": [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
                "species_at_sites": ['Si', 'O', 'O']
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch('requests.get', return_value=mock_response):
        structure = structure_lookup.get_mc3d_structure_by_id('mc3d-1234-pbe')

    assert structure is not None
    assert structure.formula == 'Si1 O2'


def test_get_oqmd_structure_table(structure_lookup, mock_oqmd_response):
    """Test get_oqmd_structure_table"""
    mock_response = MagicMock()
    mock_response.content = json.dumps(mock_oqmd_response)

    with patch('requests.get', return_value=mock_response):
        result = structure_lookup.get_oqmd_structure_table('Si2')
        
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0]['formula'] == 'Si2'
    assert result.iloc[0]['form_energy_per_atom'] == 0.1
    assert result.iloc[0]['id'] == 1234
        
def test_get_oqmd_structure_by_id(structure_lookup, mock_oqmd_response):
    """Test get_oqmd_structure_by_id"""
    mock_response = MagicMock()
    mock_response.content = json.dumps(mock_oqmd_response)
    
    with patch('requests.get', return_value=mock_response):
        structure = structure_lookup.get_oqmd_structure_by_id('1234')

    assert structure is not None
    assert isinstance(structure, Structure)

def test_select_structure_from_table(structure_lookup, sample_structure):
    """Test select_structure_from_table"""
     
    test_df = pd.DataFrame({
            'select': [False],
            'formula': ['Si O2'],
            'form_energy_per_atom': [-5.0],
            'sg': ['Cmcm'],
            'natoms': [3],
            'abc': [[5.0, 5.0, 5.0]],
            'angles': [[90.0, 90.0, 90.0]],
            'id': ['test-id']
        })
    
    mock_return_value = pd.DataFrame({
            'select': [True],
            'formula': ['Si O2'],
            'id': ['test-id']
        })

    mock_lookup = MagicMock()
    mock_lookup.return_value = sample_structure
    
    # Mock streamlit components
    with patch('streamlit.data_editor', return_value=mock_return_value), \
         patch('streamlit.selectbox', return_value="leave as is"):
       result = structure_lookup.select_structure_from_table(test_df, mock_lookup)

    # select_structure_from_table now returns a tuple (primitive_structure, structure)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    primitive_structure, structure = result
    assert structure.formula == 'Si1 O2'
    assert primitive_structure.formula == 'Si1 O2'

