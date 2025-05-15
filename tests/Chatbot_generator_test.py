import sys
import os
# import numpy as np
from streamlit.testing.v1 import AppTest
import pytest
from unittest.mock import patch #, MagicMock
import streamlit as st
from pymatgen.core.structure import Structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input/pages')))

# from utils import convert_openai_to_gemini, generate_kpoints_grid, atomic_positions_list

def test_sidebar():
    """Testing Openai chatbot page"""
    at = AppTest.from_file("src/qe_input/QE_input.py")
    at.run(timeout=10)
    at.switch_page("pages/Chatbot_generator.py")
    at.run(timeout=10)
    assert not at.exception

    for x in at.sidebar.get('selectbox'):
        if(x.label=='assistant LLM'):
            assert 'gpt-4o' in x.options
            assert 'gpt-4o-mini' in x.options
            assert 'gpt-3.5-turbo' in x.options
            x._value='gpt-3.5-turbo'
            x.run(timeout=10)
            assert at.session_state['llm_name']=='gpt-3.5-turbo'
            at.sidebar.get('text_input')[0]._value='openai_api_key'
            at.run(timeout=10)
            assert at.session_state['openai_api_key'] == 'openai_api_key'
            
            assert 'llama-3.3-70b-versatile' in x.options
            assert 'gemma2-9b-it' in x.options
            x._value='llama-3.3-70b-versatile'
            x.run(timeout=10)
            assert at.session_state['llm_name']=='llama-3.3-70b-versatile'
            at.sidebar.get('text_input')[0]._value='groq_api_key'
            at.run(timeout=10)
            assert at.session_state['groq_api_key'] == 'groq_api_key'
            assert 'gemini-2.0-flash' in x.options

@pytest.fixture
def mock_streamlit():
    with patch('streamlit.sidebar'), \
         patch('streamlit.text_input'), \
         patch('streamlit.selectbox'), \
         patch('streamlit.title'), \
         patch('streamlit.info'), \
         patch('streamlit.markdown'), \
         patch('streamlit.chat_message'), \
         patch('streamlit.chat_input'), \
         patch('streamlit.write_stream'):
        # Create a mock for st.session_state that works like a dict
        mock_session_state = {}
        with patch.object(st, 'session_state', mock_session_state):
            yield

@pytest.fixture
def sample_structure():
    """Create a sample structure for testing"""
    # Create a simple SiO2 structure
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice,species=species,coords=coords)


# Test utils functions
# class TestUtilsFunctions:
#     def test_atomic_positions_list(self):
#         # Create a mock pymatgen Structure object
#         mock_site1 = MagicMock()
#         mock_site1.species_string = "Na"
#         mock_site1.frac_coords = np.array([0.0, 0.0, 0.0])
        
#         mock_site2 = MagicMock()
#         mock_site2.species_string = "Cl"
#         mock_site2.frac_coords = np.array([0.5, 0.5, 0.5])
        
#         mock_structure = MagicMock()
#         mock_structure.sites = [mock_site1, mock_site2]
#         mock_structure.lattice.matrix = np.array([
#             [3.43609631, 0.0, 1.98383169],
#             [1.14536544, 3.23958308, 1.9838317],
#             [0.0, 0.0, 3.96766243]
#         ])
        
#         # Call the function and check results
#         result = atomic_positions_list(mock_structure)
#         assert len(result) == 2
#         # Check that the function returned the right structure
#         assert all(isinstance(item, dict) for item in result)
#         assert all("species" in item and "position" in item for item in result)
#         assert result[0]["species"] == "Na"
#         assert result[1]["species"] == "Cl"

#     def test_generate_kpoints_grid(self):
#         mock_structure = MagicMock()
#         mock_structure.lattice.reciprocal_lattice.abc = (1.0, 2.0, 3.0)
        
#         # Test with default kspacing
#         result = generate_kpoints_grid(mock_structure, 0.5)
#         assert len(result) == 3
#         assert all(isinstance(k, int) for k in result)
        
#         # Test with custom kspacing
#         result = generate_kpoints_grid(mock_structure, 0.3)
#         assert all(result[i] >= int(1.0/(0.3*mock_structure.lattice.reciprocal_lattice.abc[i])) for i in range(3))

#     def test_convert_openai_to_gemini(self):
#         # Test conversion from OpenAI message format to Gemini format
#         openai_messages = [
#             {"role": "system", "content": "You are an assistant"},
#             {"role": "user", "content": "Hello"},
#             {"role": "assistant", "content": "Hi there"},
#             {"role": "user", "content": "How are you?"}
#         ]
        
#         gemini_format = convert_openai_to_gemini(openai_messages)
#         assert "You are an assistant" in gemini_format
#         assert "Hello" in gemini_format
#         assert "Hi there" in gemini_format
#         assert "How are you?" in gemini_format

