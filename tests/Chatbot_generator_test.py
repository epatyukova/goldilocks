import sys
import os
from streamlit.testing.v1 import AppTest
import pytest
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
def sample_structure():
    """Create a sample structure for testing"""
    # Create a simple SiO2 structure
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice,species=species,coords=coords)

