import sys
import os
from streamlit.testing.v1 import AppTest
import pytest
from pymatgen.core.structure import Structure
from unittest.mock import patch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input/pages')))

from utils import generate_llm_response

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
            x._value='gemini-2.0-flash'
            x.run(timeout=10)
            assert at.session_state['llm_name']=='gemini-2.0-flash'
            at.sidebar.get('text_input')[0]._value='gemini_api_key'
            at.run(timeout=10)
            assert at.session_state['gemini_api_key'] == 'gemini_api_key'


@pytest.fixture
def sample_structure():
    """Create a sample structure for testing"""
    # Create a simple SiO2 structure
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice,species=species,coords=coords)

@pytest.fixture
def fake_openai_stream():
    def _generator():
        yield {
            "choices": [{
                "delta": {"content": "Hello"},
                "index": 0,
                "finish_reason": None
            }]
        }
        yield {
            "choices": [{
                "delta": {"content": " world"},
                "index": 0,
                "finish_reason": None
            }]
        }
        yield {
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }]
        }
    return _generator()

def test_messages_in_session_state(sample_structure, fake_openai_stream):
    at = AppTest.from_file("src/qe_input/pages/Chatbot_generator.py")
    at.run(timeout = 10)
    
    at.session_state['structure'] = sample_structure
    at.session_state['kspacing'] = 0.02
    at.session_state['list_of_element_files'] = {'O':'O.upf','Si': 'Si.upf'}
    at.session_state['cutoffs'] = {'max_ecutwfc': 40, 'max_ecutrho': 320}
    at.session_state['all_info'] = True
    
    assert 'messages' not in at.session_state
    at.sidebar.get('selectbox')[0]._value = 'gpt-3.5-turbo'
    at.run(timeout = 10)
    at.get('text_input')[0]._value = 'api_key'
    at.run(timeout = 10)
    assert not at.exception
    assert at.session_state['openai_api_key'] == 'api_key'
    assert at.session_state['messages']
    assert 'system' in [m['role'] for m in at.session_state['messages']]
    assert 'user' not in [m['role'] for m in at.session_state['messages']]
    assert 'assistant' not in [m['role'] for m in at.session_state['messages']]

    at.get('chat_input')[0]._value = 'Hello!'
    assert 'user' not in [m['role'] for m in at.session_state['messages']]

    with patch("utils.generate_llm_response", return_value=fake_openai_stream):
        at.run(timeout = 10)
        assert 'user' in [m['role'] for m in at.session_state['messages']]
        assert 'assistant' in [m['role'] for m in at.session_state['messages']]


