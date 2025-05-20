import sys
import os
from streamlit.testing.v1 import AppTest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input/pages')))

def test_parent_page():
    """Testing colling the pages"""
    at = AppTest.from_file("src/qe_input/QE_input.py")
    at.run(timeout=10)
    assert not at.exception
    at.switch_page("pages/Documentation.py")
    at.run(timeout=10)
    assert not at.exception
    at.switch_page("pages/Intro.py")
    at.run(timeout=10)
    assert not at.exception
    at.switch_page("pages/Chatbot_generator.py")
    at.run(timeout=10)
    assert not at.exception
    at.switch_page("pages/Deterministic_generator.py")
    at.run(timeout=10)
    assert not at.exception