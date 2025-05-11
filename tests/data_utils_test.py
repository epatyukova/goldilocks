import os
import sys
from dotenv import load_dotenv
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))
load_dotenv()


# from data_utils import jarvis_structure_lookup, mp_structure_lookup, mc3d_structure_lookup,oqmd_strucutre_lookup

@pytest.fixture
def formula():
    return 'SiO2'

