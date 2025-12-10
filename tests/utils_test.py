import sys
import os
import pytest
from pymatgen.core.structure import Structure
import tempfile
import json
import math
from unittest.mock import MagicMock
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/qe_input')))

from utils import (  # noqa: E402
    list_of_pseudos,
    cutoff_limits,
    generate_input_file,
    update_input_file,
    atomic_positions_list,
    generate_kpoints_grid,
    convert_openai_to_gemini
    )

@pytest.fixture
def sample_cif():
    CIF="""# generated using pymatgen
        data_CoF2
        _symmetry_space_group_name_H-M   'P 1'
        _cell_length_a   4.64351941
        _cell_length_b   4.64351941
        _cell_length_c   3.19916469
        _cell_angle_alpha   90.00000000
        _cell_angle_beta   90.00000000
        _cell_angle_gamma   90.00000000
        _symmetry_Int_Tables_number   1
        _chemical_formula_structural   CoF2
        _chemical_formula_sum   'Co2 F4'
        _cell_volume   68.98126085
        _cell_formula_units_Z   2
        loop_
        _symmetry_equiv_pos_site_id
        _symmetry_equiv_pos_as_xyz
          1  'x, y, z'
        loop_
        _atom_type_symbol
        _atom_type_oxidation_number
          Co2+  2.0
          F-  -1.0
        loop_
        _atom_site_type_symbol
        _atom_site_label
        _atom_site_symmetry_multiplicity
        _atom_site_fract_x
        _atom_site_fract_y
        _atom_site_fract_z
        _atom_site_occupancy
          Co2+  Co0  1  0.00000000  0.00000000  0.00000000  1
          Co2+  Co1  1  0.50000000  0.50000000  0.50000000  1
          F-  F2  1  0.30433674  0.30433674  0.00000000  1
          F-  F3  1  0.69566326  0.69566326  0.00000000  1
          F-  F4  1  0.80433674  0.19566326  0.50000000  1
          F-  F5  1  0.19566326  0.80433674  0.50000000  1"""
    return CIF

ELEMENTS=['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be',\
       'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co',\
       'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fm', 'Fr',\
       'Ga', 'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir',\
       'K', 'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'N',\
       'Na', 'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa',\
       'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rh',\
       'Rn', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta',\
       'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Xe', 'Y',\
       'Yb', 'Zn', 'Zr']

@pytest.fixture
def sample_structure():
    """Create a sample structure for testing"""
    # Create a simple SiO2 structure
    lattice = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    species = ['Si', 'O', 'O']
    coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    return Structure(lattice=lattice,species=species,coords=coords)

@pytest.fixture
def mock_structure():
    """Create a mock structure that doesn't depend on real file I/O"""
    structure = MagicMock(spec=Structure)
    structure.formula = 'Si1 O2'
    structure.lattice = MagicMock()
    structure.lattice.abc = (5.0, 5.0, 5.0)  # Mock lattice parameters
    structure.lattice.angles = (90.0, 90.0, 90.0)  # Mock lattice angles
    structure.species = ['Si', 'O', 'O']
    structure.frac_coords = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    structure.cart_coords = [[0. , 0. , 0. ], [2.5, 2.5, 2.5], [2.5, 2.5, 0. ]]

    return structure


@pytest.fixture
def mock_json_cutoffs():
    """Mock JSON data for cutoffs"""
    return {
        'Si': {'cutoff_wfc': 40, 'cutoff_rho': 320},
        'O': {'cutoff_wfc': 50, 'cutoff_rho': 400}
    }

@pytest.fixture
def temp_pseudo_folders():
    """Create temporary folders for pseudopotentials and cutoffs"""
    with tempfile.TemporaryDirectory() as pseudo_folder, \
         tempfile.TemporaryDirectory() as cutoffs_folder, \
         tempfile.TemporaryDirectory() as target_folder:
        
        # Create a mock pseudopotentials subfolder
        pbe_efficiency_folder = os.path.join(pseudo_folder, 'pbe_efficiency/')
        os.makedirs(pbe_efficiency_folder)
        
        # Create mock pseudo files
        mock_pseudo_files = {
            'Si': 'Si.pbe-efficiency.UPF',
            'O': 'O.pbe-efficiency.UPF'
        }
        
        for element, filename in mock_pseudo_files.items():
            with open(os.path.join(pbe_efficiency_folder, filename), 'w') as f:
                f.write('Dummy pseudo potential')
        
        # Create a mock cutoffs JSON file
        cutoffs_data = {
            'Si': {'cutoff_wfc': 40, 'cutoff_rho': 320},
            'O': {'cutoff_wfc': 50, 'cutoff_rho': 400}
        }
        cutoffs_file = os.path.join(cutoffs_folder, 'pbe_efficiency_cutoffs.json')
        with open(cutoffs_file, 'w') as f:
            json.dump(cutoffs_data, f)
        
        yield {
            'pseudo_folder': pseudo_folder,
            'cutoffs_folder': cutoffs_folder,
            'target_folder': target_folder,
            'mock_pseudo_files': mock_pseudo_files
        }

# test for list_of_pseudos function which creates a list of pseudo files for compound
def test_list_of_pseudos(temp_pseudo_folders):
    """Test the list_of_pseudos function"""
    pseudo_folder = temp_pseudo_folders['pseudo_folder']
    target_folder = temp_pseudo_folders['target_folder']
    mock_pseudo_files = temp_pseudo_folders['mock_pseudo_files']
    
    # Test successful pseudo file selection
    chosen_subfolder, element_files = list_of_pseudos(
        pseudo_potentials_folder=pseudo_folder+'/',
        functional='PBE',
        mode='efficiency', 
        compound='SiO2',
        target_folder=target_folder+'/'
    )
    
    # Check if subfolder is correctly selected
    assert chosen_subfolder == 'pbe_efficiency'
    
    # Check if correct pseudo files are selected
    for element in ['Si', 'O']:
        assert element in element_files
        assert element_files[element] == mock_pseudo_files[element]

    # Check if files are copied to target folder
    for filename in mock_pseudo_files.values():
        assert os.path.exists(os.path.join(target_folder, filename))


def test_cutoff_limits(temp_pseudo_folders):
    """Test the cutoff_limits function"""
    cutoffs_folder = temp_pseudo_folders['cutoffs_folder']
    
    # Test for SiO2
    limits = cutoff_limits(
        pseudo_potentials_cutoffs_folder=cutoffs_folder+'/',
        functional='PBE',
        mode='efficiency',
        compound='SiO2'
    )
    
    # Check maximum cutoffs
    assert limits['max_ecutwfc'] == 50  # Max of Si (40) and O (50)
    assert limits['max_ecutrho'] == 400  # Max of Si (320) and O (400)


def test_generate_input_file(temp_pseudo_folders, sample_structure):
    """Test the generate_input_file function"""
    with tempfile.TemporaryDirectory() as save_dir, \
         tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as structure_file:
        
        # Save the sample structure to a temporary file
        sample_structure.to(filename=structure_file.name)
        structure_file.close()
        
        # Prepare pseudo file mapping
        pseudo_files = {
            'Si': 'Si.pbe-efficiency.UPF',
            'O': 'O.pbe-efficiency.UPF'
        }
        
        # Generate input file
        input_content = generate_input_file(
            save_directory=save_dir,
            structure_file=structure_file.name,
            pseudo_path_temp=temp_pseudo_folders['pseudo_folder'],
            dict_pseudo_file_names=pseudo_files,
            max_ecutwfc=50,
            max_ecutrho=400,
            kspacing=0.2
        )
        
        # Check input file was created
        qe_input_path = os.path.join(save_dir, 'qe.in')
        assert os.path.exists(qe_input_path)
        assert input_content
        assert len(input_content)>0
      
        ############## # #           #                  #           # # ###############               
        # in the future we can also check that qe.in has the right content and format #
        ############## # #           #                  #           # # ###############

def test_update_input_file(temp_pseudo_folders):
    """Test the update_input_file function"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('Original content')
        temp_file.close()
        
        # Update the file
        update_input_file(temp_file.name, 'New content')
        
        # Check file content
        with open(temp_file.name, 'r') as f:
            assert f.read() == 'New content'

def test_atomic_positions_list(sample_structure):
    """Test atomic_positions_list function"""
    positions_str = atomic_positions_list(sample_structure)
    
    # Split into lines and check expected format
    lines = positions_str.strip().split('\n')
    assert len(lines) == 3  # 3 atoms in the structure
    
    # Basic format checking
    for line in lines:
        parts = line.split()
        assert len(parts) == 4  # Element + 3 coordinates
        assert parts[0] in ['Si', 'O']  # Correct elements

def test_generate_kpoints_grid(sample_structure):
    """Test generate_kpoints_grid function"""
    kspacing = 0.2
    kpoints = generate_kpoints_grid(sample_structure, kspacing, offset=False)
    
    # Check generated kpoints
    assert len(kpoints) == 3  # 3 grid points
    assert all(isinstance(x, int) for x in kpoints)  # Grid points are integers
    
    # Verify calculated grid points
    # ceil(|b| / kdist) where |b| is reciprocal vector norm; for cubic a=5Å, |b|=2π/a≈1.2566
    expected = math.ceil((2 * math.pi / 5) / kspacing)  # ≈7
    assert kpoints[0] == expected
    assert kpoints[1] == expected
    assert kpoints[2] == expected

def test_convert_openai_to_gemini():
    """Test the convert_openai_to_gemini function"""
    openai_prompt = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    gemini_prompt = convert_openai_to_gemini(openai_prompt)
    
    # Check conversion
    assert len(gemini_prompt) == 3
    assert gemini_prompt[0]['role'] == 'user'  # System converted to user
    assert gemini_prompt[0]['parts'][0]['text'] == "You are a helpful assistant"
    assert gemini_prompt[1]['role'] == 'user'
    assert gemini_prompt[2]['role'] == 'model'  # Assistant converted to model