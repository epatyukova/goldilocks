import os
import streamlit as st
import shutil
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.io.cif import CifWriter
from utils import list_of_pseudos, cutoff_limits
from data_utils import StructureLookup  # your class here
from kspacing_model import predict_kspacing

# Next step button
@st.fragment
def next_step():
    st.info('Next choose how to generate an input file:')
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Chatbot generator"):
            st.switch_page("pages/Chatbot_generator.py")
    with col4:
        if st.button("Deterministic generator"):
            st.switch_page("pages/Deterministic_generator.py")

# Intro page
st.write("# Welcome to QE input generator! 👋")
st.markdown("This app will help you generate input files for Quantum Espresso calculations.")
st.sidebar.success("Provide specifications and select a way to generate input")

st.session_state['all_info'] = False
structure = None

# Sidebar for selecting the functional and mode
col1, col2 = st.columns(2)
with col1:
    functional_value = st.selectbox('XC-functional', ('PBE', 'PBEsol'), index=None, placeholder='PBE')
    mode_value = st.selectbox('pseudopotential flavour', ('efficiency', 'precision'), index=None, placeholder='efficiency')
with col2:
    kspacing_model = st.selectbox('ML model to predict kspacing', ('CGCNN',), index=None, placeholder='CGCNN')

st.session_state['functional'] = functional_value or 'PBE'
st.session_state['mode'] = mode_value or 'efficiency'
st.session_state['kspacing_model'] = kspacing_model or 'CGCNN'

tab1, tab2 = st.tabs(["Upload structure", "Search for structure"])
with tab1:
    structure_file = st.file_uploader("Upload the structure file", type=("cif"))

with tab2:
    input_formula = st.text_input("Chemical formula (try to find structure in free databases)")

if not structure_file and not input_formula:
    st.info("Please add your structure file or chemical formula to continue")

elif input_formula:
    composition = Composition(input_formula)
    formula, _ = composition.get_reduced_formula_and_factor()

    structure_database = st.radio('Choose the database to search for the structure',
                                  options=['Jarvis', 'MP', 'MC3D', 'OQMD'],
                                  horizontal=True)
    
    if structure_database == 'Jarvis':
        lookup = StructureLookup() 
        try:
            result = lookup.get_jarvis_table(formula)
            structure = lookup.select_structure_from_table(result, lookup.get_jarvis_structure_by_id)
        except Exception as exc:
            st.error(f'Error: {exc}')
    
    elif structure_database == 'MP':
        mp_api_key = st.text_input("Materials Project API Key", key="mp_api_key", type="password")
        if mp_api_key:
            lookup = StructureLookup(mp_api_key)
            try:
                result = lookup.get_mp_structure_table(formula)
                structure = lookup.select_structure_from_table(result, lookup.get_mp_structure_by_id)
            except Exception as exc:
                st.error(f"Error: {exc}")

    elif structure_database == 'MC3D':
        lookup = StructureLookup() 
        try:
            result = lookup.get_mc3d_structure_table(formula)
            structure = lookup.select_structure_from_table(result, lookup.get_mc3d_structure_by_id)
        except Exception as exc:
            st.error(f'Error: {exc}')

    elif structure_database == 'OQMD':
        lookup = StructureLookup() 
        try:
            result = lookup.get_oqmd_structure_table(formula)
            structure = lookup.select_structure_from_table(result, lookup.get_oqmd_structure_by_id)
        except Exception as exc:
            st.error(f'Error: {exc}')

elif structure_file:
    temp_dir = "./src/qe_input/temp/"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, 'structure.cif')
    with open(file_path, "wb") as f:
        f.write(structure_file.getbuffer())
    structure = Structure.from_file(file_path)

if structure:
    save_directory = "./src/qe_input/temp/"
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory, ignore_errors=True)
    os.makedirs(save_directory)
    file_path = os.path.join(save_directory, 'structure.cif')
    write_cif=CifWriter(structure)
    write_cif.write_file(file_path)
    st.session_state['save_directory']=save_directory
    st.session_state['structure_file']=file_path
    st.session_state['structure']=structure

    composition = Composition(structure.alphabetical_formula)
    st.session_state['composition']=structure.alphabetical_formula
    pseudo_path="./src/qe_input/pseudos/"
    pseudo_family, list_of_element_files=list_of_pseudos(pseudo_path, st.session_state['functional'], 
                                                         st.session_state['mode'], composition,st.session_state['save_directory'])
    st.session_state['pseudo_family']=pseudo_family
    st.session_state['list_of_element_files']=list_of_element_files
    st.session_state['pseudo_path']=pseudo_path

    cutoffs=cutoff_limits('./src/qe_input/pseudo_cutoffs/', st.session_state['functional'],
                          st.session_state['mode'], composition)
    st.session_state['cutoffs']=cutoffs

    if(st.session_state['kspacing_model']=='CGCNN'):
        klength, klength_std=predict_kspacing(structure)
    
    kspacing = round(1 / klength , 4)
    st.session_state['kspacing']=kspacing
    st.session_state['klength']=klength
    st.session_state['klength_std']=klength_std
    st.session_state['all_info']=True
    
    next_step()

