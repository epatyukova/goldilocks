import os
import shutil
import re
import json
from pymatgen.core.composition import Composition
import numpy as np
from typing import Dict
import streamlit as st
from pathlib import Path
from openai import OpenAI
from groq import Groq

def list_of_pseudos(pseudo_potentials_folder: str, 
                    functional: str,
                    mode: str, 
                    compound: str,
                    target_folder: str) -> tuple:
    '''
    Function to determine the list of names of files with pseudopotentials for the compound
    Args:
        pseudo_potentials_folder: str, name of the parent forlder with pseudopotentials
        functional: str, name of the DFT functional
        mode: str, mode for pseudopotential, list of possible values: ["efficiency", "precision"]
        compound: str, composition of the compound
    '''
    list_of_subfolders=os.listdir(pseudo_potentials_folder)
    for subfolder in list_of_subfolders:
        if(re.search(functional.lower()+"_", subfolder.lower()) and re.search(mode.lower(), subfolder.lower())):
            list_of_files=os.listdir(pseudo_potentials_folder+subfolder+"/")
            chosen_subfolder=subfolder
    #print('The list of pseudo files is: ', list_of_files[0], ', ...')
    #print(list_of_files)
    list_of_element_files={}
    for file in list_of_files:
        for element in Composition(compound).elements:
            element=str(element)
            if(file[:len(element)].lower()==element.lower() and not file[len(element):len(element)+1].lower().isalpha()):
                list_of_element_files[element]=file
                shutil.copyfile(pseudo_potentials_folder+chosen_subfolder+"/"+file, target_folder+file)
                
    return chosen_subfolder, list_of_element_files

def cutoff_limits(pseudo_potentials_cutoffs_folder: str, 
                  functional: str,
                  mode: str,
                  compound: str) -> Dict:
    '''
    Function to determine the maximum energy cutoff and density cutoff possible based on cutoff values specified for pseudopotentials
    Args:
        pseudo_potentials_cutoffs: str, the main folder with pseudopotential cutoffs
        functional: str, name of the DFT functional
        mode: str, mode for pseudopotential, list of possible values: ["efficiency", "precision"]
        compound: str, composition of the compound
    Output:
        Dictionary with keys 'max_ecutwfc' and 'max_ecutrho' and float values
    '''
    list_of_cutoff_files=os.listdir(pseudo_potentials_cutoffs_folder)
    for file in list_of_cutoff_files:
        if(re.search(functional.lower()+"_", file.lower()) and re.search(mode.lower(), file.lower())):
            try:
                with open(pseudo_potentials_cutoffs_folder+file, "r") as f:
                    cutoffs=json.load(f)
            except FileNotFoundError:
                cutoffs={}
    elements=[str(el) for el in Composition(compound).elements]
    if(cutoffs!={}):
        subset={key:cutoffs[key] for key in elements}
        encutoffs=[subset[i]['cutoff_wfc'] for i in subset.keys()]
        rhocutoffs=[subset[i]['cutoff_rho'] for i in subset.keys()]
        max_ecutoff=max(encutoffs)
        max_rhocutoff=max(rhocutoffs)
    else:
        max_ecutoff=np.nan
        max_rhocutoff=np.nan
    return { 'max_ecutwfc': max_ecutoff, 'max_ecutrho': max_rhocutoff}

def generate_input_file(save_directory, structure_file, pseudo_path_temp, dict_pseudo_file_names, max_ecutwfc, max_ecutrho, kspacing):
    """
    This function generates the input file for Quantum Espresso for single point energy scf calculations.
    It save the file on disk and prints it out.
    Arguments: generator input of type PW_input_data
    """

    from ase.io.espresso import write_espresso_in
    from pymatgen.core.structure import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    
    pymatgen_structure=Structure.from_file(structure_file)
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_atoms(pymatgen_structure)

    input_data = {
        'calculation': 'scf',
        'restart_mode': 'from_scratch',
        'tprnfor': True,
        'tstress': True,
        'etot_conv_thr': 1e-5,
        'forc_conv_thr': 1e-4,
        'max_seconds': 3.42e3,
        'ecutwfc': int(max_ecutwfc),
        'ecutrho': int(max_ecutrho),
        'occupations': 'smearing',
        'degauss': 0.01,
        'smearing': 'cold',
        'conv_thr': 1e-10,
        'electron_maxstep': 80,
        'mixing_mode': 'plain',
        'mixing_beta': 0.4
    }
    save_directory = Path(save_directory)
    filename = save_directory / 'qe.in'
    write_espresso_in(str(filename), structure, input_data=input_data, pseudopotentials=dict_pseudo_file_names, 
                      kpts=generate_kpoints_grid(pymatgen_structure, kspacing), format='espresso-in')
    input_file_content=''
    with open(str(filename),'r') as file:
        for line in file:
            input_file_content+=line
            if('&CONTROL' in line):
                indent=3
                key='pseudo_dir'
                value=pseudo_path_temp
                input_file_content+=f"{' ' * indent}{key:16} = '{value}'\n"
    with open(str(filename),'w') as file:
        file.write(input_file_content)
    input_file_content=input_file_content[:-1]
    return input_file_content

def update_input_file(file_path: str, new_content: str) -> None:
    """The function to update the content of input file
       Input: file_path the location of the file to be update
              new_content new content to write in the file
    """
    with open(file_path,'w') as file:
       file.write(new_content)
    st.write('qe.in file was updated')
    return

def atomic_positions_list(structure):
    """
    Convert the atomic positions of a structure to a string
    Args:
        structure: pymatgen.core.structure.Structure
    Returns:
        string: string of atomic positions
    """
    string=""
    for site in structure.sites:
        string+=site.as_dict()['species'][0]['element']+' '+str(site.coords[0])+\
        ' '+str(site.coords[1])+' '+str(site.coords[2])+'\n'
    return string

def generate_kpoints_grid(structure, kdist, offset = False):
    """
    Compute the maximum distance between k-points (kdist) from a given mesh and reciprocal cell.
 
    Input:
    kdist: kpoints spacing in 1/Å
    structure: pymatgen structure
    Output:
    calculated k-point grid: list of integers for kpoints grid
    """
    recip_lattice = structure.lattice.reciprocal_lattice
    g1, g2, g3 = recip_lattice.matrix
    norms = np.linalg.norm([g1, g2, g3], axis=1)  
    kmesh = []
   
    for bn in norms:
        kmesh.append(int(np.ceil(bn/kdist)))
    
    if offset:
        kmesh.extend([0,0,0])
        
    return kmesh


def create_task(structure,kspacing,list_of_element_files, cutoffs):
    """
    Create a task for the agent
    Args:
        structure: pymatgen.core.structure.Structure
        kspacing: float
        list_of_element_files: list
        cutoffs: dict
    Returns:
        tuple: input_file_schema, task
    """
    input_file_schema="Below is the QE input file for SCF calculations for NaCl. Can you generate the \
                    similar one for my compound for which I will give parameters? \
                    Check line by line that only material parameters are different.\
                    &CONTROL\
                    pseudo_dir       = './'\
                    calculation      = 'scf'\
                    restart_mode     = 'from_scratch'\
                    tprnfor          = .true.\
                    /\
                    &SYSTEM\
                    ecutwfc          = 40  ! put correct energy cutoff here\
                    ecutrho          = 320 ! put correct density cutoff here\
                    occupations      = 'smearing'\
                    degauss          = 0.01 ! you can change the number\
                    smearing         = 'cold' ! choose correct smearing method\
                    ntyp             = 2 ! put correct number of atoms types\
                    nat              = 2 ! put correct number of atoms\
                    ibrav            = 0\
                    /\
                    &ELECTRONS\
                    electron_maxstep = 80\
                    conv_thr         = 1e-10\
                    mixing_mode      = 'plain'\
                    mixing_beta      = 0.4\
                    / \
                    ATOMIC_SPECIES \
                    Na 22.98976928 na_pbe_v1.5.uspp.F.UPF \
                    Cl 35.45 cl_pbe_v1.4.uspp.F.UPF \
                    K_POINTS automatic\
                    9 9 9  0 0 0\
                    CELL_PARAMETERS angstrom\
                    3.43609630987442 0.00000000000000 1.98383169159751\
                    1.14536543840311 3.23958308210503 1.98383169547732\
                    0.00000000000000 0.00000000000000 3.96766243000000\
                    ATOMIC_POSITIONS angstrom \
                    Na 0.0000000000 0.0000000000 0.0000000000\
                    Cl 2.2907350089 1.6197900184 3.9676599923\
                     "
    
    cell_params=structure.lattice.matrix
    atomic_positions=atomic_positions_list(structure)
    kpoints=generate_kpoints_grid(structure, kspacing, offset=True)

    task=f"You are the assitant for generation input file for single point \
              energy calculations with Quantum Espresso. If the user asks to generate an input file, \
              the following information is availible to you: \
              the formula of the compound {structure.formula},\
              the list of pseudo potential files {list_of_element_files},\
              the path to pseudo potential files './',\
              the cell parameters in angstroms {cell_params},\
              the atomic positions in angstroms {atomic_positions},\
              the energy cutoff is {cutoffs[ 'max_ecutwfc']} in Ry,\
              the density cutoff is {cutoffs[ 'max_ecutrho']} in Ry,\
              kpoints automatic are {kpoints}, \
              number of atoms is {len(structure.sites)} \
              Please calculate forces, and do gaussian smearing for dielectrics and semiconductors \
              and cold smearing for metals.  Try to assess whether the provided compound is \
              metal, dielectric or semiconductor before generation."
    
    return input_file_schema, task

def generate_response(messages,client,llm_model):
    """Generator function to stream response from Groq API
    Args:
        messages: list
        client: Groq
        llm_model: str
    Returns:
        generator: generator of response
    """
    response = client.chat.completions.create(
        model=llm_model,  # Example model, change as needed
        messages=messages,
        temperature=0,
        stream=True  # Enable streaming
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def convert_openai_to_gemini(openai_prompt):
    """Converts prompt in openai format to gemini format (no 'system' role)
    Args:
        openai_prompt: list
    Returns:
        gemini_prompt: list
    """
    gemini_prompt = []  # List of message objects

    for message in openai_prompt:
        role = message["role"]
        content = message["content"]

        # Convert OpenAI roles to Gemini-compatible ones
        if role == "system":
            role = "user"  # Gemini doesn't support "system", use first user message
        elif role == "assistant":
            role = "model"  # Gemini uses "model" instead of "assistant"

        gemini_prompt.append({"role": role, "parts": [{"text": content}]})

    return gemini_prompt

def gemini_stream_to_streamlit(gemini_stream):
    """Stream response from Gemini API to Streamlit
    Args:
        gemini_stream: generator
    Returns:
        generator: generator of response
    """
    for chunk in gemini_stream:
        yield chunk.candidates[0].content.parts[0].text

def create_client(llm_name, api_key):
    """Create a client for the LLM
    Args:
        llm_name: str
        api_key: str
    Returns:
        client: client for the LLM
    """
    if llm_name in ["gpt-4o", "gpt-4o-mini", 'gpt-3.5-turbo']:
        return OpenAI(api_key=api_key)
    elif llm_name in ['llama-3.3-70b-versatile']:
        return Groq(api_key=api_key)
    

def generate_llm_response(llm_name, messages, client):
    """Generate a response from the LLM
    Args:
        llm_name: str
        messages: list
        client: client for the LLM
    Returns:
        generator: generator of response
    """
    if llm_name in ["gpt-4o", "gpt-4o-mini", 'gpt-3.5-turbo']:
        return client.chat.completions.create(
            model=llm_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=0,
            stream=True,
        )
    elif llm_name in ['llama-3.3-70b-versatile']:
        return generate_response(messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                                       client=client,
                                       llm_model=llm_name)
    