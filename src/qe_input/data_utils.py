import json
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from mp_api.client import MPRester
from typing import Callable, Optional
import streamlit as st
import requests
from bs4 import BeautifulSoup


class StructureLookup:
    """
    Class for looking up structures in databases
    Args:
        mp_api_key: str
    """
    def __init__(self, mp_api_key = None):
       self.mp_api_key = mp_api_key
    
    def optimade_formula(self, formula: str) -> str:
        comp = Composition(formula)
        items = sorted(comp.get_el_amt_dict().items(), key=lambda x: x[0])  # alphabetical

        result = ""
        for el, amt in items:
            if amt == 1 or amt == 1.0:
                result += f"{el}"
            else:
                if float(amt).is_integer():
                    result += f"{el}{int(amt)}"
                else:
                    result += f"{el}{amt}"
        return result

    def get_jarvis_table(self, formula):
        """
        Get the Jarvis table for a formula
        Args:
            formula: str
        Returns:
            pd.DataFrame: DataFrame of the Jarvis table
        """
        df = pd.read_pickle('./src/qe_input/Jarvis.pkl')
        comp = Composition(formula)
        formula = comp.hill_formula
        da = df.loc[df['formula'] == formula].reset_index(drop=True)

        if da.empty:
            return pd.DataFrame()

        rows = []
        for _, row in da.iterrows():
            atoms = row['atoms']
            structure = Structure(
                lattice = atoms['lattice_mat'],
                species = atoms['elements'],
                coords = atoms['coords'],
                coords_are_cartesian = True
            )
            spacegroup = structure.get_space_group_info(symprec=0.01, angle_tolerance=5.0)[0]
            structure = structure.get_reduced_structure()

            rows.append({
                'select': False,
                'formula': Composition(structure.formula).hill_formula,
                'form_energy_per_atom': row['formation_energy_peratom'],
                'sg': spacegroup,
                'sg_jarvis': row['spg_symbol'],
                'natoms': structure.num_sites,
                'abc': [round(x, 2) for x in structure.lattice.abc],
                'angles': [round(x, 1) for x in structure.lattice.angles],
                'id': row['jid']
            })

        result = pd.DataFrame(rows)
        result = result.sort_values(by='form_energy_per_atom').reset_index(drop=True)
        return result

    def get_jarvis_structure_by_id(self, jid):
        """
        Get the Jarvis structure by id
        Args:
            jid: str
        Returns:
            pymatgen.core.structure.Structure: Structure of the Jarvis structure
        """
        df = pd.read_pickle('./src/qe_input/Jarvis.pkl')
        da = df.loc[df['jid'] == jid]
        if da.empty:
            return None
        atoms = da.iloc[0]['atoms']
        return Structure(lattice=atoms['lattice_mat'],
                         species=atoms['elements'],
                         coords=atoms['coords'],
                         coords_are_cartesian=True)
    
    def mp_request(self,formula):
        """
        Get the MP docs by formula
        Args:
            formula: str
        Returns:
            list: list of MP docs
        """
        with MPRester(self.mp_api_key) as mpr:
            docs = mpr.materials.summary.search(formula=formula)
        return docs

    def get_mp_structure_table(self, formula):
        """
        Get the MP structure table by formula
        Args:
            formula: str
        Returns:
            pd.DataFrame: DataFrame of the MP structure table
        """
        docs=self.mp_request(formula)

        if not docs:
            return pd.DataFrame()

        rows = []
        for doc in docs:
            structure = doc.structure
            spacegroup = structure.get_space_group_info(symprec=0.01, angle_tolerance=5.0)[0]
            structure = structure.get_reduced_structure()

            rows.append({
                "select": False,
                "formula": Composition(structure.formula).hill_formula,
                "form_energy_per_atom": doc.formation_energy_per_atom,
                "sg": spacegroup,
                "sg_mp": doc.symmetry.symbol,  # Not available from MP
                "natoms": structure.num_sites,
                "abc": [round(x, 2) for x in structure.lattice.abc],
                "angles": [round(x, 1) for x in structure.lattice.angles],
                "id": doc.material_id
            })

        result = pd.DataFrame(rows)
        result = result.sort_values(by='form_energy_per_atom').reset_index(drop=True)
        return result
    
    def mp_request_id(self,material_id):
        """
        Get the MP doc by id
        Args:
            material_id: str
        Returns:
            MPDoc: MP doc
        """
        with MPRester(self.mp_api_key) as mpr:
            doc = mpr.materials.summary.get_data_by_id(material_id)
        return doc

    def get_mp_structure_by_id(self, material_id):
        """
        Get the MP structure by id
        Args:
            material_id: str
        Returns:
            pymatgen.core.structure.Structure: Structure of the MP structure
        """
        doc = self.mp_request_id(material_id)
        return doc.structure if doc else None

    def get_mc3d_structure_table(self, formula):
        """
        Query MC3D OPTIMADE endpoint and return a table of matching structures.
        """
        base_url = "https://optimade.materialscloud.org/main/mc3d-pbesol-v1/v1/structures"

        # Convert to OPTIMADE alphabetical format, e.g. "SiO2" → "O2Si"
        formula_opt = self.optimade_formula(formula)

        filter_expr = f'chemical_formula_reduced="{formula_opt}"'
        params = {
            "filter": filter_expr,
            "page_limit": 50,
            "response_format": "json",
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        entries = data.get("data", [])
        if not entries:
            return pd.DataFrame()

        rows = []
        for entry in entries:
            attr = entry["attributes"]
            lattice = attr["lattice_vectors"]           # 3×3 matrix
            cart_coords = attr["cartesian_site_positions"]
            species = attr["species_at_sites"]

            structure = Structure(
                lattice=lattice,
                species=species,
                coords=cart_coords,
                coords_are_cartesian=True
            )
            spacegroup = structure.get_space_group_info(symprec=0.01, angle_tolerance=5.0)[0]

            reduced_structure = structure.get_reduced_structure()

            rows.append({
                "select": False,
                "formula": Composition(reduced_structure.formula).hill_formula,
                "form_energy_per_atom": "-",  # MC3D does not provide formation energy
                "sg": spacegroup,
                "sg_mc3d": attr.get("space_group_symbol", "-"),
                "natoms": reduced_structure.num_sites,
                "abc": [round(x, 2) for x in reduced_structure.lattice.abc],
                "angles": [round(x, 1) for x in reduced_structure.lattice.angles],
                "id": entry["id"],
            })

        df = pd.DataFrame(rows)
        return df.reset_index(drop=True)

    
    def get_mc3d_structure_by_id(self, material_id):
        """
        Fetch a single MC3D structure from OPTIMADE by ID.
        """
        base_url = f"https://optimade.materialscloud.org/main/mc3d-pbesol-v1/v1/structures/{material_id}"

        response = requests.get(base_url)
        response.raise_for_status()
        entry = response.json().get("data")

        if entry is None:
            return None

        attr = entry["attributes"]

        structure = Structure(
            lattice=attr["lattice_vectors"],
            species=attr["species_at_sites"],
            coords=attr["cartesian_site_positions"],
            coords_are_cartesian=True,
        )
        return structure

    
    def get_oqmd_structure_table(self, formula):
        """
        Get the OQMD structure table by formula
        Args:
            formula: str
        Returns:
            pd.DataFrame: DataFrame of the OQMD structure table
        """
        response = requests.get(f"http://oqmd.org/oqmdapi/formationenergy?composition={formula}&limit=50&")
        soup = BeautifulSoup(response.content, 'html.parser')
        content = json.loads(soup.text)

        if not content['data']:
            return None
        
        da = pd.DataFrame(content['data'])

        rows = []
        for _, row in da.iterrows():
            species=[]
            coords=[]
            for line in row['sites']:
                el, merged_coord = line.split(' @ ')
                species.append(el)
                x,y,z=merged_coord.split(' ')
                coords.append([float(x),float(y),float(z)])
            structure=Structure(lattice=row['unit_cell'],species=species,coords=coords)
            spacegroup = structure.get_space_group_info(symprec=0.01, angle_tolerance=5.0)[0]
            structure = structure.get_reduced_structure()

            rows.append({
                'select': False,
                "formula": Composition(structure.formula).hill_formula,
                'form_energy_per_atom': row['delta_e'],
                'sg': spacegroup,
                'sg_oqmd': row['spacegroup'],
                'natoms': structure.num_sites,
                'abc': [round(x, 2) for x in structure.lattice.abc],
                'angles': [round(x, 1) for x in structure.lattice.angles],
                'id': row['entry_id']
            })

        result = pd.DataFrame(rows)
        result = result.sort_values(by='form_energy_per_atom').reset_index(drop=True)
        return result
    
    def get_oqmd_structure_by_id(self, material_id):
        """
        Get the OQMD structure by id
        Args:
            material_id: str
        Returns:
            pymatgen.core.structure.Structure: Structure of the OQMD structure
        """
        response = requests.get(f"http://oqmd.org/oqmdapi/entry/{material_id}")
        soup = BeautifulSoup(response.content, 'html.parser')
        content = json.loads(soup.text)
        
        if not content:
            return None
        
        species=[]
        coords=[]
        for line in content['data'][0]['sites']:
            el, merged_coord = line.split(' @ ')
            species.append(el)
            x,y,z=merged_coord.split(' ')
            coords.append([float(x),float(y),float(z)])
        structure=Structure(lattice=content['data'][0]['unit_cell'],species=species,coords=coords)
        return structure

    def select_structure_from_table(self, result_df, id_lookup_func: Callable[[str], Structure]) -> Optional[Structure]:
        """
        Select a structure from a table and modify it (niggli reduced cell, primitive, supercell)
        Args:
            result_df: pd.DataFrame
            id_lookup_func: Callable[[str], Structure]
        Returns:
            pymatgen.core.structure.Structure: Structure of the selected structure
        """
        if result_df.empty:
            st.info('No structure found for this formula.')
            return None

        selected_row = st.data_editor(
            result_df,
            column_config={
                "select": st.column_config.CheckboxColumn("Which structure?", default=False),
            },
            disabled=["formula", "form_energy_per_atom", "sg", "natoms", "abc", "angles"],
            hide_index=True,
        )

        selected = selected_row[selected_row["select"]]
        if len(selected) == 1:
            struct_id = selected["id"].values[0]
            structure = id_lookup_func(struct_id)
            st.success("Structure selected.")

            unit_cell = st.selectbox(
                "Transform unit cell",
                ("leave as is", "niggli reduced cell", "primitive", "supercell"),
                index=None,
                placeholder="leave as is",
            )
            
            primitive_structure=structure.get_primitive_structure()

            if unit_cell == "niggli reduced cell":
                structure = structure.get_reduced_structure()
            elif unit_cell == "primitive":
                structure = structure.get_primitive_structure()
            elif unit_cell == "supercell":
                multi = st.text_input("Multiplication factor (na,nb,nc)", placeholder="(2,2,2)")
                try:
                    multi = tuple(map(int, multi.strip("()").split(",")))
                    structure.make_supercell(multi)
                    st.info("Supercell created.")
                except Exception:
                    st.info("Specify supercell.")
            
            return primitive_structure, structure

        elif len(selected) == 0:
            st.info("Please select a structure.")
        else:
            st.info("Please select only one structure.")

        return None
