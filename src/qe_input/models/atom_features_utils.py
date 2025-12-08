import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from typing import Dict
import json


def load_atom_features(atom_init_path: str) -> Dict:
    """Load atomic embedding file (traditionally keys are atomic numbers)"""
    with open(atom_init_path, 'r') as f:
        data = json.load(f)
    return data


def atomic_soap_features(structure, soap_params):
    """Produce SOAP features for a structure assuming that the structure is a single element
       (so it is a lattice representation)
    """
    from dscribe.descriptors import SOAP
    from pymatgen.io.ase import AseAtomsAdaptor
    desc = SOAP(
                    species=['X'],  # or whatever elements you're using
                    r_cut=soap_params['r_cut'],
                    n_max= soap_params['n_max'],
                    l_max= soap_params['l_max'],
                    sigma= soap_params['sigma'],
                    periodic=True,
                    sparse=False
                )
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_chemical_symbols(["X"] * len(atoms))
    return desc.create(atoms)

def atomic_soap_features_for_composition(structure, soap_params):
    """Produce SOAP features for a structure assuming that the structure is a single element
       (so it is a lattice representation). Suitable for Crabnet.
    """
    from dscribe.descriptors import SOAP
    from pymatgen.io.ase import AseAtomsAdaptor
    desc = SOAP(
                    species=['X'],  # or whatever elements you're using
                    r_cut=soap_params['r_cut'],
                    n_max= soap_params['n_max'],
                    l_max= soap_params['l_max'],
                    sigma= soap_params['sigma'],
                    periodic=True,
                    sparse=False
                )
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms1 = AseAtomsAdaptor.get_atoms(structure)
    atoms.set_chemical_symbols(["X"] * len(atoms))
    soap = desc.create(atoms)
    seq=atoms1.get_chemical_symbols()
    comp=Composition(structure.formula)
    vecs=[]
    for i in range(len(comp)):
        mask = [1 if el == comp.elements[i].symbol else 0 for el in seq]
        avg = soap[np.array(mask, dtype=bool)].mean(axis=0)
        vecs.append(avg)
    return np.array(vecs)

def atom_features_from_structure(structure: Structure, atomic_features: Dict):
    """Calculate an array of atomic features for structure
    """
    embedding_path = atomic_features['atom_feature_strategy']['atom_feature_file']
    atom_features_dict=load_atom_features(embedding_path)
    
    # Verify embedding dimension by checking first element
    if atom_features_dict:
        first_key = list(atom_features_dict.keys())[0]
        expected_dim = len(atom_features_dict[first_key])
        print(f'DEBUG: Loaded embedding from {embedding_path}, feature dimension: {expected_dim}')
    
    if atomic_features['atom_feature_strategy']['soap_atomic']:
        soap=atomic_soap_features(structure, atomic_features['soap_params'])

    atom_features=[]
    for i, site in enumerate(structure):
        number = site.specie.number
        feature = atom_features_dict.get(str(number))
        if feature is None:
            raise ValueError(f"Atomic feature not found for element: {number}")
        if atomic_features['atom_feature_strategy']['soap_atomic']:
            feature=np.concatenate([feature,soap[i,:]],axis=0)
        atom_features.append(feature)
    
    # Verify all features have consistent dimension
    if atom_features:
        feature_dim = len(atom_features[0])
        for i, feat in enumerate(atom_features):
            if len(feat) != feature_dim:
                raise ValueError(f"Inconsistent feature dimension: atom {i} has {len(feat)} features, expected {feature_dim}")

    return atom_features