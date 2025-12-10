# Developer Documentation

This page provides information for developers who want to extend or contribute to Goldilocks.

## Project Structure

```
goldilocks/
├── src/
│   └── qe_input/
│       ├── QE_input.py          # Main Streamlit app entry point
│       ├── pages/                # Streamlit pages
│       │   ├── Intro.py          # Data input page
│       │   ├── Chatbot_generator.py
│       │   ├── Deterministic_generator.py
│       │   └── Documentation.py
│       ├── models/               # ML model implementations
│       │   ├── alignn.py         # ALIGNN model
│       │   ├── cgcnn.py          # CGCNN model
│       │   ├── alignn_graph.py   # ALIGNN graph construction
│       │   ├── cgcnn_graph.py    # CGCNN graph construction
│       │   ├── atom_features_utils.py
│       │   └── compound_features_utils.py
│       ├── kspacing_model.py     # Main prediction function
│       ├── data_utils.py         # Database lookup utilities
│       ├── utils.py              # Input file generation utilities
│       └── embeddings/            # Atomic feature embeddings
├── tests/                        # Test suite
└── docs/                         # Documentation
```

## Key API Functions

### K-Point Prediction

#### `predict_kspacing(structure, model_name, confidence_level=0.95)`

Predicts k-point spacing for a crystal structure using machine learning models.

**Parameters:**
- `structure` (pymatgen.core.structure.Structure): Crystal structure to predict k-point spacing for
- `model_name` (str): Model to use, either `'RF'` (Random Forest) or `'ALIGNN'`
- `confidence_level` (float): Confidence level for prediction intervals (0.95, 0.9, or 0.85)

**Returns:**
- `tuple`: `(kdist, kdist_upper, kdist_lower)` where:
  - `kdist`: Predicted k-point spacing (median)
  - `kdist_upper`: Upper bound of confidence interval
  - `kdist_lower`: Lower bound of confidence interval

**Example:**
```python
from pymatgen.core.structure import Structure
from kspacing_model import predict_kspacing

structure = Structure.from_file("structure.cif")
kdist, kdist_upper, kdist_lower = predict_kspacing(
    structure, 
    model_name='ALIGNN', 
    confidence_level=0.95
)
```

### Structure Lookup

#### `StructureLookup(mp_api_key=None)`

Class for looking up crystal structures from materials databases.

**Parameters:**
- `mp_api_key` (str, optional): Materials Project API key (required for MP lookups)

**Methods:**
- `get_jarvis_table(formula)`: Search JARVIS database
- `get_mp_structure_table(formula)`: Search Materials Project database
- `get_mc3d_structure_table(formula)`: Search MC3D database
- `get_oqmd_structure_table(formula)`: Search OQMD database
- `select_structure_from_table(result_df, id_lookup_func)`: Select structure from search results

**Example:**
```python
from data_utils import StructureLookup

lookup = StructureLookup(mp_api_key="your_key")
results = lookup.get_jarvis_table("SiO2")
structure, primitive = lookup.select_structure_from_table(
    results, 
    lookup.get_jarvis_structure_by_id
)
```

### Input File Generation

#### `generate_input_file(structure, functional, mode, kdist, ...)`

Generates a Quantum Espresso input file.

**Parameters:**
- `structure`: pymatgen Structure object
- `functional` (str): XC functional ('PBEsol' or 'PBE')
- `mode` (str): Pseudopotential mode ('efficiency' or 'precision')
- `kdist` (float): K-point spacing
- Additional parameters for magnetic configuration, etc.

**Returns:**
- `str`: Quantum Espresso input file content

#### `list_of_pseudos(pseudo_potentials_folder, functional, mode, compound, target_folder)`

Determines and copies required pseudopotential files for a compound.

**Returns:**
- `tuple`: `(chosen_subfolder, list_of_element_files)`

#### `cutoff_limits(pseudo_potentials_cutoffs_folder, functional, mode, compound)`

Determines maximum energy and density cutoffs from SSSP tables.

**Returns:**
- `dict`: Dictionary with 'ecutwfc' and 'ecutrho' keys

## Machine Learning Models

### ALIGNN Model

The ALIGNN (Atomistic Line Graph Neural Network) model captures both bond and angle information in crystal structures.

**Key Components:**
- `ALIGNN_PyG`: Main model class
- `build_alignn_graph_with_angles_from_structure()`: Constructs atomic and line graphs
- `atom_features_from_structure()`: Generates atomic features

**Model Architecture:**
- Uses both atomic graph (nodes=atoms, edges=bonds) and line graph (nodes=bonds, edges=angles)
- Supports quantile regression for uncertainty prediction
- Models are stored on Hugging Face Hub: `STFC-SCD/kpoints-goldilocks-ALIGNNd`

### Random Forest Model

The Random Forest model provides fast k-point predictions.

**Key Features:**
- Uses composition and structure features
- Quantile regression for confidence intervals
- Models stored on Hugging Face Hub: `STFC-SCD/kpoints-goldilocks-QRF`

### CGCNN Model

CGCNN (Crystal Graph Convolutional Neural Network) model is used to predict metallicity featuers which are used as input for both ALIGNN and RF models. This CGCNN model was trained on Materials Project 'is_metal' dataset (version October 2025).

## Graph Construction

### ALIGNN Graphs

```python
from models.alignn_graph import build_alignn_graph_with_angles_from_structure
from models.atom_features_utils import atom_features_from_structure

atom_features = atom_features_from_structure(structure, atomic_features_config)
data_g, data_lg = build_alignn_graph_with_angles_from_structure(
    structure, 
    atom_features,
    radius=10.0,
    max_neighbors=12
)
```

### CGCNN Graphs

```python
from models.cgcnn_graph import build_radius_cgcnn_graph_from_structure

atom_features = atom_features_from_structure(structure, atomic_features_config)
data = build_radius_cgcnn_graph_from_structure(
    structure,
    atom_features,
    radius=10.0,
    max_neighbors=12
)
```

## Testing

The test suite uses pytest and includes:

- Unit tests for model components
- Integration tests for prediction functions
- UI tests using Streamlit Testing Library

**Running tests:**
```bash
pytest tests/
```

**With coverage:**
```bash
pytest --cov=src/qe_input tests/
```

## Adding New Models

To add a new ML model for k-point prediction:

1. Implement the model in `src/qe_input/models/`
2. Add prediction logic to `predict_kspacing()` in `kspacing_model.py`
3. Upload trained model to Hugging Face Hub
4. Add model option to UI in `pages/Intro.py`
5. Write tests in `tests/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Dependencies

Key dependencies:
- `pymatgen`: Materials analysis
- `torch`: Deep learning framework
- `torch_geometric`: Graph neural networks
- `matminer`: Materials features
- `dscribe`: SOAP descriptors
- `streamlit`: Web application framework
- `huggingface_hub`: Model storage and download

See `pyproject.toml` or `requirements.txt` for complete dependency list.

