# K-Point Mesh Prediction Models

This page describes the machine learning models used to predict optimal k-point meshes for Quantum Espresso calculations.

## Overview

The k-point mesh is crucial for accurate DFT calculations. Too coarse a mesh leads to inaccurate results, while too fine a mesh wastes computational resources. Goldilocks uses machine learning models to predict optimal k-point spacing (`kdist`) based on crystal structure features.

The models predict:
- **kdist**: Optimal k-point spacing (median prediction)
- **kdist_upper**: Upper bound of confidence interval
- **kdist_lower**: Lower bound of confidence interval

The k-point grid is then generated as: `kpoints = ceil(2π / (lattice_constant × kdist))`

## Available Models

### Random Forest (RF)

The Random Forest model provides fast and reliable k-point predictions.

**Features:**
- Uses composition and structure-based features
- Quantile regression for uncertainty estimation
- Fast inference time
- Good performance on diverse crystal structures

**Model Details:**
- Trained on MC3D database structures
- Uses quantile regression to predict confidence intervals
- Model stored on Hugging Face Hub: `STFC-SCD/kpoints-goldilocks-QRF`

**When to use:**
- Default choice for most structures
- When fast prediction is needed
- For standard crystal structures

### ALIGNN (Atomistic Line Graph Neural Network)

The ALIGNN model captures both bond and angle information, providing more accurate predictions for complex structures.

**Features:**
- Graph neural network architecture
- Uses both atomic graph (bonds) and line graph (angles)
- Better captures local chemical environment
- More accurate for structures with complex bonding

**Model Architecture:**
- **Atomic Graph**: Nodes are atoms, edges are bonds (within cutoff radius)
- **Line Graph**: Nodes are bonds, edges represent angles between bonds
- **Features**: 86-dimensional atomic embeddings from SSSP cutoffs
- **Output**: Quantile regression predictions (lower, median, upper)

**Model Details:**
- Trained on MC3D database structures
- Uses PyTorch Geometric for graph operations
- Models stored on Hugging Face Hub: `STFC-SCD/kpoints-goldilocks-ALIGNNd`
- Separate models for different confidence levels (0.95, 0.9, 0.85)

**When to use:**
- For structures with complex bonding environments
- When higher accuracy is needed
- For structures with unusual coordination

## Confidence Levels

Both models support three confidence levels:

- **0.95 (95% confidence)**: Most conservative, widest intervals
- **0.90 (90% confidence)**: Balanced option
- **0.85 (85% confidence)**: Narrower intervals, more aggressive

The confidence intervals account for:
- Model uncertainty
- Training data distribution
- Structure-specific factors

## Model Training

### Training Data

Models were trained on:
- **Database**: MC3D (Materials Cloud three-dimensional crystals database)
- **Structures**: Diverse crystal structures from various space groups
- **Target**: Optimal k-point spacing determined from convergence studies

### Feature Engineering

**Random Forest Features:**
- Composition features (matminer)
- Structure features (matminer)
- Lattice parameters
- SOAP descriptors (optional)

**ALIGNN Features:**
- Atomic embeddings (86-dimensional from SSSP cutoffs)
- Graph structure (bonds and angles)
- Edge distances
- Angle cosines

### Model Calibration

Models are calibrated using conformalized quantile regression to ensure proper coverage of confidence intervals. Corrections are applied based on the selected confidence level.

## Usage in Goldilocks

1. **Select Model**: Choose RF or ALIGNN on the data input page
2. **Select Confidence Level**: Choose 0.95, 0.9, or 0.85
3. **Provide Structure**: Upload CIF file or search database
4. **Get Prediction**: Model predicts kdist with confidence intervals
5. **Generate Grid**: K-point grid is automatically generated

## Performance Considerations

**Random Forest:**
- Inference time: ~0.1-1 seconds
- Memory: Low
- Accuracy: Good for standard structures

**ALIGNN:**
- Inference time: ~1-5 seconds (includes model download on first use)
- Memory: Moderate (requires PyTorch)
- Accuracy: Better for complex structures

## Model Updates

Models are stored on Hugging Face Hub and downloaded automatically on first use. To update models:

1. Models are versioned on Hugging Face Hub
2. New versions can be uploaded without code changes
3. Model versioning ensures reproducibility

## Limitations

- Models are trained on MC3D database structures
- Performance may vary for very unusual structures
- Confidence intervals are approximate
- Models assume standard DFT calculations (not specialized for specific properties)

## Future Improvements

Potential enhancements:
- Models trained on more diverse datasets
- Property-specific models (e.g., for band gap calculations)
- Active learning for model improvement
- Uncertainty quantification improvements
- Support for more confidence levels

## References

- ALIGNN: [Choudhary & DeCost (2021)](https://doi.org/10.1038/s41524-021-00650-1)
- CGCNN: [Xie & Grossman (2018)](https://doi.org/10.1103/PhysRevLett.120.145301)
- MC3D Database: [Huber et al. (2022)](https://doi.org/10.24435/materialscloud:rw-t0)

