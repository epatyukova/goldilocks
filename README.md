# Goldilocks
A web application for generation input for Quantum Espresso single point SCF calculations. The link to the application can be found below.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://goldilocks.streamlit.app/)
[![Coverage Status](https://coveralls.io/repos/github/epatyukova/goldilocks/badge.svg?branch=main)](https://coveralls.io/github/epatyukova/goldilocks?branch=main)
![Docs](https://github.com/stfc/goldilocks/actions/workflows/docs.yml/badge.svg)

## Contents
* [Running the application](#running-the-application)
   * [Web application](#web-application)
   * [Running locally](#running-locally)
       * [Docker container](#docker-container)
       * [Running in python virtual environment](#running-in-python-virtual-environment)
* [Features](#features)
* [Related research](#related-research)
* [License](#license)
* [Funding](#funding)

Further documentation can be found [here](https://stfc.github.io/goldilocks/)

## Running the application

### Web application
At the moment the application is deployed on Streamlit community cloud and can be found here: https://goldilocks.streamlit.app/.

### Running locally
The application can be run locally whether in a Docker container, or in a python environment.

#### Docker container
* Make sure that Docker is installed
* Clone the repository
```
git clone https://github.com/stfc/goldilocks.git
cd goldilocks
```
* Build a docker image from the project folder
```
docker build -t goldilocks-app .
```
* Run the container
```
docker run -p 8501:8501 goldilocks-app
```
* Open the browser and go to
```
http://localhost:8501
```
#### Running in python virtual environment
* Install Python (>=3.10,<3.13)
* Install Poetry
* Create a clean environment (here it is done with venv, but conda, etc. can also be used)
```
python -m venv .venv
source .venv/bin/activate
```
* Clone the repository
```
git clone https://github.com/stfc/goldilocks.git
cd goldilocks
```
* Install PyTorch, torch-scatter, torch-sparse, torch-cluster, torch-spline-conv first (required for torch-geometric) before running poetry install. Pytorch Geometric requires binary wheels for torch_scatter, torch_sparse, torch_cluster, torch_spline_conv. These cannot be installed by Poetry automatically, so they must be installed first.
   * Install PyTorch
      * CPU-only:
      ```
      pip install torch --index-url https://download.pytorch.org/whl/cpu
      ```
      * CUDA-enabled (example: cu124):
      ```
      pip install torch --index-url https://download.pytorch.org/whl/cu124
      ```
   * Install PyTorch Geometric dependencies
      * CPU wheels:
      ```
      pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
            -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
      ```
      * CUDA wheels (example: cu124):
      ```
      pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
         -f https://data.pyg.org/whl/torch-2.7.0+cu124.html
      ```
      * Install torch_geometric
      ```
      pip install torch_geometric
      ```
* Install remaining dependencies with Poetry
```
poetry install
```
* Run the application
```
streamlit run src/qe_input/QE_input.py
```
* Open a browser and go to
```
http://localhost:8501
```

## Features

The purpose of this application is to predict k-points density for 3D inorganic compounds and generate an input file for a single point SCF energy calculation with 
Quantum Espresso package [1,2].

### Current Strategy for Parameter Selection

The current strategy for choosing parameters for single point SCF calculations is the following:

* **Pseudopotentials**: Use SSSP1.3, PBEsol [3] (with 'efficiency' and 'precision' modes, both available through the application)
* **Cutoffs**: Choose the maximum value for all elements in the compound from tables supplied with the corresponding set of pseudopotentials
* **Smearing**: Constant smearing for all calculations: 0.01 Ry Marzari-Vanderbilt
* **K-point mesh**: Predicted using machine learning models (Random Forest or ALIGNN) trained on data generated for structures from the MC3D database [5]. The models predict k-point spacing with confidence intervals (95%, 90%, or 85% confidence levels available)

### Input File Generation Methods

The input file can be generated via:

* **Deterministic generator**: Uses ASE deterministic function [4] to generate input files. After generation, a `.zip` archive is created containing the structure file, pseudopotential files, and input file ready for QE
* **LLM-based generator**: Uses Large Language Models (OpenAI GPT models or Groq Llama models) to generate input files. Users must provide appropriate API keys (obtainable through links shown in the app). Some LLMs require payment. The LLMs can also:
  - Answer questions about the content of the input file
  - Introduce corrections in the generated file
  - Answer general questions about DFT simulations with QE

## Related research

Performing high-throughput calculations requires an appropriate choice of calculation parameters. For Quantum Espresso single-point self-consistent field energy calculations, this includes selection of a suitable k-mesh, cutoff values for energy and density, suitable pseudo-potentials, a suitable smearing method, and the corresponding degauss value. In some cases, this also includes a suitable functional choice depending on the property that needs to be calculated and the system.

Choosing the right parameters is crucial for achieving accurate results while optimizing computational resources, as improper settings can lead to convergence issues or excessive computation times. Automating this process may help with energy efficiency, accuracy of the calculations, and lower the entry barrier to the field for users.

However, choosing the right strategy of parameter choice is a non-trivial issue, as in general, the given level of accuracy in a property can be achieved with multiple sets of parameters, and the effects of different parameters are often interdependent. However, some strategies in parameter choice can be more beneficial than others due to the robustness toward small changes in their values or the structure.

To develop this application which partially solves the problem. To do it we first generated a training dataset comprising over 20,000 materials, each with an energy convergence threshold of 1 meV/atom. Several ML models were evaluated for their ability to predict k-points distance, and uncertainty estimation was incorporated to guarantee that, for at least 85-95% of compounds, the predicted k-distance lies within the convergence region. The best-performing models, RF and ALIGNN, are availible through the app.

## License
CC BY 4.0

## Funding
* EPSRC EP/Z530657/1 (Goldilocks convergence tools and best practices for numerical approximations in Density Functional Theory calculations)
* Ada Lovelace Center

**References**

[1] *Advanced capabilities for materials modelling with Quantum ESPRESSO* 
   P Giannozzi, O Andreussi, T Brumme, O Bunau, M Buongiorno Nardelli, 
   M Calandra, R Car, C Cavazzoni, D Ceresoli, M Cococcioni, N Colonna, I Carnimeo, 
   A Dal Corso, S de Gironcoli, P Delugas, R A DiStasio Jr, A Ferretti, A Floris, 
   G Fratesi, G Fugallo, R Gebauer, U Gerstmann, F Giustino, T Gorni, J Jia, 
   M Kawamura, H-Y Ko, A Kokalj, E Küçükbenli, M Lazzeri, M Marsili, N Marzari, 
   F Mauri, N L Nguyen, H-V Nguyen, A Otero-de-la-Roza, L Paulatto, S Poncé, D Rocca, 
   R Sabatini, B Santra, M Schlipf, A P Seitsonen, A Smogunov, I Timrov, T Thonhauser, 
   P Umari, N Vast, X Wu and S Baroni, J.Phys.:Condens.Matter 29, 465901 (2017)

[2] *QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials* 
   P. Giannozzi, S. Baroni, N. Bonini, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, 
   G. L. Chiarotti, M. Cococcioni, I. Dabo, A. Dal Corso, S. Fabris, G. Fratesi, 
   S. de Gironcoli, R. Gebauer, U. Gerstmann, C. Gougoussis, A. Kokalj, M. Lazzeri, 
   L. Martin-Samos, N. Marzari, F. Mauri, R. Mazzarello, S. Paolini, A. Pasquarello, 
   L. Paulatto, C. Sbraccia, S. Scandolo, G. Sclauzero, A. P. Seitsonen, A. Smogunov, 
   P. Umari, R. M. Wentzcovitch, J. Phys. Condens. Matter 21, 395502 (2009)

[3] *Precision and efficiency in solid-state pseudopotential calculations* 
   G. Prandini, A. Marrazzo, I. E. Castelli, N. Mounet and N. Marzari, 
   npj Computational Materials 4, 72 (2018), http://materialscloud.org/sssp
            
[4] *The atomic simulation environment—a Python library for working with atoms*.
   Ask Hjorth Larsen, Jens Jørgen Mortensen, Jakob Blomqvist, Ivano E Castelli,  
   Rune Christensen, Marcin Dułak, Jesper Friis, Michael N Groves, Bjørk Hammer, 
   Cory Hargus, Eric D Hermes, Paul C Jennings, Peter Bjerre Jensen, 
   James Kermode, John R Kitchin, Esben Leonhard Kolsbjerg, 
   Joseph Kubal, Kristen Kaasbjerg, Steen Lysgaard, Jón Bergmann Maronsson, 
   Tristan Maxson, Thomas Olsen, Lars Pastewka, Andrew Peterson, Carsten Rostgaard, 
   Jakob Schiøtz, Ole Schütt, Mikkel Strange, Kristian S Thygesen, Tejs Vegge, 
   Lasse Vilhelmsen, Michael Walter, Zhenhua Zeng and Karsten W Jacobsen 
   2017 J. Phys.: Condens. Matter 29 273002

[5] *Materials Cloud three-dimensional crystals database (MC3D)* Sebastiaan Huber, Marnik Bercx, Nicolas Hörmann, Martin Uhrin, Giovanni Pizzi, Nicola Marzari, 
Materials Cloud Archive 2022.38 (2022), https://doi.org/10.24435/materialscloud:rw-t0

