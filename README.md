# Goldilocks
A web application for generation input for Quantum Espresso single point SCF calculations. The link to the application can be found below.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://goldilocks.streamlit.app/)

## Contents
* [Running the application](#running-the-application)
   * [Web application](#web-application)
   * [Running locally](#running-locally)
       * [Docker container](#docker-container)
       * [Running in python environment](#running-in-python-environment)
* [Features](#features)
* [Related research](#related-research)
* [License](#license)
* [Funding](#funding)

## Running the application

### Web application
At the moment the applicaiton is deployed on Streamlit community cloud and can be tried out here https://goldilocks.streamlit.app/. In nearest future it will be transfered to https://goldilocks.ac.uk/.

### Running locally
The application can be run locally whether in a Docker container, or in python environment.

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
#### Running in python environment
* Install Python (>=3.10,<3.13)
* Install Poetry
* Create clean environment (here it is done with venv, but conda, etc. can also be used)
```
python -m venv .venv
source .venv/bin/activate
```
* Clone the repository
```
git clone https://github.com/stfc/goldilocks.git
cd goldilocks
```
* Install dependencies
```
poetry install
```
* Run the application
```
run streamlit run src/qe_input/QE_input.py
```
* Open the browser and go to
```
http://localhost:8501
```

## Features

The purpose if this application is to help unexperienced user to setup single point SCF energy calculation with 
Quantum Espresso package [1,2].

Current strategy of choice of parameters for single point SCF calulations is the following:

* Use SSSP library of pseudo-potentials [3] (there are versions for PBE, PBEsol funcitonals and 'efficiency' and 'precision' calculations, all of which are availible through the app)
* For cutoffs choose the maximum value for all elements in the compound from tables supplied with corresponding set of pseudopotentials
* We use constant smearing for all calculations 0.01 Ry Marzari-Vanderbilt smearing
* We predict k-point mesh with machine learning model (at the moment it is CGCNN) which was trained on data generated for the set of strutures from MC3D database [8] (additional relaxation was not perfomed). We also predict confidece interval for k-points prediction (95% confidence). 
            
The input file can be generated via:
* ASE deterministic function [4] (after generation .zip archive is created with the structure file, PP files, and input file which can be used as QE input)
* With a set of LLMs (the choice is availible, the user should provide appropriate API key, which can be obtained through links shown in the app. Some LLMs require payment).
* LLMs can also answer questions about the content of the input file, introduce corrections in the generated file, and answer general questions about DFT simulations with QE.

## Related research

Performing high-throughput calculations requires the appropriate choice of calculation parameters. For Quantum Espresso single-point self-consistent field calculations, this includes selection of a suitable k-mesh, cutoff values for energy and density, suitable pseudo-potentials, a suitable smearing method, and the corresponding degauss value. In some cases, this also includes a suitable functional choice depending on the property that needs to be calculated and the system.

Choosing the right parameters is crucial for achieving accurate results while optimizing computational resources, as improper settings can lead to convergence issues or excessive computation times. Automating this process may help with energy efficiency, accuracy of the calculations, and lower the entry barrier to the field for users.

However, choosing the right strategy of parameter choice is a non-trivial issue, as in general, the given level of accuracy in a property can be achieved with multiple sets of parameters, and the effects of different parameters are often interdependent. However, some strategies in parameter choice can be more beneficial than others due to the robustness toward small changes in their values or the structure.

One of the most common approaches to a parameter choice in high-throughput calculations is fixing all those parameters at certain values for all compounds, usually at sufficiently large k-point densities and cutoffs.  In this situation, there is no control over errors, calculations often turn out to be overconverged, and electricity is overconsumed. However, if the parameters have sufficiently high values, at least the calculations are converged [6].

Another common approach is related to an attempt to separate all the contributions of different parameters, analyse them separately, and give recommendations for the choice of the these parameters, based on performance on some pre-defined benchmark. The SSSP library of pseudopotentials follows this approach [3]. Extensive analysis of the performance of pseudopotentials allowed to derive recommendations concerning pseudopotentials and the cutoff values. This analysis was performed for properties of elemental crystals with k-point grids 20×20×20 + MV smearing 0.002Ry, 10×10×10 + MV smearing 0.02Ry, or 6×6×6 + MV smearing 0.02Ry, depending on the property and type of material. The result of this work is a recommendation of PP for each atom, and energy + density cutoff tables.  Although the benchmark contains only elemental crystals, the same recommendations are expected to be extrapolated to multielement compounds. Also, it is discussed that PP performance strongly depends on the task. So, having the fixed choice of PP based on the errors averaged across different tasks is an oversimplification. 

The same authors continue the pursuit of automatic parameter choice to suggest the values of k-point density and smearing temperature recently [10]. At the end, they arrived at 3 sets of recommendations for 3 classes of compounds: isolators, metals, and compounds containing lanthanides.

In another recent paper  the convergence of equilibrium volume with respect to the values of cutoffs and k-point density for VASP package (and collection of pseudopotentials) was studied [11]. They showed that the given accuracy is achieved at a set of values cutoff + k-points (curve resembling cutoff * kpoints = constant). They also quantified errors, showing (and explaining from physical perspective) that systematic errors due to cutoff and k-points are largely independent from each other.

According to our knowledge there were no systematic studies of the influence of magnetism on convergence behavior, however in the SSSP paper [3] authors calculated the dependence of error on energy cutoff (for fixed duals, k-meshes, smearings) both for non-magnetic and magnetic cases for magnetic elemental crystal and found no significant differences in convergence trends. This suggests that DFT parameters can be chosen based on non-magnetic case.

## License
MIT

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

[5] *Python Materials Genomics (pymatgen) : A Robust,
   Open-Source Python Library for Materials Analysis.* 
   Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier,
   Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent Chevrier, Kristin A.
   Persson, Gerbrand Ceder. Computational Materials
   Science, 2013, 68, 314–319. https://doi.org/10.1016/j.commatsci.2012.10.028 

[6] *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation* 
   Anubhav Jain, Shyue Ping Ong, Geoffroy Hautier, Wei Chen, William Davidson Richards, 
   Stephen Dacek, Shreyas Cholia, Dan Gunter, David Skinner, Gerbrand Ceder, and Kristin A. Persson

[7] *The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design* 
   Choudhary, K., Garrity, K.F., Reid, A.C.E. et al. npj Computational Materials 6, 173 (2020) https://doi.org/10.1038/s41524-020-00440-1
   We use the partial copy of dft3d dataset to query the structures by formula.

[8] *Materials Cloud three-dimensional crystals database (MC3D)* Sebastiaan Huber, Marnik Bercx, Nicolas Hörmann, Martin Uhrin, Giovanni Pizzi, Nicola Marzari, 
Materials Cloud Archive 2022.38 (2022), https://doi.org/10.24435/materialscloud:rw-t0

[9] To provide reference and advise we suggest to use *OpenAI* models, see usage conditions https://openai.com/policies/row-terms-of-use/ 

[10] *Accurate and efficient protocols for high-throughput first-principles materials simulations* Nascimento, G.M., Santos, F.J., Bercx, M., Grassano, D., Pizzi, G., Marzari, N., 
https://doi.org/10.48550/arXiv.2504.03962

[11] *Automated optimization of convergence parameters in plane wave density functional theory calculations via a tensor decomposition-based uncertainty quantification* Janssen, J., Makarov, E., Hickel, T., Shapeev, A.V., and Neugebauer, J., npj Comput Mater 10, 263 (2024). https://doi.org/10.1038/s41524-024-01388-2 .
