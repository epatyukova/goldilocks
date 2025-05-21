# Welcome to Goldilocks

Goldilocks helps you generate Quantum Espresso SCF input files.

Use the menu to navigate the documentation.

## Introduction

This simple application aims to generate input files for Quantum Espresso single point energy calculations.

Performing simple SCF energy calculation requires a correct choice of parameters: functional, pseudo-potentials, plane wave cutoff values for energy and density, k-mesh, suitable smearing method, and corresponding degauss value. Magnetic compounds require an appropriate spin configuration to be specified along with starting magnetizations for each atomic species, and possibly the use of starting_magnetization or constrained magnetization settings to ensure convergence to the correct magnetic state.

The parameters ususally can be chosen in a non-unique way. Choosing the set of parameters providing a given calculation accuracy usually requiers additional calculations to check convergence. Here we aim at predicting good sets of parameters. This may help to avoid running convergence studies or provide a good starting point for such calculations.

Scientific details, and model evaluations can be found in [Related research](related-research.md) section.

## Contributors

This application is one of the deliverables of the Goldilocks project (EPSRC EP/Z530657/1, Goldilocks convergence tools and best practices for numerical approximations in Density Functional Theory calculations) led by Barbara Monatani (STFC, UK), Gilberto Teobaldi (STFC, UK), Alin Marin Elena (STFC, UK), and Susmita Basak (STFC, UK).

The creator of the application is Elena Patyukova (STFC, UK).