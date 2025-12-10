The data used to train the models is availible at [PSDI](https://data-collections.psdi.ac.uk/records/75959-bwa52)

### Data generation approach

With all interconnected factors affecting k-points convergence for single point energy calculations in mind, the following set of parameters was chosen to generate the data:

1.  The valence electrons are represented using the Standard Solid-State Pseudopotential (SSSP) library (version 1.3, PBEsol, efficiency set), with plane-wave cutoff energies determined following the efficiency-level recommendations.
2.  The Marzari–Vanderbilt cold-smearing scheme, with a smearing width of 0.01 Ry, is applied in all calculations. The choice of the cold smearing is justified by its ability to yield a free energy that is quite insensitive to variations in the smearing temperature. The smearing temperature is set as 0.01 Ry, which ensures the smearing convergence for most materials according to Nascimento’s report [1]. This also implies that even for insulators, the smearing is applied to the electronic states around the fermi level, rather than fixed occupations.
3.  The 20187 structures were selected randomly from the MC3D PBEsol-v1 dataset and no additional structural relaxation was performed [2]. 
4.  Spin averaged (i.e. without spin-polarisation)  calculations were performed for all materials and no magnetic configurations were taken into consideration. 
5.  The k-mesh convergence is determined as follows: when the energy difference among three consecutive k-point distances becomes smaller than 1 meV per atom, the first point of these three is identified as the converged k-mesh. This procedure does not guarantee that one has the optimal set of parameters, all pockets of Fermi surface are resolved, or that total energy of compounds with a band gap below 0.14eV is resolved correctly. However, it is a straightforward an easy way to generate the data. 
6.   We employ the definition of k-distance, as implemented in the AiiDA–QuantumESPRESSO package [3,4], to generate various k-point meshes. The k-distance represents the maximum spacing (in Å⁻¹) between adjacent k-points in reciprocal space, such that the number of k-points along each reciprocal lattice vector bᵢ is given by ⌈|b_i |⁄(k_dist)⌉. Starting from a k-distance of 1.0 Å⁻¹, we systematically scan all possible k-meshes by varying the k-distance in steps of 0.005 Å⁻¹. We always include the Gamma point for future calculations regarding electronic structures, phonon properties, etc. 

###References
[1] G. de Miranda Nascimento, F. J. dos Santos, M. Bercx,
D. Grassano, G. Pizzi and N. Marzari, Accurate and efficient
protocols for high-throughput first-principles materials simula-
tions, 2025, [https://arxiv.org/abs/2504.03962](https://arxiv.org/abs/2504.03962).

[2] S. P. Huber, M. Minotakis, M. Bercx, T. Reents, K. Eimre,
N. Paulish, N. Hörmann, M. Uhrin, N. Marzari and G. Pizzi,
MC3D: The Materials Cloud computational database of ex-
perimentally known stoichiometric inorganics, 2025, [https://arxiv.org/abs/2508.19223](https://arxiv.org/abs/2508.19223).

[3] S. P. Huber, S. Zoupanos, M. Uhrin, L. Talirz, L. Kahle,
R. Häuselmann, D. Gresch, T. Müller, A. V. Yakutovich, C. W.
Andersen, F. F. Ramirez, C. S. Adorf, F. Gargiulo, S. Kumbhar,
E. Passaro, C. Johnston, A. Merkys, A. Cepellotti, N. Mounet,
N. Marzari, B. Kozinsky and G. Pizzi, Scientific Data, 2020, 7,
300, [https://www.nature.com/articles/s41597-020-00638-4](https://www.nature.com/articles/s41597-020-00638-4).

[4] M. Uhrin, S. P. Huber, J. Yu, N. Marzari and G. Pizzi, Compu-
tational Materials Science, 2021, 187, 110086, [https://doi.org/10.1016/j.commatsci.2020.110086](https://doi.org/10.1016/j.commatsci.2020.110086).