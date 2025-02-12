---
tags:
  - Stat-Mech
---
The canonical ensemble is an alternative to the [[Canonical Ensemble]] where the system is still constrained to a fixed temperature, but particle unmber can change.

Accordingly, the probability assigned to each microstate changes from $e^{ -\beta E_i }$ to $e^{ -\beta (E_i - \mu N_i) }$. Hence, the grand canonical ensemble has grand partition function
$$
\mathcal{Z} = \sum_i e^{ -\beta (E_i - \mu N_i) }
$$
The canonical partition function naturally relates to the [[Grand Potential]] as 
$$
\Phi = - k_B T \ln \mathcal{Z}.
$$
Derivatives of the grand partition function give most thermodynamic quantities as statistical averages with respect to the partition function.
Average internal energy is still 
$$
\langle E \rangle = - \frac{\partial }{\partial \beta} \ln \mathcal{Z}.
$$
Average total particle number is
$$
\langle N \rangle = \frac{\partial }{\partial (\beta\mu)} \ln \mathcal{Z} = z \frac{\partial }{\partial z} \ln \mathcal{Z}.
$$
($z = e^{\beta \mu}$).

In the occupation number representation, necessary for ensembles of identical particles, each microstate is characterized by the number of particles occupying each energy level. For a particular state $\nu$ with occupation numbers $\ket{ n_1, n_2,\ldots,n_j,\ldots }$, the grand partition function is 
$$
\mathcal{Z} = \sum_\nu e^{-\beta (E_\nu - \mu N_\nu)} = \sum_{n_1,\ldots,n_j,\ldots}e^{ -\beta (E[n_1,\ldots,n_j,\ldots] - \mu \sum_j n_j) }.
$$
Further, when an ensemble contains non-interacting particles, the energy of each microstate is proportional to the number of particles in each energy level. The partition function decomposes further.
$$
\mathcal{Z} = \sum_{n_1,\ldots,n_j,\ldots} e^{ -\beta (\sum_j \epsilon_j n_j - \mu \sum_j n_j) } = \prod_i \left(  \sum_{n_i} e^{ -\beta(\epsilon_i - \mu) n_i }\right).
$$
The average number occupancy of each energy level is obtained as 
$$
\langle n_i \rangle = - \frac{\partial }{\partial (\beta \epsilon_i)} \ln \mathcal{Z}.
$$
Despite only applying to systems with a varying number of particles, this occupation number formalism actually tends to make it easier to study systems with fixed number of particles in the grand canonical ensemble instead of the canonical ensemble. 
The fixed particle number can be restored by solving for the chemical potential $\mu$ that enforces $\sum \langle n_i \rangle = N$. 
This approach is most notably used when studying the [[Ideal Bose Gas]] and [[Ideal Fermi Gas]], to avoid the complicated combinatorics of quantum statistics.