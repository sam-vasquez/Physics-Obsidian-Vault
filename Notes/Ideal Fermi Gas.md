---
tags:
  - Stat-Mech
---
The ideal Fermi gas is an ensemble of non-interacting identical particles with fermion statistics. 

Their thermodynamics can be derived using the [[Grand Canonical Ensemble]]. Following the discussion on that page of non-interacting particles, we get 
$$
\mathcal{Z} = \prod_i \left(  \sum_{n_i} e^{ -\beta(\epsilon_i - \mu) n_i }\right).
$$
For fermions, each occupation number is only either 0 or 1. Each sum is trivially
$$
\mathcal{Z}_F = \prod_i \left( 1 + e^{ -\beta(\epsilon_i - \mu) } \right).
$$
From here, for example, the average occupation number of each energy level is 
$$
\langle n_j \rangle_F = - \frac{\partial }{\partial (\beta \epsilon_j)} \ln \mathcal{Z}_F = \frac{1}{e^{ \beta(\epsilon_j - \mu) } + 1}.
$$

Note the symmetry. 