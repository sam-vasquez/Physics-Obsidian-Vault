---
tags:
  - Stat-Mech
---
The Ising model is the simplest example of an interacting spin model. It models a simple magnetic system as a lattice of classical spins aligned with each other and/or an external magnetic field.

The Ising model has Hamiltonian
$$
\mathcal{H} = - J \sum_{(ij)} s_i s_j - \sum H \mu s_i,
$$
where $J$ is an interaction energy, $(ij)$ denotes a sum over nearest neighbor pairs, $H$ is the external (applied) magnetic field, and $\mu$ is the magnetic moment. When $J > 0$, parallel spins are favored, and this models a ferromagnet. When $J<0$, this models an anti-ferromagnet.

In the absence of the external field, the Hamiltonian has $\mathbb{Z}_2$ [[Cyclic Group|symmetry]]. This symmetry is also spontaneously broken as $T \rightarrow T_C$, inducing a net magnetization:
$$
m(T) = \lim_{H\rightarrow 0} m(T) = \sqrt{ \lim_{|\vec{r}| \rightarrow \infty} \langle \sigma(0) \sigma(\vec{r}) \rangle }
$$


In 1D, the Ising model is [[Exact Solution of 1D Ising Model|exactly solvable]]. However, it fails to exhibit any phase transition at any finite temperature. There is a [[Peierl's Theorem|heuristic]] for this - regions of nonzero magnetic moments are unstable in 1 dimension.

In 2D, the Ising model is also exactly solvable, albeit much more difficultly so. Onsager did this. 

In some sense, the Ising model is already a [[Mean Field Theory]]: its coupling to the external magnetic field is through the mean magnetization $\langle m \rangle = \frac{1}{N} \sum_i s_i$. Its internal interactions in the other term can also be replaced with mean field interaction, yielding the [[Curie-Weiss Mean Field Theory]].


Can generalize to take on $q$ values, $\sigma_i \in \{1,2,\ldots,q\}$. This is the $q$-state [Potts model]]. 
