---
tags:
  - Stat-Mech
---
In statistical mechanics, the notion of taking the classical limit of a quantum gas has various equivalent representations.

No matter the case, the classical limit is a limiting case of the [[Ideal Bose Gas]] where the exponential term dominates and the occupation number distribution becomes
$$
\langle n(\epsilon_j) \rangle = \frac{1}{e^{ -\beta \mu } e^{ \beta\epsilon_j } \pm 1} \sim e^{\beta \mu} e^{ -\beta \epsilon_j }.
$$
**This limit is observed when $\mu$ is large and negative.** 
It is easily determined that 
$$
\langle N \rangle = \sum_j \langle n_j \rangle = e^{ \beta\mu }\sum_j e^{ -\beta\epsilon_j } = e^{ \beta\mu } Z_1,
$$
$$
\mu = - k_B T \ln \left( \frac{Z_1}{\langle N \rangle} \right).
$$
**Therefore, this is equivalent to a limit where there are many more (thermally accessible) states than there are particles. $\langle n(\epsilon) \rangle \ll 1$ for all $\epsilon$.**
It is worth noting that thermal accessibility is important, because any state with energy much larger than $k_B T$ has negligible contribution to the occupancy. It is safe to neglect these states when integrating to get total particle number.  

Because of the general tendency for [[Chemical Potential]] to scale with temperature, it can be said for fermions in particular that they are in the classical limit **when the temperature is much higher than the [[Fermi Energy]], $k_B T \gg \epsilon_F$,** since the Fermi energy is the zero-temperature limit of the chemical potential.

