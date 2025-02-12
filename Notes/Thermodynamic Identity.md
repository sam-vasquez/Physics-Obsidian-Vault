---
tags:
  - Stat-Mech
---
The thermodynamic identity is the fundamental equation of state for all thermodynamics, built out of the three laws of thermodynamics.

For internal energy $U$, temperature $T$, entropy $S$, and an arbitrary collection of intensive state variables $f_i$ and corresponding extensive state variables $X_i$, the following holds:
$$
dU = TdS + f \cdot dX_i.
$$
The canonical example is the case of a gas with pressure $p$ in a container of volume $V$, made of at least one constituent particle type with chemical potential $\mu$ and number $n$, which gives the identity as
$$
dU = T dS - p \,dV + \sum_i \mu_i \; dn_i.
$$
Strictly speaking, we could just collect $T$ into the arbitrary intensive state variables and $S$ into the arbitrary extensive state variables. But $S$ is a distinguished variable due to its fundamental relation to energy and statistical disorder given by the [[Second Law of Thermodynamics]].

Sometimes, it would be more useful to rewrite the thermodynamic identity in terms of differentials of different state variables, because it isn't always practical to hold a given variable constant, and it isn't always relevant to vary a given variable. [[Legendre Transformations]] allow us to swap our conjugate variables by introducing new thermodynamic potentials.

For $U = U(S,V,n)$, the internal energy identity is
$$
dU = T dS - p \,dV + \sum_i \mu_i \; dn_i.
$$

For $U - TS = F(T,V,n)$, the [[Free Energy]] identity is
$$
dF = -S dT - p dV + \sum_i \mu_i \; dn_i.
$$

For $U - TS + pV = G(T,p,n)$, the [[Gibbs Free Energy]] identity is
$$
dG = -S dT + V dp + \sum_i \mu_i \; dn_i.
$$

For $U + pV = H(S,p,n)$, the [[Enthalpy]] identity is
$$
dH = T dS + V dp + \sum_i \mu_i \; dn_i.
$$
