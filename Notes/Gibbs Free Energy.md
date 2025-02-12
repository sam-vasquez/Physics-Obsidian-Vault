---
tags:
  - Stat-Mech
---
Gibbs free energy is a thermodynamic potential that is a [[Legendre Transformations|Legendre transformation]] of internal energy $U$ to replace state variables $(S,V,n)$ with conjugate variables $(T,p,n)$ via $G = U - TS + pV$.

The motivation comes mostly from chemistry, where we don't expect most processes to involve changes in volume, but rather changes in pressure and temperature. And we don't expect to be able to control volume or entropy, but we can control pressure and temperature readily. (Honestly, the same is true in physics, but it's a stubborn tradition to use [[Free Energy]] instead. But for what it's worth, at least free energy has a very motivating microscopic origin.)

The [[Thermodynamic Identity]] for Gibbs free energy $G(T,p,n)$ is given as
$$
dG = -S dT + V dp + \mu_i \; dn_i.
$$

It therefore provides convenient expressions for [[Entropy]], pressure, and [[Chemical Potential]]:
$$
S = -\left( \frac{\partial G}{\partial T} \right)_{p,n}
$$
$$
V = \left( \frac{\partial G}{\partial p} \right)_{T,n}
$$
$$
\mu = \left( \frac{\partial G}{\partial n} \right)_{T,p}.
$$
In fact, since $G$ is only extensive in $N$, it can be written as
$$
G = \mu(p,T) n.
$$

Gibbs free energy is interesting when it comes to phase transitions. Compare with the discussion in [[Enthalpy]]: Suppose a liquid is held at constant temperature and pressure, and heat is supplied such that it undergoes a phase transition and goes from liquid to gas. Enthalpy is appropriate to determine how much work was done to affect this change. But in fact, Gibbs free energy would be unchanged: $\Delta G = 0$, since all state variables of interest to $G$ were unchanged. 


