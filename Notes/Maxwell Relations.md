---
tags:
  - Stat-Mech
---
Maxwell relations are relations between derivatives of various [[State Variable|thermodynamic state variables]] as a simple consequence of the symmetry of second derivatives of analytic functions.

The [[Thermodynamic Identity|thermodynamic identities]] provide various expressions for differentials of internal energy and other thermodynamic potentials in terms of differentials of state variables. 

For example, given the internal energy identity
$$
dU = T dS - p dV + \sum_i \mu_i \; dn_i,
$$
it is taken that $T = \left( \frac{\partial U}{\partial S} \right)_{V,n}$ and $p = - \left( \frac{\partial U}{\partial V} \right)_{S,n}$. If you take a derivative $\left( \frac{\partial }{\partial V} \right)_{S,n}$ of $T$ and a derivative of $\left( \frac{\partial }{\partial S} \right)_{V,n}$ of $p$, the derivatives commute and the expressions are equal (up to a minus sign):
$$
\left( \frac{\partial T}{\partial V} \right)_{S,n} = - \left( \frac{\partial p}{\partial S} \right)_{V,n}.
$$
Similarly, for a magnetic system (with $+HdM$ convention),
$$
\left( \frac{\partial T}{\partial M} \right)_S = \left( \frac{\partial H}{\partial S} \right)_M.
$$
[[Enthalpy]] gives:
$$
\left( \frac{\partial T}{\partial p} \right)_S = \left( \frac{\partial V}{\partial S} \right)_p.
$$
$$
- \left( \frac{\partial T}{\partial H} \right) = \left( \frac{\partial M}{\partial S} \right)_H.
$$
[[Free Energy]] gives:
$$
\left( \frac{\partial S}{\partial V} \right)_T = \left( \frac{\partial p}{\partial T} \right)_V.
$$
$$
\left( \frac{\partial S}{\partial M} \right)_T = -\left( \frac{\partial H}{\partial T} \right)_M.
$$
[[Gibbs Free Energy]] gives:
$$
- \left( \frac{\partial S}{\partial p} \right)_T = \left( \frac{\partial V}{\partial T} \right)_p.
$$
$$
\left( \frac{\partial S}{\partial M} \right)_T = \left( \frac{\partial M}{\partial T} \right)_H.
$$
