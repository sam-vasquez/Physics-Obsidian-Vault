---
tags:
  - Stat-Mech
---
Free energy is a thermodynamic potential that is a [[Legendre Transformations|Legendre transformation]] of internal energy $U$ to replace state variables $(S,V,n)$ with conjugate variables $(T,V,n)$.

The motivation is that it is excessively unclear how exactly you're supposed to hold entropy constant while varying volume or particle number. In fact, it's essentially impossible. But controlling temperature is simpler.

The [[Thermodynamic Identity]] for free energy $F(T,V,n)$ is given as
$$
dF = -S dT - p dV + \mu_i \; dn_i.
$$
It therefore provides convenient expressions for [[Entropy]], pressure, and [[Chemical Potential]]:
$$
S = -\left( \frac{\partial F}{\partial T} \right)_{V,n}
$$
$$
p = -\left( \frac{\partial F}{\partial V} \right)_{T,n}
$$
$$
\mu = \left( \frac{\partial F}{\partial n} \right)_{T,V}
$$
Note that when temperature is held constant, $dF|_T = -p dV + \mu_i dn_i = \delta W$. Free energy therefore becomes a convenient way to understand how much work we can extract from a system, hence the name. (Perhaps "freeable" energy is more appropriate?)

Free energy is related to the [[Canonical Ensemble]] via
$$
F = - k_B T \ln Z.
$$
Proof:
$$
\frac{\partial }{\partial \beta} (\beta F) = \beta \frac{\partial F}{\partial \beta} + F = \frac{1}{k_B T} \frac{\partial F}{\partial T} \frac{\partial T}{\partial \beta} + F = TS + F = E = -\frac{\partial }{\partial \beta} \ln Z.
$$
Therefore $F = -\frac{1}{\beta} \ln Z + C/\beta$, integrating constant $C$ is determined by $T\rightarrow 0$ behavior, partition function dominated by ground state $g_0 e^{ -\beta \epsilon_0 }$, $g_0$ is degeneracy of the ground state, 
$$
 F \rightarrow -\frac{1}{\beta} (\ln g_0 -\beta \epsilon_0) + C/\beta = \epsilon_0 - \frac{1}{\beta}\ln g_0 + \frac{1}{\beta} C = \epsilon_0 - T S(T=0).
$$
Since ground state entropy is $S(T=0) = k_B \ln g_0$, $C=0$. $F = -\frac{1}{\beta} \ln Z$.
