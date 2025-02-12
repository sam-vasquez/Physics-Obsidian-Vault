---
tags:
  - Stat-Mech
---
The Grand potential is a thermodynamic potential that is a [[Legendre Transformations|Legendre transformation]] of internal energy $U$ to replace state variables $(S,V,n)$ with conjugate variables $(T,V,\mu)$.

The [[Thermodynamic Identity]] for grand potential $\Phi(T,V,\mu)$ is 
$$
d\Phi = -S dT - p dV - N_i \; d\mu_i.
$$
It therefore provides convenient expressions for [[Entropy]], pressure, and particle number:
$$
S = -\left( \frac{\partial \Phi}{\partial T} \right)_{V,\mu}
$$
$$
p = -\left( \frac{\partial \Phi}{\partial V} \right)_{T,\mu}
$$
$$
N = - \left( \frac{\partial \Phi}{\partial \mu} \right)_{T,V}
$$
Since $\Phi$ is only extensive in $V$, it can be written as
$$
\Phi = - p(T,\mu) V.
$$
The grand potential is related to the grand canonical ensemble:
$$
\Phi = - k_B T \ln \mathcal{Z}.
$$
This fact gives an important and general expression for the thermodynamic state equation:
$$
\begin{align}
S &= -k_B \sum_i p_i \ln p_i  \\
&= -k_B \sum_i p_i (-\beta E_i + \beta \mu N_i - \ln \mathcal{Z}) \\
&= k_B \beta \langle E \rangle - k_B \beta \mu \langle N \rangle + k_B \ln \mathcal{Z}
\end{align}
$$
$$
\implies -k T \ln \mathcal{Z} = E - TS - \mu N = \Phi = -pV.
$$
$$
\implies \beta p V = \ln \mathcal{Z}.
$$
