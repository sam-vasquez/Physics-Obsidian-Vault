---
tags:
  - Stat-Mech
---
The ideal Bose gas is an ensemble of non-interacting identical particles with boson statistics. 

Their thermodynamics can be derived using the [[Grand Canonical Ensemble]]. Following the discussion on that page of non-interacting particles, we get 
$$
\mathcal{Z} = \prod_i \left(  \sum_{n_i} e^{ -\beta(\epsilon_i - \mu) n_i }\right).
$$
For bosons, each occupation number can range from 0 to infinity. Each sum converges to a geometric series, but only when $\mu < \epsilon_i$ for all $i$. In that case,
$$
\mathcal{Z}_B = \prod_i \left( \frac{1}{1 - e^{ -\beta(\epsilon_i - \mu) }} \right).
$$
From here, for example, the average occupation number of each energy level is 
$$
\langle n_j \rangle_B = - \frac{\partial }{\partial (\beta \epsilon_j)} \ln \mathcal{Z}_B = \frac{1}{e^{ \beta(\epsilon_j - \mu) } - 1}.
$$
It is sometimes written in terms of fugacity $z = e^{\beta \mu}$:
$$
\langle n_j \rangle_B = \frac{1}{z^{-1}e^{ \beta\epsilon_j} - 1}.
$$

When the bosons are massless, the chemical potential is zero.


When the bosons have mass, the chemical potential is finite and negative, $0 < z < 1$.
Approximating as an integral and integrating with respect to the massive dispersion relation to get total boson number gives
$$
\begin{align}
N &= \int d\epsilon g(\epsilon) \langle n(\epsilon) \rangle  \\
&\propto \int_0^\infty d\epsilon \frac{\epsilon^{\frac{d}{2} - 1}}{z^{-1} e^{\beta \epsilon} - 1}  \\
&= \frac{V}{\lambda_{\textrm{th}}^d} g_{d/2}(z),
\end{align}
$$
where $g_\nu(z)$ is the [[Polylogarithm]]. 
However, this expression is not quite correct. For some dimensions, $g$ is bounded above by $z=1$, which implies a non-physical restriction on the particle density $\rho < \zeta(d/2)/\lambda^d$. In truth, the failure of this expression is because the ground state is given zero weight in the density of states. This gives rise to a [[Bose-Einstein Condensate]]. 

In [[The Classical Limit]], 
the exponential term becomes much larger than unity, and the difference between bosons and fermions vanishes.
$$
\langle n_j \rangle = e^{ - \beta \epsilon_j } e^{ \beta \mu }.
$$

Contrast with a system of interacting bosons: [[Superfluid]] 