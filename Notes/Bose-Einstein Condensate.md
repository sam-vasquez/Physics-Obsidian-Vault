---
tags:
  - Stat-Mech
---
The Bose-Einstein condensate is a phase transition of non-interacting massive bosons, driven by particle density.

Its existence is derived by considering the [[Ideal Bose Gas]] for massive bosons.
$$
N = \int d\epsilon g(\epsilon) \langle n(\epsilon) \rangle = \frac{V}{\lambda_{\textrm{th}}^d} g_{d/2}(z).
$$
This expression is bounded from above for particular dimensions. For example, in 3 dimensions, the maximum value of the [[Polylogarithm]] is $2.612$. This non-physical restriction implies that bosons actually condense into the ground state, which has zero weight in the integral.
(Note that Bose-Einstein condensation does not happen in 2D systems. See Problem 36 for details.)

It is corrected by extracting the ground state explicitly.
$$
\begin{align}
N &= \langle n(\epsilon = 0) \rangle + \int d\epsilon g(\epsilon) \langle n(\epsilon) \rangle  \\
&= \frac{1}{z^{-1} e^{\beta 0} - 1} + \int_0^\infty d\epsilon g(\epsilon) \langle n(\epsilon) \rangle  \\
&= \frac{z}{1-z} + \frac{V}{\lambda^d} g_{d/2}(z).
\end{align}
$$
The first term $N_0$ is the number of particles in the ground state. The second term $N'$ is the number of particles in all other ground states, and below the critical temperature its value is approximately $N'_{\textrm{max}} = \frac{V}{\lambda^{d}}g_{d/2}(1)$.
(See Problem 36 and Pathria appendix F for rigorous arguments that justify extracting only the ground state and approximating the higher order states.)


The 3D Bose-Einstein condensate phase transition is in the same universality class as the 3D classical XY model.

The phase transition order parameter is the ground state wavefunction. Normally it normalizes to 1, but in BEC it normalizes to the number of particles in ground state. The broken symmetry is that it has to choose all the same quantum phase, and phase symmetry $U(1)$ is spontaneously broken. 