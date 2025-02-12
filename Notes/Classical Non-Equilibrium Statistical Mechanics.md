---
tags:
  - Stat-Mech
---
In classical mechanics, [[Non-Equilibrium Statistical Mechanics]] is treated using a generalized [[Ensemble]] instead of the [[Canonical Ensemble]]. In particular, the ensemble does not necessarily satisfy the [[Ergodic System]] condition. 

The ensemble is typically identified with some driving force $F(r^N,p^N)$. Following the setup in [[Ergodic System]], the [[Ensemble Average]] of the observable $A$ at time $t$ is 
$$
\langle A(t) \rangle = \int dr^N \int dp^N F(r^N,p^N) A(t|r^N,p^N),
$$
and analogous for fluctuations $\langle \delta A(t) \rangle$.

We study the fluctuations using the [[Classical Time Correlation Function]]
$$
K(t) = \langle \delta A(0) \delta A(t) \rangle.
$$
It is possible to express this correlation function in frequency space thanks to the [[Wiener-Khinchin Theorem]], yielding the [[Power Spectral Density]]
$$
S_A(\omega) = \int_{-\infty}^\infty dt\; e^{ i \omega t } K(t),
$$
or for systems where only positive frequency is physical, the one-sided PSD
$$
S_A(\omega) = 2 \int_0^\infty dt \cos(\omega t) K(t).
$$

