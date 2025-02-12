---
tags:
  - Stat-Mech
---
The canonical ensemble is an alternative to the [[Microcanonical Ensemble]] where the energy of the system is replaced with a surrogate intensive quantity, temperature. This is essentially the same motivation as using the Legendre transform to switch from the energy to the free energy.

The canonical ensemble is constructed from variables of particle number $N$, volume $V$, and temperature $T$ / inverse temperature $\beta$. 

The canonical ensemble is constructed by assigning a probability $e^{ - \beta E }$ to each [[Density of States|microstate]] $\Omega(E)$ and integrating over energy.
$$
\int dE \; \Omega(E) e^{ -\beta E } = \int dE \; e^{ S(E) - \beta E }
$$
See [[Notes/Boltzmann Factor|here]] for a derivation of this distribution using heat bath considerations.

These integrals will be very sharply peaked, letting us evaluate the integral by [[Method of Steepest Descent|steepest descent]]. The maximum of the integral occurs at 
See problem 14 of stat mech hw for details.


Almost every thermodynamic quantity can be obtained from derivatives of the partition function. 

In particular, the (logarithm of the) partition function is a generating function of statistical moments of energy. For example, the average [[Internal Energy (Mechanical System)]] is 
$$
\langle E \rangle = -\frac{\partial }{\partial \beta} \ln Z,
$$
and the fluctuations of the internal energy are 
$$
\langle (\delta E)^2 \rangle = \frac{\partial^2}{\partial \beta^2} \ln Z.
$$

Argument from Landau: Most fluctuations go like $\frac{\delta A}{A} \propto \frac{1}{\sqrt{ N }}$. This justifies treating all ensembles averages of observables like their corresponding thermodynamic [[State Variable]].

The canonical partition function naturally relates to the [[Free Energy]] as 
$$
F = - k_B T \ln Z.
$$

When the particle number is allowed to vary, the canonical ensemble generalizes to the [[Grand Canonical Ensemble]].

In fact, it's often more convenient to work with the grand canonical ensemble rather than the canonical ensemble, and only fix the particle number later by solving for the chemical potential $\mu$ that resotres $\sum \langle n_i \rangle = N$. This approach is notably used when studying the [[Ideal Bose Gas]]. The quantum statistics makes working with the canonical ensemble excessively difficult, but the grand canonical ensemble permits the use of the occupation number representation to remove all difficult combinatorics.