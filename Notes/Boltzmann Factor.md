---
tags:
  - Stat-Mech
---
The Boltzmann factor is the probability measure that gives the probability that a thermodynamic system will be in a particular microstate for a given temperature and energy. This is used to describe the [[Canonical Ensemble]].

The Boltzmann factor may be derived from the [[Microcanonical Ensemble]] as follows:

Consider a system that exchanges energy with a very large bath. The system and bath together have a fixed energy $E = E_{\textrm{bath}} + E_{\textrm{sys}}$, while the energy of each might fluctuate. The bath has a [[Density of States]] 
$$
\Omega_{\textrm{bath}} = \Omega(E_{\textrm{bath}},N,V) = \Omega(E - E_{\textrm{sys}},N,V).
$$

By the multiplicity of states, the bath and system collectively have a total density of states
$$
\Omega_{\textrm{bath}}(E - E_{\textrm{sys}}) \Omega_{\textrm{sys}}(E_{\textrm{sys}}).
$$
According to the microcanonical ensemble, the probability that the system has energy $E_{\textrm{sys}}$ is proportional to that density.

$$
\begin{align*}
P_{\textrm{sys}} &\propto \Omega_{\textrm{bath}}(E - E_{\textrm{sys}}) \Omega_{\textrm{sys}}(E_{\textrm{sys}}) \\
&= \Omega_{\textrm{sys}}(E_{\textrm{sys}}) e^{\ln \Omega_{\textrm{bath}}(E - E_{\textrm{sys}})}.
\end{align*}
$$
Because the bath can be taken as much larger than the system, a Taylor series expansion of $\ln \Omega_{\textrm{bath}}$ is justified:
$$
P_{\textrm{sys}} \propto e^{\ln \Omega - E_{\textrm{sys}} \frac{\partial }{\partial E} \ln \Omega}.
$$
Noting that the derivative of $\ln \Omega = S/k$ is $\frac{1}{T}$, and throwing out any irrelevant proportionality terms (they come back in the normalization), we get
$$
P \propto e^{ - \beta E }.
$$

