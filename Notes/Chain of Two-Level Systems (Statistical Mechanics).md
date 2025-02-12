---
tags:
  - Stat-Mech
---
Title was chosen for lack of a better name.

Let $N$ non-interacting atoms reside on a 1D lattice, each with two energy states, 0 or $\epsilon$.

This is the most prototypical example of a common class of statistical mechanics problems. For example, it comes up in the physisorption problem from stat mech hw. 

The [[Canonical Ensemble]] partition function decomposes into one for each atom:
$$
\begin{align*}
Z = \sum_v e^{ -\beta E_v } &= \sum_{n_1=0,1} \ldots \sum_{n_N = 0,1} e^{ -\beta (n_1 + \ldots + n_N) \epsilon } \\
&= \left( \sum_{n=0,1} e^{ -\beta n \epsilon } \right)^N\\
&= Z_1^N.
\end{align*}
$$
Each sum comes out to 
$$
Z_1 = 1 + e^{ -\beta \epsilon },
$$
so the full partition function is
$$
Z = (1 + e^{ -\beta \epsilon })^N.
$$

The [[Internal Energy (Mechanical System)]] is
$$
\langle E \rangle = N \epsilon \frac{e^{-\beta \epsilon}}{1 + e^{-\beta \epsilon}} = N \epsilon P(\epsilon).
$$
In the limit of zero temperature, the average energy is 0, no atoms are excited. In the limit of high temperature, the average energy is $\frac{N\epsilon}{2}$, half of all atoms are excited.


The [[Density of States]] for a given energy $E = m\epsilon$, that is, for $m$ excited atoms, is simply
$$
\Omega(\epsilon) 
= 
\begin{pmatrix}
N \\
m
\end{pmatrix}
= \frac{N!}{m!(N-m)!}.
$$
So by Stirling's approximation, the [[Entropy]] is approximately
$$
\begin{align*}
S(E) &= k_B \ln \Omega(\epsilon)\\ 
&\approx k_B (N \ln N - N - m \ln m + m - (N-m) \ln (N-m) + N-m).
\end{align*}
$$
The temperature is 
$$
\begin{align*}
\frac{1}{T} &= \left( \frac{\partial S}{\partial E} \right)\\ 
&= \frac{1}{\epsilon} \left( \frac{\partial S}{\partial m}  \right)\\
&\approx \frac{k_B}{\epsilon} \ln \left( \frac{N-m}{m} \right) \\
&= \frac{k_B}{\epsilon} \ln \left( \frac{N\epsilon}{E} - 1 \right).
\end{align*}
$$
In its large $N$ limit, this is consistent with the result derived earlier, justifying the approximate equality between $E$ and $\langle E \rangle$.

Entropy is maximized when half of the atoms are excited, just as we saw that the high-temperature limit of energy corresponds to half of the atoms being excited. Together with the low-temperature behavior that no states are excited, this demonstrates the [[Energy Minimization Principles]]: At a given temperature, the [[Free Energy]] $F = E - TS$ is minimized, not either of $E$ or $S$ individually. At low temperature, the energy dominates and is minimized, at high temperature, the entropy dominates and is maximized.

It would seem that the entropy and temperature violate the [[Second Law of Thermodynamics]], since adding energy in the form of exciting additional atoms will lower the entropy, implying negative temperature. This isn't physical, and indeed this system is not at all physical. There's typically no restriction on energy levels, and configurational entropy would typically keep increasing without bound. Even if this were a physical system, it's worth noting that there's no way to take advantage of negative temperature because of [[Energy Stability Conditions|free energy stability]].