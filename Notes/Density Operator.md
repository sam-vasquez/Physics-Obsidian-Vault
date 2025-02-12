---
tags:
  - Quantum
---
Quantum mechanical states assume complete knowledge over the entire physical
system described by those states, they describe so-called *pure states*. But
that's rare. The density operator compensates for this by formalizing the
concept of statistical ensembles of multiple interacting quantum systems to
describe **mixed states**.

For an ensemble of systems each described by the same quantum system $A$, with classical probability $p_n$ of being in the pure state $\ket{ \psi_n }$, its density operator is
$$
\rho = \sum_n p_n \ket{ \psi_n } \bra{ \psi_n }.
$$
Measurement of an observable in this ensemble is given by an average of expectation values over all pure states, 
$$
\begin{align}
\langle \mathcal{O} \rangle &= \sum_n p_n \bra{ \psi_n } \mathcal{O} \ket{ \psi_n } \\
&= \sum_n p_n \textrm{Tr}(\bra{ \psi_n }\mathcal{O}\ket{ \psi_n }) \\
&= \sum_n p_n \textrm{Tr}(\ket{ \psi_n }\bra{ \psi_n }\mathcal{O}) \\
&= \textrm{Tr}\left( \sum_n p_n \ket{ \psi_n }\bra{ \psi_n }\mathcal{O} \right) \\
&= \textrm{Tr} (\rho \mathcal{O}).
\end{align}
$$
(You may want to divide by $\textrm{Tr} \rho$ if the density operator wasn't normalized.)

The [[Canonical Ensemble]] for quantum systems assigns the probability $e^{-\beta E}$ to each pure state describing the ensemble at a given temperature $\beta$. That is to say, the canonical ensemble has density operator
$$
\rho =\frac{1}{Z} \sum_i e^{-\beta E_i} \ket{ \psi_i } \bra{ \psi_i },
$$
where $\ket{ \psi_i }$ are specifically the energy eigenstates. 
See problem 16 of stat mech hw for an example of thermodynamics of particles with their spins coupled to a magnetic field.

The density operator can be applied to 