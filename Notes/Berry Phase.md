---
tags:
  - Quantum
---
For energy eigenstate $\ket{ \Psi_m(s) }$ of an [[Adiabatic Theorem|adiabatically evolving Hamiltonian]] under a single adiabatic parameter $s$, the Berry phase is 
$$
\gamma_m = i \int_{s_i}^{s_f} ds \bra{ \Psi_m(s) } \frac{d}{ds} \ket{ \Psi_m(s) }.
$$
Note that the Berry phase does not explicitly depend on time, it is independent of the functional form of $s(t)$.

Note: Because $\braket{ \Psi_m(s) | \Psi_m(s) } = 1$, take its derivative knowing it to be zero, 
$$
\left( \frac{d}{ds} \bra{ \Psi_m(s) } \right) \ket{ \Psi_m(s) } + \bra{ \Psi_m(s) } \frac{d}{ds} \ket{ \Psi_m (s)} = \left( \bra{ \Psi_m(s) } \frac{d}{ds} \ket{ \Psi_m(s) } \right)^* + \bra{ \Psi_m(s) } \frac{d}{ds} \ket{ \Psi_m(s) } = 0,
$$
so $\bra{ \Psi_m(s) } \frac{d}{ds} \ket{ \Psi_m(s) }$ is purely imaginary and thus the Berry phase is purely real.

For a Hamiltonian of $N$ adiabatic parameters $\vec{R} = R_1, \ldots, R_N$, the Berry phase is
$$
\gamma_m = \int_{\vec{R}_i}^{\vec{R}_f} d\vec{R} \cdot \vec{A}(\vec{R}),
$$
with vector potential $\vec{A}(\vec{R}) \equiv (A_1(\vec{R}),\ldots,A_N(\vec{R}))$ defined as
$$
\vec{A}_i(\vec{R}) = i \bra{ \Psi_m(\vec{R}) } \frac{\partial }{\partial R_i} \ket{ \Psi_m(\vec{R}) } = - \textrm{Im} \bra{ \Psi_m(\vec{R}) } \frac{\partial }{\partial R_i} \ket{ \Psi_m(\vec{R}) }.
$$
This supports the interpretation of the Berry phase as a connection.