---
tags:
  - Quantum
---
The [[Adiabatic Theorem]] states that 
$$
\ket{ \Psi_m(t=T) } = e^{ i \phi } \ket{ \Psi_m^{(0)}(t=T) } + \mathcal{O}(1/T).
$$

**Proof:**

At any given moment of time $t$, the functions $\{ \ket{ \Psi_m (s(t))} \}$ comprise an orthonormal basis in Hilbert space. Write wavefunction coefficients $c_m(t)$:
$$
\ket{ \Psi(t) } = \sum_m c_m(t) \ket{ \Psi_m(s(t)) }, c_m(t) = \braket{ \Psi_m(s(t)) | \Psi(t) }.
$$
Plug into the [[Time-Dependent Schrodinger Equation]]:
$$
i \hbar \sum_m \left( \dot{c}_m(t) \ket{ \Psi_m(s(t)) } + c_m(t) \frac{d}{dt} \ket{ \Psi_m (s(t)) } \right) = \sum_m c_m(t) E_m(s(t)) \ket{ \Psi_m(s(t)) }.
$$
The $k$ component of this equation is
$$
i \hbar \dot{c}_k(t) + i \hbar \sum_m c_m(t) \bra{ \Psi_k (s(t)) } \frac{d}{dt} \ket{ \Psi_m(s(t)) } = c_k(t) E_k(s(t)).
$$
Solve this equation with the initial condition $c_n(t=0) = 1$, $c_{m\neq n} = 0$ ($c_k(0) = \delta_{kn}$) to determine the flow of the $n$ eigenstate under the Hamiltonian:
$$
i\hbar \dot{c}_k(0) + i\hbar \bra{ \Psi_k(0) } \frac{d}{dt} \ket{ \psi_n(0) } = \delta_{kn} E_k(0),
$$
Todo: Notes continue by assuming that $c_{k\neq n}(t) \ll 1$. Can this be defended? Presumably because the energy gaps are large for all $t$.