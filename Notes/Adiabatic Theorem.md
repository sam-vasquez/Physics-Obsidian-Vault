---
tags:
  - Quantum
---
Let $H(s(t))$ be a Hamiltonian that depends on a time-varying parameter $s(t)$. Assume that it always has discrete eigenstates, labeled by $m$, such that for all $t$,
$$
H(s) \ket{ \Psi_m (s) } = E_m(s) \ket{ \Psi_m(s) }.
$$
Assume that there is always an energy gap $\delta E > \Delta$ between all adjacent energy states at all times.
![[Adiabatic Gaps.jpg|300]]

Assume that the parameter $s$ varies very slowly in time over the interval $0 < t < T$, such that $s(t)$ can be written as an arbitrary smooth function $s(t) = f(t/T)$. 

Write the initial eigenstate as a function of $s(t)$, $\ket{ \Psi_m^{(0)} (t)}$ such that $H(t=0) \ket{ \Psi_m^{(0)} (t=0) } = E_m(t=0) \ket{ \Psi_m^{(0)} (t=0) }$.

The **adiabatic theorem** states that 
$$
\ket{ \Psi_m(t=T) } = e^{ i \phi } \ket{ \Psi_m^{(0)}(t=T) } + \mathcal{O}(1/T).
$$
In other words, the $m$ eigenstate of the system retains the same functional form, up to a phase.
See [[Proof of Adiabatic Theorem (Single Parameter)]].

This phase is given by the sum of a dynamical phase and a so-called [[Berry Phase]] $\gamma_m$ defined as
$$
\gamma_m = i \int_{s(0)}^{s(T)} ds \bra{ \Psi_m(s) } \frac{d}{ds} \ket{ \Psi_m (s) }.
$$

The adiabatic theorem can be generalized to mulitple parameters:
![[Adiabatic.jpg|200]]
If $H(\vec{R})$ is a function of $N$ parameters $\vec{R} = R_1(t),\ldots,R_N(t)$, and each path slowly varies, we get Berry phase 
$$
\gamma_m = \int_{\vec{R}_i}^{\vec{R}_f} d\vec{R} \cdot \vec{A}(\vec{R}),
$$
with vector potential $\vec{A}(\vec{R}) \equiv (A_1(\vec{R}),\ldots,A_N(\vec{R}))$ defined as
$$
\vec{A}_i(\vec{R}) = i \bra{ \Psi_m(\vec{R}) } \frac{\partial }{\partial R_i} \ket{ \Psi_m(\vec{R}) } = - \textrm{Im} \bra{ \Psi_m(\vec{R}) } \frac{\partial }{\partial R_i} \ket{ \Psi_m(\vec{R}) }.
$$

References:
[[(2023) Tarnopolsky 33659 Lecture Notes.pdf]]
Told to read [[Bernevig & Hughes]], [[Moessner & Moore]].