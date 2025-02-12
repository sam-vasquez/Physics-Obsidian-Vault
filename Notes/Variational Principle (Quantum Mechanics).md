---
tags:
  - Quantum
---
Let $\mathcal{H}$ be a Hamiltonian that we would like to diagonalize but probably cannot. Instead, consider a family of Hamiltonians $\mathcal{H}(a)$ that we can diagonalize, parameterized by $a$. Let $\ket{ 0 }$ be the (normalized) ground state of $\mathcal{H}$ with ground state energy $E_0$, and let $\ket{ a }$ be the (normalized) ground state of $\mathcal{H}(a)$ with ground state energy $E_a$. Then
$$
E_0 \leq E_a + \bra{ a } (\mathcal{H} - \mathcal{H_a}) \ket{ a }.
$$
If we minimize the right hand side with respect to $a$, we get a good model $\mathcal{H}(a)$ for $\mathcal{H}$ that is diagonalizable.
