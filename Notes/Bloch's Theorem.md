---
tags:
  - Quantum
---
[!thm] Bloch's Theorem
Consider a Hamiltonian that has Bravais lattice symmetry: That is, for any arbitrary lattice vector $\vec{R}_{\vec{n}} = n^\alpha e_\alpha$, $H = \frac{p^2}{2m} + U(\vec{r})$ with $U(\vec{r} + \vec{R}_{\vec{n}}) = U(\vec{r})$. Bloch's theorem states that eigenfunctions of this Hamiltonian can be written as "Bloch waves",
$$
\psi_{\vec{k}}(\vec{r}) = e^{ i \vec{k} \cdot \vec{r} } u_\vec{k} (\vec{r}).
$$

**Proof:**
Define a translation operator $T_{\vec{R}_\vec{n}}$ that maps $f(\vec{r})$ to $f(\vec{r} + \vec{R}_\vec{n}) = f(\vec{r} + n^\alpha \vec{e}_\alpha)$.
It is evident that $T_{R+R'} = T_R T_{R'} = T_{R'} T_{R}$. All translation operators commute.
Since the potential function has lattice symmetry, $\left[ H, T_R \right] = 0$.
So $H$ and every translation operator commute, they are simultaneously diagonalizable. 
A good ansatz for the eigenvalues might be an exponential.
Consider $e^{ 2\pi i n^\alpha \theta_\alpha}$ as an ansatz. 