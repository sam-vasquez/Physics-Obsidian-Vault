---
tags:
  - theorem
  - Conformal-Bootstrap
---
> [!thm] Conformal Generators in $d\geq3$
> The generators of the conformal Lie algebra corresponding to the [[Conformal Killing Vectors in d≥3|conformal Killing vectors in $d\geq3$]] are as follows:
> $$
> \begin{align}
> P_\mu &= -i\partial_\mu \\
> M_{\mu\nu} &= -i \left( x_\nu \partial_\mu - x_\mu \partial_\nu \right)\\
> D &= -i x^\mu \partial_\mu \\
> K_\mu &= -i \left( 2(x \cdot \partial)x_\mu - x^2\partial_\mu \right).
> \end{align}
> $$

#### Proof:

See the derivation in [[Poincare Group#^ceba13|Poincare Group]] for the standard translation and rotation generators.

The dilatation generator is calculated trivially.

For the special conformal transformation:

Let $f'(x') = f(x^\mu + 2(x_\nu b^\nu)x^\mu - b^\mu x^2)$. Taylor expand to get 
$$
f(x^\mu + 2(x_\nu b^\nu)x^\mu - b^\mu x^2) \approx f(x) + 2(x \cdot b)x^\mu\partial_\mu f(x) - x^2b^\mu \partial_\mu f(x).
$$

Compare with the Taylor expansion of $f'(x') = e^{ i b \cdot K } f(x)$:
$$
\begin{align} 
ib^\mu K_\mu &= 2(x_\mu b^\mu)x^\nu \partial_\nu - x^2b^\mu\partial_\mu \\
\implies K_\mu &= -i\left( 2(x_\nu \partial^\nu)x_\mu - x^2\partial_\mu \right).
\end{align}
$$
