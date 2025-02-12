---
tags:
  - Probability-Theory
---
Given a non-negative random variable $X \geq 0$, for any positive constant $k > 0$, the following inequality holds:
$$
\mathbb{P}(X \geq k) \leq \frac{\mathbb{E}[X]}{k}.
$$

**Proof:**
Let $X$ be continuous. The discrete case will work exactly the same way.

$$
\begin{align}
\mathbb{E}[X] &= \int_0^\infty dx \; x f_X(x) \\
&= \left( \int_0^k dx \; x f_X(x) \right) + \left( \int_k^\infty dx \; x f_X(x) \right) \\
&\geq \int_k^\infty dx \; x f_X(x) \\
&\geq \int_k^\infty dx \; k f_X(x) \\
&= k \int_k^\infty dx \; f_X(x) \\
&= k \mathbb{P}(X \geq k).
\end{align}
$$
$\blacksquare$
