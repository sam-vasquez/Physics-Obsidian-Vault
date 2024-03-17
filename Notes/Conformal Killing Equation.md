---
tags:
  - theorem
  - Conformal-Bootstrap
  - 2d-CFT
---

> [!thm] The conformal Killing equation
> For any infinitesimal conformal coordinate transformation of the form $x^\mu \rightarrow x^\mu + \epsilon^\mu$, $\epsilon^\mu$ satisfies the following equation:
> $$
>\partial_\mu \epsilon_\nu + \partial_\nu \epsilon_\mu = \frac{2}{d} (\partial \cdot \epsilon) \eta_{\mu\nu}.
>$$

^000401

#### Proof:
Consider an infinitesimal transformation $x^\mu \rightarrow x^\mu + \epsilon^\mu$. To leading order in $\epsilon$,

$$
\begin{align}
 \Omega^2(x) \eta_{\mu\nu} = \eta'_{\mu\nu} &= \frac{\partial x^\alpha}{\partial x'^\mu} \frac{\partial x^\beta}{\partial x'^\nu} \eta_{\alpha\beta}\\ \\
 & = (\delta^\alpha_\mu - \partial_\mu \epsilon^\alpha)(\delta^\beta_\nu - \partial_\nu \epsilon^\beta)\eta_{\alpha\beta}\\ \\
 & = \eta_{\mu\nu} - (\partial_\mu \epsilon_\nu + \partial_\nu \epsilon_\mu).
\end{align}
$$

^a82407

Contract both sides with $\eta^{\mu\nu}$:

$$
\begin{align}
 \Omega^2(x) d &= d - 2 \partial \cdot \epsilon\\ \\
 \implies \Omega^2(x) &= 1 - \frac{2}{d}\partial \cdot \epsilon
.
\end{align}
$$

^45fa92

Combining [[#^a82407]] and [[#^45fa92]], we get the conformal Killing equation. $\blacksquare$