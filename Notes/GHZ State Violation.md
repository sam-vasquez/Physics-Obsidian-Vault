---
tags:
  - Quantum-Information
---
This version of [[Bell's Theorem]] generalizes the local hidden variable violation of the [[CHSH Inequality]] to three particles:

The three particle version of the Bell states are the GHZ states. At least one is known as 
$$
\ket{ \textrm{GHZ} } = \frac{\ket{ 000 } + \ket{ 111 }}{\sqrt{ 2 }}.
$$
This state is the eigenstate of the following four mutually commuting observables:
$$
\begin{align*}
X \otimes X \otimes X \ket{ \textrm{GHZ} } &= (+1) \ket{ \textrm{GHZ} } \\
Y \otimes Y \otimes X \ket{ \textrm{GHZ} } &= (-1) \ket{ \textrm{GHZ} } \\
Y \otimes X \otimes Y \ket{ \textrm{GHZ} } &= (-1) \ket{ \textrm{GHZ} } \\
X \otimes Y \otimes Y \ket{ \textrm{GHZ} } &= (-1) \ket{ \textrm{GHZ} }.
\end{align*}
$$
(The observables commuting because they each involve pairs of differing operators, and the phases of $i$ introduced cancel out.)

If this were regarded as a classical theory, it would be assumed that there is a definite value for each observable,
$$
x_1(\lambda) x_2 (\lambda) x_3 (\lambda) = 1, y_1 y_2 x_3 = y_1 x_2 y_3 = x_1 y_2 y_3 = -1,
$$
$x_i,y_i = \pm 1$.
But when all three are multiplied together,
$$
x_1 x_2 x_3 y_1 y_2 x_3 y_1 x_2 y_3 x_1 y_2 y_3 = (x_1 x_2 x_3)^2 (y_1 y_2 y_3)^2 = -1.
$$
This implies that one product is complex-valued, which contradicts their values of $\pm1$. 

This can be formulated as a three-player game. 