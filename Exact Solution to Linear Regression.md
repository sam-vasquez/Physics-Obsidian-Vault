---
tags:
  - ML
  - theorem
---
> [!thm] Exact Solution to Linear Regression
> The most optimal solution $\mathbf{w}$ to the least-squares fit in linear regression,
> $$
> \mathcal{L} \equiv (x^a{}_j w^j - y^a) (x_a{}_k w^k - y_a) = || \mathbf{Xw - y} ||_2^2,
> $$
> when the solution exists, is $\mathbf{w} = \mathbf{(X^T X)^{-1} X^T y}$.

#### Proof:

Differentiate $\mathcal{L}$ with respect to $w^i$ to get 

$$
\begin{align}
\partial_{w^i} \mathcal{L}  & = x^a{}_j \delta^{ij} x_{ak} w^k + x^a{}_j w^j x_{ak} \delta^i{}^k - y^a x_a{}_k \delta^i{}^k - x^a{}_j \delta^i{}^j y_a \\
 & = x^a{}^i x_a{}_k w^k + x^a{}_j w^j x_a{}^i - y^a x_a{}^i - x^a{}^i y_a \\
 & = (x^T)^i{}_a x^a{}_k w^k + (x^T)^i{}_a x^a{}_j w^j - (x^T)^i{}_a y^a - (x^T)^i{}_a y^a \\
 & = 2 \mathbf{X^T X w} - 2 \mathbf{X^T y} \\ 
\end{align}
$$

Setting to zero implies that $\mathbf{w} = \mathbf{(X^T X)^{-1} X^T y}$. $\blacksquare$