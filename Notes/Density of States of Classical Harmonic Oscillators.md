---
tags:
  - Stat-Mech
---
The system of $N$ coupled classical harmonic oscillators can have its density of states written as 
$$
\Omega(E) = \prod_i \left( \int \frac{dx_i dp_i}{2\pi} \right) \delta\left( E - \sum_j \left[ \frac{p_j^2}{2m} + \frac{k x_j^2}{2} \right] \right).
$$
The square sum in the delta function suggests that this can be simplified with a transformation to spherical coordinates.

Making the change of variables
$$
y_{2i-1} = \frac{1}{\sqrt{ 2mE }}p_i, \quad y_{2i} = \sqrt{ \frac{k}{2E} } x_i,
$$
this becomes
$$
\begin{align}
\Omega(E) &= \prod_i \left( \int \frac{2E}{2\pi} \sqrt{ \frac{m}{k} } dy_{2i-1} dy_{2i} \right) \delta \left(   E - \sum_j E y_j^2  \right) \\ \\
&= \left( 2 \sqrt{ \frac{m}{k} } E \right)^N \prod_i \left( \int dy_i \right) \frac{1}{E} \delta \left(  1 - \sum_j y_j^2  \right) \\
&= \left( 2\sqrt{ \frac{m}{k}} E \right)^N \cdot \frac{1}{E} \cdot \int \delta\left( 1 - \sum_j y_j^2 \right)  \\
&= \left( 2 \sqrt{ \frac{m}{k} } E \right)^N \cdot \frac{1}{E} \cdot \mathcal{A}(2N),
\end{align}
$$
where $\mathcal{A}(2N)$ is the area of the unit sphere in $2N$ dimensions.
