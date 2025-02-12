---
tags:
  - Stat-Mech
---
The method of steepest descent is a method for approximating contour integrals of the form 
$$
\int_C dz \; f(z) e^{ \lambda g(z) },
$$
for $\lambda$ large, $f$ and $g$ analytic. The method is to deform the contour into a new contour $C'$ such that 
1. $\textrm{Im}\{g(z)\}$ is constant on $C'$ ($g(z)$ has constant phase),
2. $C'$ passes through at least one zero of $g'(z)$.

The integral becomes 
$$
e^{ i \lambda \textrm{Im}\{g(z)\} } \int_{C'} dz \; f(z) e^{ \lambda \textrm{Re}\{ g(z) \} },
$$
and can now be evaluated using [[Laplace's Method]].


It's called "steepest descent" because the contours of constant phase are equivalent to the contours of steepest descent: writing $g(z = x + iy) = u(z) + i v(z)$, by the [[Cauchy-Riemann Equations]],
$$
\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \frac{\partial u}{\partial y} = - \frac{\partial v}{\partial x},
$$
and thus
$$
\frac{\partial u}{\partial x} \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y} \frac{\partial v}{\partial y} = \nabla u \cdot \nabla v = 0.
$$
Since $\nabla v$ is orthogonal to the level set $v = \textrm{Const.}$ (see [[Gradient is Orthogonal to Level Set]]), and $\nabla u$ points in the direction of the steepest descent (ascent) of $u$ (see [[Gradient Points in Steepest Direction]]), the contours of constant phase $v$ match the contours of steepest direction of $u$.

