---
tags:
  - CM
---
For canonical momenta $p_i = \frac{\partial \mathcal{L}}{\partial \dot{q_i}}$, where $\mathcal{L} = \mathcal{L}(q_i,\dot{q}_i, t)$, the Legendre transform is
$$
\mathcal{H}(p_i,q_i,t) = \sum_i p_i \dot{q}^i - \mathcal{L}(p_i, q_i, t).
$$
In terms of differential forms,
$$
\mathcal{H} = \mathcal{L} - p\dot{q},
$$
$$
d\mathcal{H} = d\mathcal{L} - d(p\dot{q}) = 
$$

Legendre transform of convex function is also convex.

Efficient calculation process:
- Calculate canonical momenta $p(q,\dot{q},t)$.
- Solve for $\dot{q}(p,q,t)$.
- Solve for $\mathcal{L}(p,q,t)$.
- Plug $q(p,q,t)$ and $\mathcal{L}(p,q,t)$ into definition of $\mathcal{H}$.
