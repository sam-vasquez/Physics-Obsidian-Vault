---
tags:
  - ML
---
Gradient descent with momentum (GDM) is an extension of [[Gradient Descent]] that introduces a new parameter and term into the algorithm that allows the descent algorithm to build up momentum in its descent and help it skip over small fluctuations.

Let $\mathcal{L}\left[ w^i \right]$ be the function we are attempting to optimize. We can descend along its gradient starting from some initial point with the following approximation:
$$
\Delta \mathcal{L} \approx \partial_{w_i} \mathcal{L} \cdot \Delta w_i.
$$

Simple gradient descent chooses $\Delta w_{i;t}$ at each iteration $t$ according to the following equations:
$$
\Delta w_{i; t} = - \eta \partial_{w_i} \mathcal{L}[\omega_{i; t}]
$$
$$
w_{i; t+1} = w_{i; t} - \Delta w_{i; t}
$$

GDM adds an extra term to $\Delta w_{i; t}$:

$$
\Delta w_{i; t} = \gamma \Delta w_{i; t-1} + \eta \partial_{w_i} \mathcal{L}[w_{i; t}]
$$
with $0 \leq \gamma \leq 1$ the momentum parameter.

A slight modification of GDM is to calculate the gradient one step ahead, at the expected value of the parameters given our current momentum. This is called Nesterov Accelrated Gradient (NAG), and the NAG update rule is
$$
\Delta w_{i; t} = \gamma \Delta w_{i; t-1} + \eta \partial_{w_i} \mathcal{L}[w_{i; t} + \gamma \Delta w_{i; t-1}].
$$
