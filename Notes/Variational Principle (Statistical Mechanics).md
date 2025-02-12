---
tags:
  - Stat-Mech
---
The variational principle from statistical mechanics is very similar to the [[Variational Principle (Quantum Mechanics)|variational principle from quantum mechanics]].

For any positive continuous probability distribution $P(x)$, with
$$
\int dx P(x) = 1, \quad \left< f(x) \right> = \int dx P(x) f(x),
$$
it is true that
$$
\left< e^{ -x } \right> \geq e^{ - \left< x \right> }.
$$
That is, the average of values of $e^{ -x }$ at two points $x_1$ and $x_2$ is always larger than the value of $e^{ -x }$ at the average of $x_1$ and $x_2$. This is true because the exponential function is a convex function.

Let $\mathcal{H}$ be a Hamiltonian for which we would like to compute the partition function $Z$ but probably cannot. As a model for $\mathcal{H}$, we consider a family of Hamiltonians $\mathcal{H}(a)$ for which we can compute the partition function. Let $\left< f \right>_a$ be the expectation value of $f$ with respect to the partition function of $\mathcal{H}_a$. Then
$$
Z = \int e^{ -\beta \mathcal{H} } = \int e^{ -\beta \mathcal{H}_a } e^{ -\beta (\mathcal{H} - \mathcal{H}_a) }
$$