---
tags:
  - Probability-Theory
---
When the transition matrix $P$ of a Markov chain with state space $\mathcal{X}$ is symmetric, that is when $P_{ij} = P_{ji}$ for all $x_i,x_j \in \mathcal{X}$, the uniform distribution on $\mathcal{X}$ is stationary for $P$.

**Proof:**
The uniform distribution on $\mathcal{X}$ with $|\mathcal{X}| = n$ is $u_i = \frac{1}{n}$ for all $x_i \in \mathcal{X}$.
It can be directly confirmed that $u$ is stationary on symmetric $P$.
$$
(uP)_j = \sum_i u_i P_{ij} = \frac{1}{n}\sum_i P_{ij} = \frac{1}{n}\sum_i P_{ji} = \frac{1}{n} = u_j,
$$
where the final sum is over a row of $P$ and therefore equal to 1.