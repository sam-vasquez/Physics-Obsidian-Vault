---
tags:
  - Probability-Theory
---
Given a state space $\mathcal{X}$ and transition matrix $P$, any probability distribution $\pi$ on $\mathcal{X}$ that satisfies the [[Detailed Balance]] equations 
$$
\pi_i P_{ij} = \pi_j P_{ji} \quad \textrm{for all } x_i, x_j \in \mathcal{X}
$$
is a stationary distribution for $P$.

**Proof**
Sum both sides over $j$:
$$
(\pi P)_i = \sum_j \pi_j P_{ji} = \sum_j \pi_i P_{ij} = \pi_i \sum_j P_{ij} = \pi_i, 
$$
where the last equality holds because $P$ is stochastic.

The detailed balance condition can be motivated by the first equality. We wish to find $\pi$ such that $\pi P = \pi$, or in components, $(\pi P)_i = \sum_j \pi_j P_{ji} = \pi_i$. One way to satisfy this equation is if the summed terms equal $\pi_i P_{ij}$, allowing us to sum over only a row of $P$ and get 1. 
