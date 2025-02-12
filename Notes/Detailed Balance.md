---
tags:
  - Probability-Theory
---
Consider a probability distribution $\pi$ on a finite state space $\mathcal{X}$. The equations
$$
\pi_i P_{ij} = \pi_j P_{ji} \textrm{ for all } x_i, x_j \in \mathcal{X}
$$
are called the *detailed balance equations* for the transition matrix $P$.

If $P$ is the transition matrix of a [[Markov Chains|Markov chain]], then $\pi$ satisfying these equations is automatically [[Stationary Distributions|stationary]] for $P$. This provides a simple way to verify that a given distribution is stationary. [[Detailed Balance Solution is Stationary|Proof]].

