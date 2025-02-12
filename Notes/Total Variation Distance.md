---
tags:
  - Probability-Theory
---
Total variation distance is one of many distance measures for probability distributions. It is helpful for characterizing the distance between two distributions when considering, say, convergence of a sequence of random variables.

The total variation distance between two probability distributions $\mu$ and $\nu$ on a finite space $\mathcal{X}$ is defined by
$$
||\mu - \nu||_{TV} = \frac{1}{2} \sum_{x \in \mathcal{X}} |\mu(x) - \nu(x)|.
$$

(We divide by $1/2$ because the furthest apart any two distributions can be is two.)

One example of a use for this notion of distance is to compute the distance between a distribution $\mu \sim X_n$ of a Markov chain on state space $\mathcal{X}$ and the Markov chain's stationary distribution $\pi$. In doing so, we can make a statement about the [[Convergence Theorem (Markov Chains)|convergence of a Markov chain's distribution to its stationary distribution]].