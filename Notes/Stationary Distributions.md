---
tags:
  - Probability-Theory
---
Consider a [[Markov Chains|Markov chain]] with finite state space $\mathcal{X}$ and transition matrix $P$. Distributions over $\mathcal{X}$ can be updated by the chain by interpreting them as a stochastic vector and right-multiplying by $P$. We might want to know if a given distribution will converge to a limit as $n \rightarrow \infty$. Any limit distribution $\pi$ would have to satisfy $\pi P = \pi$, which we call a **stationary distribution**.

A stationary distribution exists for all finite transition matrices. This is thanks to the Perron-Frobenius theorem for non-negative matrices, where the Perron-Frobenius eigenvector is non-negative with real spectral radius 1. However, in general it is not unique. The stationary distribution is only unique when the Markov chain is [[Markov Chains#^2ce78f|irreducible]] (which we often assume). See proof [[Uniqueness of Stationary Distribution|here]].

The stationary distribution can be explicitly constructed in terms of the long-term fraction of time that the chain spends in each state. See 1.5.2, 1.5.3.

Irreducibility is necessary to ensure that there is a unique stationary distribution that the chain can converge to, but it is not sufficient. A periodic chain could get stuck in a cycle, after all. Aperiodicity is required to prove convergence. See the proof [[Convergence Theorem (Markov Chains)|here]].

A simple way to verify that a particular distribution is stationary can be to check for detailed balance. Any distribution that satisfies the detailed balance equations is stationary (although it is not a necessary requirement). See the detailed balance equations [[Detailed Balance|here]].

If the transition matrix $P$ is symmetric (i.e. if $P_{ij} = P_{ji}$), the uniform distribution on $\mathcal{X}$ is stationary. [[Symmetric Markov Chain has Uniform Stationary Distribution|Proof]].

Here we have described how to find the limiting stationary distribution of a given irreducible and aperiodic transition matrix. Often we are interested in the inverse question: Given a probability distribution $\pi$, can we find a transition matrix for which $\pi$ is its stationary distribution? This is the motivation for the [[Metropolis-Hastings Algorithm]], the basis of [[Markov Chain Monte Carlo]].




Good reading about stationary distributions:
https://homepages.math.uic.edu/~furman/preprints/whatis-published.pdf