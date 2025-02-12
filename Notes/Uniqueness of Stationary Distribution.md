---
tags:
  - Probability-Theory
---
If $P$ is the transition matrix of an irreducible Markov chain, the stationary distribution is unique.

**Proof**:
Consider two stationary distributions $\pi_1$ and $\pi_2$.
Choose a state $x_i$ that minimizes $\pi_1{}_i/\pi_2{}_i$.
Because $\pi_1$ is a stationary distribution,
$$
\pi_1{}_i = \sum_j \pi_1{}_j P_{ji}.
$$
We can use the property that state $x_i$ minimizes $\pi_1{}_i/\pi_2{}_i$ by introducing a factor of $\pi_2{}_j/\pi_2{}_j$ into the sum:
$$
\pi_1{}_i = \sum_j \pi_1{}_j P_{ji} = \sum_j \frac{\pi_1{}_j}{\pi_2{}_j} P_{ji} \pi_2{}_j.
$$
Stationary distributions, like any stochastic vector, are non-negative, so the sum over proportions of the two vector components must be greater than its minimum proportion $\pi_1{}_i/\pi_2{}_i$. 
$$
\pi_1{}_i = \sum_j \pi_1{}_j P_{ji} = \sum_j \frac{\pi_1{}_j}{\pi_2{}_j} P_{ji} \pi_2{}_j \geq \frac{\pi_1{}_i}{\pi_2{}_i} \sum_j P_{ji} \pi_2{}_j.
$$
$\pi_2$ is a stationary distribution, so the last term is $\frac{\pi_1{}_i}{\pi_2{}_i}\pi_2{}_i$, or $\pi_1{}_i$. But since both sides are $\pi_1{}_i$, the inequality must be equality. The equality can only hold if 
$$
\frac{\pi_1{}_j}{\pi_2{}_j} = \frac{\pi_1{}_i}{\pi_2{}_i} \textrm{ for all } i \textrm{ such that } P_{ji} > 0. 
$$
Irreducibility implies that we can repeat this argument for every power of $P$ (for which $\pi_1$ and $\pi_2$ are also stationary) until it holds for every state $x_i$. 

This proves that $\pi_1$ and $\pi_2$ are equivalent up to a rescaling, and therefore all stationary distributions of an irreducible Markov chain are equivalent upon normalization.