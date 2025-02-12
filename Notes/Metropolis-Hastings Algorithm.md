---
tags:
  - Probability-Theory
---
The Metropolis-Hastings algorithm answers the following question: Given some [[Markov Chains|Markov chain]] with state space $\mathcal{X}$ and an arbitrary [[Stationary Distributions|stationary distribution]], can the chain be modified so that the new chain has a desired stationary distribution $\pi$?

The algorithm proceeds as follows:

1. Pick any irreducible transition matrix $\Psi$ over $\mathcal{X}$. Ideally, choose a matrix that has a stationary distribution close to $\pi$.
2. Choose any initial state $x_i \in \mathcal{X}$.
3. Run the Markov chain and generate a candidate, $x_j \in \mathcal{X}$.
4. Instead of advancing to state $x_j$ immediately, only do so with an acceptance probability $a_{ij} = \min \{ 1, \frac{\pi_j\Psi_{ji}}{\pi_i \Psi_{ij}} \}$.
5. If the candidate $x_j$ is accepted, take it as the next state in the sequence. If not, take the original state $x_i$ as the next state in the sequence.

This process results in a new transition matrix $P$ with probabilities
$$
P_{ij} = 
\begin{cases}
\Psi_{ij} \cdot \min \{ 1, \frac{\pi_j\Psi_{ji}}{\pi_i \Psi_{ij}} \} & \textrm{if } i \neq j  \\
1 - \sum_{k \neq i} P_{ik} & \textrm{if } i = j.
\end{cases}
$$
There are two steps to prove that this algorithm works: [[Metropolis-Hastings Has Desired Stationary Distribution|One]], that $\pi$ really is a stationary distribution for the Metropolis-Hastings transition matrix, and [[Metropolis-Hastings Acceptance Probability is Optimal|two]], that the acceptance probability $a_{ij}$ described here is optimal.
