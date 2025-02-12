---
tags:
  - Probability-Theory
---
The [[Metropolis-Hastings Algorithm|Metropolis-Hastings]] transition matrix $P$ is given by
$$
P_{ij} = 
\begin{cases}
\Psi_{ij} \cdot \min \{ 1, \frac{\pi_j\Psi_{ji}}{\pi_i \Psi_{ij}} \} & \textrm{if } i \neq j  \\
1 - \sum_{x_k \in \mathcal{X} | z \neq i} P_{ik} & \textrm{if } i = j.
\end{cases}
$$
To prove that $P$ has stationary distribution $\pi$, it is sufficient to show that $P$ satisfies the [[Detailed Balance]] equations,
$$
\pi_i P_{ij} = \pi_j P_{ji} \quad \textrm{for all } x_i, x_j \in \mathcal{X}.
$$
It is trivial that this is satisfied when $i = j$. 
Let $i \neq j$. 
Without loss of generality, let $\min \{ 1, \frac{\pi_j\Psi_{ji}}{\pi_i \Psi_{ij}} \} = 1$. Consequently, $\min \{ 1, \frac{\pi_i \Psi_{ij}}{\pi_j\Psi_{ji}} \} = \frac{\pi_i \Psi_{ij}}{\pi_j\Psi_{ji}}$.
Therefore $\pi_i P_{ij} = \pi_i \Psi_{ij}$ and $\pi_j P_{ji} = \pi_j \Psi_{ji} \frac{\pi_i \Psi_{ij}}{\pi_j\Psi_{ji}} = \pi_i \Psi_{ij}$, and the two sides are equal. 