---
tags:
  - Probability-Theory
---
The acceptance probability quoted in the [[Metropolis-Hastings Algorithm]] is 
$$
a_{ij} = \min \{ 1, \frac{\pi_j\Psi_{ji}}{\pi_i \Psi_{ij}} \}.
$$
We wish to show that beyond simply allowing the Metropolis-Hastings transition matrix to satisfy the [[Detailed Balance]] conditions for $\pi$, this choice of $a_{ij}$ also does so efficiently.

To see where this acceptance probability came from, first assume a general acceptance probability $a_{ij}$. For $P$ to have stationary distribution $\pi$, it is sufficient that detailed balance holds, i.e. that 
$$
\pi_i \Psi_{ij} a_{ij} = \pi_j \Psi_{ji} a_{ji}
$$
for all $i \neq j$. 
Ideally we want to maximize $a_{ij}$, because rejecting moves is wasteful and it is obvious that it slows down convergence. 
Because $a_{ij}$ is a probability, it must at least be bounded by $1$. Therefore the following must hold:
$$
\begin{align}
\pi_i \Psi_{ij} a_{ij} &\leq \pi_j \Psi_{ji}  \\
\pi_i \Psi_{ij} a_{ij} &\leq \pi_i \Psi_{ij}.
\end{align}
$$
Isolating $a_{ij}$ in both inequalities shows that $a_{ij}$ is bounded above by $\min \{ 1, \frac{\pi_j\Psi_{ji}}{\pi_i \Psi_{ij}} \}$. 