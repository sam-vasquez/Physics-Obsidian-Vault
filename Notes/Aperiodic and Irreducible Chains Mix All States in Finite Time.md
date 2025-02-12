---
tags:
  - Probability-Theory
---
If $P$ is aperiodic and irreducible, then there is an integer $r_0$ such that $P^r_{ij} > 0$ for all $x_i, x_j \in \mathcal{X}$ and $r \geq r_0$.

**Proof:**
For each $x_i \in \mathcal{X}$, recall that $\mathcal{T}(x_i) = \{ t \geq 1 | P^t_{ii} > 0 \}$. Since the chain is aperiodic, the gcd of $\mathcal{T}(x_i)$ is 1.

The set $\mathcal{T}(x_i)$ is closed under addition: If one cycle can return to $x_i$ in $n$ steps and another in $m$ steps, there is a cycle that returns to $x_i$ in $n+m$ steps. Formally, if $n,m \in \mathcal{T}(x_i)$, then $P^{n+m}_{ii} = \sum_j P^n_{ij} P^m_{ji} \geq \sum_j P^n_{ii} P^m_{ii} > 0$.

It is known from number theory that whenever a set of positive integers has gcd $g$, every multiple of $g$ exceeding a particular value can be represented as a non-negative integer combination of elements of the set. See a partial proof [[Schur's Theorem (Number Theory)|here]]. Since our set has gcd 1, it must be the case that every single integer larger than some value $t(x_i)$ can be represented as a linear combination of elements of $\mathcal{T}(x_i)$, and therefore by closure they all belong to $\mathcal{T}(x_i)$. 

By irreducibility, we know that we can reach any other state $x_j$ from $x_i$ in up to $n(x_i)$ additional steps, making $r$ at least $t(x_i) + n(x_i)$ And we can repeat this procedure for all $x_i$ to get a single value $r_0$ for which every state $x_j$ has a chance of being reached from state $x_i$ at every time step $r \geq r_0$.