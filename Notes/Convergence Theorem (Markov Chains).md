---
tags:
  - Probability-Theory
  - todo
---
Suppose that $P$ is irreducible and aperiodic, with stationary distribution $\pi$. Then
$$
 \lim_{n\rightarrow\infty} || \textrm{dist}(X_n) - \pi ||_{TV} = 0.
$$


**Proof**
Since $P$ is irreducible and aperiodic, [[Aperiodic and Irreducible Chains Mix All States in Finite Time|there exists]] an $r$ such that $P^r$ has strictly positive entries.
$P$ being non-zero everywhere, we can say that for sufficiently small $\delta > 0$, we have
$$
P^r_{ij} \geq \delta \pi_j
$$
for all $x_i, x_j \in \mathcal{X}$.
Let $\Pi$ be the matrix with $|\mathcal{X}|$ rows, where each row is the row vector $\pi$. 
We could rewrite the inequality as 
$$
P^r = \delta \Pi + (1 - \delta)Q,
$$
$Q$ being the difference between $P^r$ and $\delta \Pi$, where a factor of $(1-\delta)$ has been taken out to make $Q$ stochastic. It is actually simpler to use $\theta = 1 - \delta$, making this
$$
P^r = (1-\theta)\Pi + \theta Q.
$$
We use induction to show that 
$$
P^{rk} = (1-\theta^k)\Pi + \theta^k Q^k.
$$
This would show that greater powers of $P$ do get closer to $\Pi$, as $\theta$ gets smaller and reduces $Q$'s contribution. 
The case $k=1$ trivially holds.
Assuming that $k = n$ holds,
$$
P^{r(n+1)} = P^{rn}P^r = (1-\theta^n)\Pi P^r + \theta^n Q^n P^r.
$$
As expected from the definition of the stationary distribution and easy to confirm, $\Pi P = \Pi$. Using that in the first term and expanding $P^r$ in the second term gives
$$
P^{r(n+1)} = (1-\theta^n)\Pi + (1-\theta)\theta^n Q^n \Pi + \theta^{n+1} Q^{n+1}. 
$$
It is also easy to confirm that $M \Pi = \Pi$ for every stochastic matrix $M$. Using this gives
$$
P^{r(n+1)} = (1-\theta^n)\Pi + (1-\theta)\theta^n \Pi + \theta^{n+1} Q^{n+1} = (1 - \theta^{n+1}) \Pi + \theta^{n+1} Q^{n+1},
$$
completing the induction proof.
Now we know that large powers of $P$ get closer to $\Pi$ in a controlled way, according to 
$$
P^{rk} = (1-\theta^k)\Pi + \theta^k Q^k.
$$
To make this explicit, iterate another $j$ times by multiplying each side by $P^j$, and rearrange:
$$
P^{rk+j} = (1-\theta^k)\Pi + \theta^k Q^k P^j,
$$
which implies that
$$
P^{rk+j} - \Pi = \theta^k (Q^k P^j - \Pi).
$$
For an arbitrary row $x$ of $P$, we can sum over row $x$ of both sides and divide by two to get the [[Total Variation Distance]] of both sides. On the right side, the total variation distance will be at most 1, meaning we can bound it as $\theta^k$:
$$
||P^{rk+j}_{ij} - \pi_j||_{TV} \leq \theta^k.
$$
Finally, consider an initial random variable $X_0 \sim \mu_0$. Pick $n = Nk+j$. For the variation distance between $X_n$ and $\pi$, we have 
$$
\sum_j ||\mathbb{P}(X_n = x_j) - \pi_j|| = \sum_j \left| \sum_i \mu_{0i} P^{Nk+j}_{ij} - \pi_j \right|.
$$
We already know that $P$ is close to $\pi$. We can extract $\mu_0$ 