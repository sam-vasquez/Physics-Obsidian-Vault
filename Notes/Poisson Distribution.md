---
tags:
  - Probability-Theory
---
A Poisson distribution represents the probability distribution of a given number of events occurring in a fixed interval of time.

When the expectation is $\lambda$ events in a given reference interval $T$ (or when the events occur at an average rate of $r$, such that $\lambda = rT$), the probability of $k$ events in any such interval is 
$$
P(X=k) = \frac{\lambda^k e^{ -\lambda }}{k!}.
$$
It has expected value $\lambda$ and variance $\lambda$.

The following assumptions must hold:
- Events are independent.
- Expectation value is constant.
- Only one event happens at a time.

Poisson distributions are invariant under [[SL(N)]] transformations: Any volume-preserving map of the distribution will still be Poisson-distributed with the same expected value. 

In the limit of a very large set, the number of $k$-cycles in a random permutation approaches a Poisson distribution of mean $1/k$. [Source](https://math.ucr.edu/home/baez/permutations/permutations_7.html).