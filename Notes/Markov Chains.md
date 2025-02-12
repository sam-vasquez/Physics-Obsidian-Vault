---
tags:
  - Probability-Theory
---
A Markov chain is a stochastic process that describes a sequence of events, in which the probability of each event depends only on the state of the previous event.  

A sequence of random variables $X_0, X_1,\cdots$ is a Markov chain with state space $\mathcal{X}$ and transition matrix $P$ if for all $n \in \mathbb{N}$ and $x_i, x_j \in \mathcal{X}$ we have 
$$
\mathbf{P}(X_{n+1} = x_j | \{ X_k = x_k | 0 \leq k \leq n \}) = \mathbf{P}(X_{n+1} = x_j | X_n = x_i) \equiv P_{ij}.
$$

A basic example of a Markov chain is a random walk. At each time step $n$, the random walk chooses a random direction and updates $X_n$ according to this choice, independent of all previous choices and states. 

The transition matrix is an important object in Markov chain analysis. Each row $i$ of $P$ is the conditional distribution of $X_{n+1}$ given that $X_n = x_i$. We can write this distribution as a row vector, and then the distribution of $X_{n+1}$ can be obtained by matrix multiplication by $P$. Taken recursively to $n=0$: If $X_0 \sim \pi_0$, then $X_n \sim \pi_0 P^n$. 

An important application of Markov chains is to study how arbitrary distributions evolve along the chain, and whether they can converge to a limit. We call such limits a stationary distribution. [[Markov Chain Monte Carlo]] is concerned with the question of whether we can construct a Markov chain process for generating a desired probability distribution as the limit of this process, starting from an initial distribution that's easier to generate explicitly (e.g. a uniform distribution), for the sake of performing [[Monte Carlo Integration]]. See [[Stationary Distributions]] for details on when convergence is possible, and [[Metropolis-Hastings Algorithm]] for details on the most simple Markov chain that accomplishes this goal.

Most interesting Markov chains possess the properties of irreducibility and aperiodicity, which also turn out to be necessary conditions for important results regarding convergence.

A Markov chain is called *irreducible* if for any $x_i, x_j \in \mathcal{X}$, there exists $n \in \mathbb{N}$ such that $P^n_{ij} > 0$.

^2ce78f

Let $\mathcal{T}(x_i) := \{ n \geq 1 : P^n_{ii} > 0 \}$ be the set of times where it is possible for a given Markov chain to return to its starting state $x_i$. The *period* of state $x_i$ is the greatest common divisor of $\mathcal{T}(x_i)$.
The Markov chain is said to be *aperiodic* if for all $x_i \in \mathcal{X}$, $\gcd \mathcal{T} = 1$.
 
Irreducibility is essentially a statement that every state in $\mathcal{X}$ is reachable by the chain. It is generally easy to assume that the chains under discussion are irreducible - sometimes it's interesting to confirm this for particular chains. For example, the construction of a Markov chain on the space of [[Self-Avoiding Walks]].

Periodicity defines the set $\mathcal{T}(x_i)$ of all lengths of cycles that start and end at $x_i$. When this set has gcd $> 1,$ there is some cyclical behavior in the chain that would prevent an initial distribution from settling on stable values. The random walk on an even $n$-cycle is periodic because every step alternates between an even state and an odd state. The stationary distribution is $\pi = \{1/n\}$, but an initial distribution defined only on even states would never achieve the stationary distribution on a single time step. Periodicity is not really a big deal because we can always break it by introducing laziness, to always say that there is some small probability that we remain where we are. 

Together, these properties ensure that the chain will always eventually reach a point where every state constantly mixes with each other for the rest of time. See a formal statement and proof [[Aperiodic and Irreducible Chains Mix All States in Finite Time|here]].

References:
https://www.math.cmu.edu/~gautam/c/2024-387/index.html