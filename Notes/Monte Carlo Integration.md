---
tags:
  - Probability-Theory
---
Monte Carlo is a group of techniques that use repeated random sampling to numerically approximate the value of integrals that are very difficult to be solved through other means. The idea is to sample random points of the integration region according to some probability distribution and then use these values to estimate the value of the desired calculation. The Law of Large Numbers guarantees that the random sampling indeed makes it so that the approximation converges almost surely to the true value. 

Monte Carlo methods are distinct from ordinary numerical integration methods largely because of how scalable these methods are. Random sampling provides useful information about the problem at a fraction of the cost of going through every single point of data. The curse of spatial dimensionality makes classical algorithms prohibitively expensive, but Monte Carlo's performance is largely independent of dimension. And Monte Carlo methods are extremely easy to make parallel across multiple cores/GPUs/systems.

Let $f$ be a function over a multidimensional region $\Omega \subseteq \mathbb{R}^d$. Say we want to compute the integral of $f$ over $\Omega$. We can think of this integral as an average value of the function against a uniform probability distribution over $\Omega$.
$$
\langle f(x) \rangle = \frac{1}{\textrm{Vol}(\Omega)}\int_\Omega f(x).
$$
Thinking of the integral this way reveals to us that we can take advantage of the [[Law of Large Numbers]] to approximate this integral as a sum of values of $f$ sampled over independent and uniformly distributed variables $x_n$:
$$
\lim_{N\rightarrow\infty} \frac{1}{N} \sum_{n=1}^N f(x_n) = \frac{1}{\textrm{Vol}(\Omega)}\int_\Omega f(x).
$$
This gives us our simplest form of Monte Carlo integration.

Given N points $x_1,\cdots,x_N \in \Omega$ sampled from a uniform distribution, the naive Monte Carlo Estimator of the integral of a function $f$ defined on $\Omega$ is given by
$$
\int_\Omega f(x) \approx \frac{\textrm{Vol}(\Omega)}{N}\sum_{n=1}^N f(x_n)
.$$

As one can imagine, it can be very inefficient to sample points uniformly over the integration region. The Law of Large Numbers guarantees that it will eventually converge, but this could take a very long time when the sampling method treats all points equally. Ideally we would rather sample points according to a distribution that more closely resembles $f$, such that peaks of $f$ are more likely to be visited sooner than troughs. 

Luckily, the Law of Large Numbers applies equally well to an average of $f$ taken against an arbitrary probability distribution $p(x)$. Let $x_n$ be a sequence of independent and identically distributed variables sampled from $p(x)$. Then
$$
\lim_{N\rightarrow\infty} \frac{1}{N} \sum_{n=1}^N f(x_n) = \int_\Omega f(x)p(x).
$$
This leads to a general form of Monte Carlo integration:

Given N points $x_1,\cdots,x_N \in \Omega$ sampled from an arbitrary probability density function $p$, the Monte Carlo Estimator of the integral of a function $f$ defined on $\Omega$ is given by
$$
\int_\Omega f(x) \approx \frac{1}{N} \sum_{n=1}^N \frac{f(x_n)}{p(x_n)}
.$$

Todo: Prove that error scales as $1/\epsilon^2$.
Todo: Prove [[Law of Large Numbers]] using Levy's continuity theorem.

Although importance sampling is very useful for the variance and convergence of Monte Carlo integration, it suffers an important problem: computers are not good at sampling from non-uniform distributions. In some cases we might be able to transform our variables to ones where the distribution is uniform, but in general we won't be able to. 

[[Markov Chain Monte Carlo]] provides a general method for sampling from a non-uniform probability distribution by achieving it as the stationary distribution of some [[Markov Chains|Markov process]]. Various processes exist, the most commonly-used process is the [[Metropolis-Hastings Algorithm]].
