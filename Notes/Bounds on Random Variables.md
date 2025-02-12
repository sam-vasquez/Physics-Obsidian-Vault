---
tags:
  - Probability-Theory
---
Many inequalities exist that relate a random variable and an upper or lower bound on the probability of particular events, often in terms of its statistical moments.

#### Markov's Inequality and Extensions
The most basic and loosest inequality is [[Markov's Inequality]]. It only requires the existence of a random variable $X \geq 0$. Given a positive constant $k$, the inequality is
$$
\mathbb{P}(X \geq k) \leq \frac{\mathbb{E}(X)}{k}.
$$
This inequality requires the least amount of information of $X$, but by consequence is a very weak bound. 

We can strengthen Markov's inequality if we can assume a defined and finite expectation value and variance of $X$. This gives us [[Chebychev's Inequality]], which is
$$
\mathbb{P}(|X - \mathbb{E}[X]| \geq c\sigma_X) \leq \frac{1}{c^2}.
$$
This inequality essentially states the probability that the random variable $X$ is within $c$ standard deviations of its expected value. Using only one statistical moment however, it's still a rather weak inequality. For example, given a normal distribution, we know that the exact probabilities for $c=1,2,3$ should be 32%, 5%, and 0.3%; the inequality suggests upper bounds of 100%, 25%, and 11%.

Chebychev's inequality can be extended to use higher moments when they exist. If we can assume that the Moment Generating Function $M_X(t) = \mathbb{E}[e^{ tX }]$ of $X$ exists, we get the [[Chernoff Bounds]],
$$
\mathbb{P}(X \geq k) \leq \inf_{0\leq t\leq b} [e^{ -tk }M_X(t)].
$$
For example, the MGF of a Gaussian variable $X \sim N(\mu,\sigma^2)$ is
$$
M_X(t) = e^{ \mu t + \frac{t^2 \sigma^2}{2} }.
$$

[[Hoeffding's Inequality]] is an application of Chernoff bounds to sums of sub-Gaussian variables. Notably, every bounded variable is sub-Gaussian, suggesting that a valuable use of Hoeffding's inequality is for the probability of polling results. 