---
tags:
  - Probability-Theory
---
Given a non-negative random variable $X \geq 0$ with defined and finite expected value $\mathbb{E}[X]$ and variance $\mathbb{V}[X] = \sigma_X^2$, for any positive constant $c > 0$, the following inequality holds: 
$$
\mathbb{P}(|X - \mathbb{E}[X]| \geq c\sigma_X) \leq \frac{1}{c^2}.
$$

**Proof:**
Start with [[Markov's Inequality]]:
$$
\mathbb{P}(X \geq k) \leq \frac{\mathbb{E}[X]}{k}.
$$

If we replace $X$ with $(X - \mathbb{E}[X])^2$, the inequality becomes
$$
\mathbb{P}( (X - \mathbb{E}[X])^2 \geq k) \leq \frac{\mathbb{V}[X]}{k}.
$$

If we replace $k$ with $c^2 \; \mathbb{V}[X]$, the inequality becomes 
$$
\mathbb{P}( (X - \mathbb{E}[X])^2 \geq c^2 \; \mathbb{V}[X]) \leq \frac{1}{c^2}.
$$

Assuming $k$ is positive, the inequality is equivalent to

$$
\mathbb{P}(|X-\mathbb{E}[X]| \geq c\sigma_X) \leq \frac{1}{c^2}.
$$
$\blacksquare$

Note that if $X$ had finite higher moments, this derivation could have been extended to those higher moments. If the degree $r$ moment is defined, the inequality becomes
$$
\mathbb{P}(|X - \mathbb{E}[X]| \geq c) \leq \frac{\mathbb{E}[(X-\mathbb{E}[X])^r]}{c^r}.
$$
This motivates the [[Chernoff Bounds]] when the moment generating function exists. 