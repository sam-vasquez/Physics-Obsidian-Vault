---
tags:
  - Probability-Theory
---
Given a non-negative random variable $X \geq 0$ with a Moment Generating Function $M_X(t) = \mathbb{E}[e^{ tX }]$ that is defined and finite for all $|t|\leq b$, for any positive constant $k > 0$, the following inequality holds:
$$
\mathbb{P}(X \geq k) \leq \inf_{0\leq t\leq b} [e^{ -tk }M_X(t)].
$$

**Proof:**
Start from [[Markov's Inequality]]:
$$
\mathbb{P}(X\geq k) \leq \frac{\mathbb{E}[X]}{k}.
$$
Replace $X$ with $e^{ tX }$ and $k$ with $e^{ tk }$:
$$
\mathbb{P}(e^{ tX } \geq e^{ tk }) = \mathbb{P}(X \geq k) \leq \frac{\mathbb{E}[e^{ tX }]}{e^{ tk }} = e^{ -tk }M_X(t).
$$
Since this is valid for every $t$ where $M_X(t)$ is defined, we can take the infinum over $t$ to get a strictest lower bound.
$$
\mathbb{P}(X \geq k) \leq \inf_{0\leq t\leq b} [e^{ -tk }M_X(t)].
$$
$\blacksquare$
