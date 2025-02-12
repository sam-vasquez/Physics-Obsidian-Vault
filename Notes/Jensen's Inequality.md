---
tags:
  - Probability-Theory
---
Let $f$ be a strictly convex function defined over a convex set $\mathcal{X}$, that is, a function such that for every $x \neq y \in \mathcal{X}$, and all $0 < \theta < 1$,
$$
f(\theta x + (1-\theta)y) < \theta f(x) + (1-\theta) f(y).
$$
Given a discrete random variable $X$ that takes values in $\mathcal{X}$, such that there is only a finite number of points where $X = x$ with nonzero probability, then
$$
\mathbb{E}[f(X)] > f(\mathbb{E}[X]).
$$

**Proof:**
We will prove this by induction on the state space of $X$.

For the base case, assume that $X$ takes on two values, $x$ and $y$.
The expectation value of $f(X)$ is written as 
$$
\mathbb{E}[x] = \mathbb{P}[X = x] x + \mathbb{P}[X = y] y,
$$
but for compactness we'll write $\mathbb{P}[X=x]$ as $\mathcal{P}(x)$.
Note that the sum of probabilities is 1, and each probability is greater than 0, making this expression a convex combination of $x$ and $y$ with coefficients $\mathcal{P}(x)$ and $\mathcal{P}(y)$. Therefore, the function can be evaluated at this point and by convexity the following holds:
$$
f(\theta x + (1-\theta)y) < \theta f(x) + (1-\theta)f(y),
$$
where $\theta = \mathcal{P}(x)$. The expression on the right is just the expectation value of $f(X)$, and therefore the claimed inequality $f(\mathbb{E}[X]) < \mathbb{E}[f(X)]$ holds for the base case.

For the induction case, assume that the inequality holds for a state space where $X$ takes on $n$ values $x_i$. In this space, each value $x_i$ has a probability $\mathcal{P}(x_i)$ and the inequality states that
$$
f(\sum_i^n x_i \mathcal{P}(x_i)) < \sum_i^n \mathcal{P}(x_i)f(x_i).
$$
Now, add another point $x_{n+1}$, with probability $\hat{\mathcal{P}}(x_{n+1})$.
To accomodate this additional state, we will scale all other probabilities by a factor of $1 - \hat{\mathcal{P}}(x_{n+1})$. This will give us 
$$
\sum_i^{n+1} \hat{\mathcal{P}}(x_i) = 1,
$$
where $\hat{\mathcal{P}}(x_i) = (1-\hat{\mathcal{P}}(x_{n+1})) \mathcal{P}(x_i)$ for $1 \leq i \leq n$.
The expectation value is 
$$
\mathbb{E}[X] = \sum_i^{n+1} x_i \hat{\mathcal{P}}(x_i) = (1 - \hat{\mathcal{P}}(x_{n+1})) \sum_i^n x_i \mathcal{P}(x_i) +  \hat{\mathcal{P}}(x_{n+1}) x_{n+1}.
$$
Note that $\sum_i^n x_i \mathcal{P}(x_i)$ is a convex combination of points in $\mathcal{X}$ and therefore in $\mathcal{X}$. And the expectation value is itself another convex combination of that point and $x_{n+1}$. This lets us evaluate the function $f$ at $\mathbb{E}[X]$ and apply convexity to get the inequality
$$
f(\mathbb{E}[X]) < (1 - \hat{\mathcal{P}}(x_{n+1})) f( \sum_i^n x_i \mathcal{P}(x_i) ) + \hat{\mathcal{P}}(x_{n+1}) f(x_{n+1}).
$$
Now apply the induction hypothesis:
$$
\begin{align}
f(\mathbb{E}[X]) &< (1 - \hat{\mathcal{P}}(x_{n+1})) \sum_i^n \mathcal{P}(x_i)f(x_i) + \hat{\mathcal{P}}(x_{n+1}) f(x_{n+1})  \\
&= \sum_i^n \hat{\mathcal{P}}(x_i)f(x_i) + \hat{\mathcal{P}}(x_{n+1}) f(x_{n+1})  \\
&= \sum_i^{n+1} \hat{\mathcal{P}}(x_i)f(x_i) = \mathbb{E}[f(X)].
\end{align}
$$
$\blacksquare$

Note: can prove positivity of variance as a special case of Jensen's inequality, providing a convenient way to remember the direction of the inequality.
