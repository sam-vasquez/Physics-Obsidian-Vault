---
tags:
  - ML
---
Linear regression is ubiquitous class of [[Supervised Learning]] technique, and is one of the most elementary cases of regression problems. Linear regression attempts to parameterize a dataset by assuming that the data can be closely approximated by a linear functional form, and attempting to find optimal values for the parameters that define this linear relation. Once this function is obtained for the sample data, it is hoped that it can extrapolate to out-of-sample effectively.

Let $\mathcal{D}^a = (x^{aj}, y^a)$ be the M-dimensional dataset of vectors $\vec{x} \in \mathbb{R}^N$.

The goal is to find parameters $w^j$ that best fit $f(x^a)$ to $y^a$ with $f(x) = x^a{}_j w^j = \mathbf{Xw}$. 

The fitting parameters can be determined by minimizing
$$
\mathcal{L} \equiv (x^a{}_j w^j - y^a) (x_a{}_k w^k - y_a) = || \mathbf{Xw - y} ||_2^2.
$$

In some cases, an exact solution to this optimization problem does exist. See [[Exact Solution to Linear Regression]] for the derivation. 

However, just because we can calculate an exact solution, doesn't mean we should. This solution's unique existence depends on the invertibility of the matrix $\mathbf{X^T X}$. It very well might not be, but even if it is, matrix inversion is very expensive. Even matrix multiplication can generally be expensive.

Alternatively, we can use an algorithm to optimize $\mathbf{w}$.

Common algorithms for linear regression include:
- [[Gradient Descent]] 

# Generalization Error
In-sample error
Out-of-sample error

See section 3.



# Regularization

[[Ridge Regression]]