---
tags:
  - Probability-Theory
  - todo
---
Probability that a large random matrix has two identical eigenvalues goes to zero. This is because the set of matrices with degenerate eigenvalues has measure zero in the set of all continuous-valued matrices.

"Let $M$ be the space of all $N\times N$ matrices over $\mathbb{R}$. For any $A$ in $M$, its discriminant $\Delta(A)$ is given by $\prod(\lambda_i - \lambda_j)^2$.
$A$ has a degenerate spectrum if and only if $\Delta(A)$ is vanishing. $\Delta(A)$ is a polynomial function of the eigenvalues, which themselves are polynomial functions of the entries of $A$ appearing in the characteristic polynomial $\det(A - I\lambda)$.
Therefore $\Delta$ is a polynomial function of the entries of $A$, hence the zero set of $\Delta(A)$, which is comprised of elements of $M$, is the same as the zero set of a single nontrivial polynomial function.
Suppose $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is an arbitrary nontrivial polynomial function. Then its zero set defines an algebraic hypersurface $S$. Generically, $\nabla f(x)$ is non-vanishing on $S$, and by the implicit function theorem $S$ is codimension-1, it can be regarded as a $(d-1)$-dimensional function embedded in $d$ dimensions, e.g. a smooth $(d-1)$-dimensional manifold.
But it is known that the Lebesque measure for a $d$-dimensional space automatically assigns a value of zero to any $(d-1)$-dimensional subset, and hence $S$ has measure zero.
Therefore, for an arbitrary nontrivial polynomial function, its zero set has measure zero. $\Delta(A)$ is included in this set of functions, and hence it has measure zero.
Therefore, the set of degenerate-eigenvalue matrices has measure zero in $M$, and the probability of selecting a matrix with degenerate spectrum from an arbitrary continuous pdf over $M$ is precisely zero.
This is equally applicable to the ginibre ensemble or more restrictively the GOE, so that in your example case of random matrices drawn using i.i.d. Gaussian entries is exactly zero."
https://www.reddit.com/r/Physics/comments/1igtq8t/i_dont_understand_spectral_distribution_in_random/masfhlm/