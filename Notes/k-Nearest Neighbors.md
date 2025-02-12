---
tags:
  - ML
---
k-Nearest Neighbors is one of the simplest [[Supervised Learning]] algorithms for classification or regression. This technique attempts to characterize features a dataset by assuming that sufficiently close data points (for some definition of close) can be clustered together and given a common or interpolated label. Out-of-sample data can then be compared with the sample data and assigned its corresponding label.

Let $\mathcal{D}^a = (x^{a}, y^a)$ be the M-dimensional in-sample dataset of objects $x$ in an unspecified metric space, and $x$ be an arbitrary member of the out-of-sample dataset.

The algorithm begins by calculating the distances $d(x,x^a)$ for all $x^a \in \mathcal{D}$. It proceeds to find the $k$ nearest neighbors of $x$. These are stored as indices $b \in I \subseteq [1,M]$, and the neighbors will be referred to as $x^*{}^b$ with corresponding labels $y^*{}^b$.

If this is a classification problem, in most cases $f(x)$ is chosen to be the majority member of $y^*{}^b$. If this is a regression problem, $f(x)$ is chosen to be the average value of $y^*{}^b$.

The hyper-parameter of this algorithm is obviously $k$. A good choice of $k$ usually only comes from heuristic evaluation of compared choices.
