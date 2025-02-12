---
tags:
  - ML
---
The bias-variance trade-off is a statement made by statistical learning theory that describes the relationship between a model's complexity, the accuracy of its predictions, and how well it can make predictions on previously unseen data that were not used to train the model.

Let $\mathcal{D} = \{(\bar{x}^a, \bar{y}^a)\}$ be the model's training dataset. This dataset is sampled from a true data distribution. Assume that this true data takes the form $y = f(x) + \epsilon$, where $\epsilon$ is normally distributed with mean zero and standard deviation $\sigma_\epsilon$. 

Assume that we have a statistical procedure (e.g. [[Linear Regression]]) for forming a predictor $\hat{f}(x;w)$ that gives the prediction of our trained model for a new data point $x$. This predictor is chosen such that a cost function $\mathcal{L}$ is minimized among all values of parameters $\mathbf{w}$ for the given training dataset $\mathcal{D}$. Assume that this cost function is the squared error on $\mathcal{D}$:
$$
\mathcal{L} \equiv \sum_a(y^a - \hat{f}(x^a ; w))^2 = || \mathbf{y} - \hat{f}( \mathbf{X} ; \mathbf{w} )||_2^2.
$$
Let $\mathbf{\hat{w}}$ denote the value of $\mathbf{w}$ which minimizes $\mathcal{L}$. This estimated value will depend on our choice of training dataset $\mathcal{D}$, so we can consider it to be a function of the dataset, $\mathbf{\hat{w}} \coloneqq \mathbf{\hat{w}}_\mathcal{D}$. 

We are interested in finding an expression for the expected error of this predictor against any sample of data $\{(x^a,y^a)\}$ from the true distribution. We can find this by considering an expectation value of the minimized loss function across the space of all possible datasets $\mathcal{D}$ and all possible noise $\epsilon$. Denote this expectation value $\mathbb{E}_\mathcal{D,\epsilon}\left[ \mathcal{L} \right]$. 

This expectation value decomposes as
$$
Bias^2 + Var + Noise,
$$
where the first term
$$
Bias^2 = \sum_a \left( f(x^a) - \mathbb{E}_\mathcal{D} [\hat{f}(x^a;\hat{w}_\mathcal{D})] \right)^2
$$
represents the deviation from the true value of the asymptotic value of our predictor function in the infinite training data limit, the second term
$$
Var = \sum_a \mathbb{E}_\mathcal{D} \left[ \left( \hat{f}(x^a ; \hat{w}_\mathcal{D}) - \mathbb{E}_\mathcal{D}[\hat{f}(x^a ; \hat{w}_\mathcal{D})] \right)^2 \right] 
$$
represents how much the predictor fluctuates due to finite-sample effects, and the third term
$$
Noise = \sum_a \sigma_\epsilon^2
$$
represents unavoidable noise of the data itself. 

See [[Bias-Variance Decomposition]] for details.
