---
tags:
  - ML
  - theorem
---
> [!thm] Bias-Variance Decomposition
> The mean squared error of a predictor $\hat{f}(x;\hat{w}_\mathcal{D})$ trained on a dataset $\mathcal{D}$ can be decomposed into a bias term, a variance term, and a noise term.
> The bias term is 
> $$ 
> Bias^2 = \sum_a \left( f(x^a) - \mathbb{E}_\mathcal{D} [\hat{f}(x^a;\hat{w}_\mathcal{D})] \right)^2, 
> $$
> the variance term is
> $$ 
> Var = \sum_a \mathbb{E}_\mathcal{D} \left[ \left( \hat{f}(x^a ; \hat{w}_\mathcal{D}) - \mathbb{E}_\mathcal{D}[\hat{f}(x^a ; \hat{w}_\mathcal{D})] \right)^2 \right], 
> $$
> and the noise term is
> $$ 
> Noise = \sum_a \sigma_\epsilon^2. 
> $$

#### Proof:
We can decompose this expectation value as follows:
$$
\begin{align}
\mathbb{E}_{\mathcal{D},\epsilon} \left[ \mathcal{L} \right]  & = \mathbb{E}_{\mathcal{D},\epsilon} \left[ \sum_a (y^a - \hat{f}(x^a ; \hat{w}_\mathcal{D}))^2 \right] \\
 & = \mathbb{E}_{\mathcal{D},\epsilon} \left[ \sum_a (y^a - f(x^a) + f(x^a) -  \hat{f}(x^a ; \hat{w}_\mathcal{D}))^2 \right] \\
 & = \sum_a \left( \underbrace{ \mathbb{E}_\epsilon \left[ (y^a - f(x^a))^2 \right] }_{= \sigma_\epsilon^2 } + \mathbb{E}_{\mathcal{D},\epsilon} \left[ (f(x^a) - \hat{f}(x^a ; \hat{w}_\mathcal{D}))^2 \right] + 2\cancelto{ 0 }{ \mathbb{E}_\epsilon\left[ y^a - f(x^a) \right] }\mathbb{E}_\mathcal{D} \left[ f(x^a) - \hat{f}(x^a ; \hat{w}_\mathcal{D}) \right] \right) \\
 & = \sum_a \left( \sigma_\epsilon^2 + \mathbb{E}_\mathcal{D} \left[ (f(x^a) - \hat{f}(x^a ; \hat{w}_\mathcal{D}))^2 \right] \right).
\end{align}
$$

We recognize the first term as $Noise = \sum_a \sigma_\epsilon^2$, an unavoidable error term coming from the noisiness of the data itself. The second term can be decomposed further. For convenience of notation, let $f \equiv f(x^a)$ and $\hat{f} \equiv \hat{f}(x^a ; \hat{w}_\mathcal{D})$, and $\mathbb{E} \equiv \mathbb{E}_\mathcal{D}$.
$$
\begin{align}
\mathbb{E}\left[ (f - \hat{f})^2 \right]  & = \mathbb{E} \left[  \left( f - \mathbb{E} [ \hat{f} ] + \mathbb{E} [ \hat{f}] - \hat{f} \right)^2 \right] \\
 & = \mathbb{E}\left[ \left( f - \mathbb{E}[\hat{f}] \right)^2 \right] + \mathbb{E}\left[ \left( \hat{f} - \mathbb{E}[\hat{f}] \right)^2 \right] + 2 \mathbb{E} \left[ \left(f - \mathbb{E}[\hat{f}] \right)\left(\hat{f} - \mathbb{E}[\hat{f}] \right) \right] .
\end{align}
$$
The first term is independent of the dataset and is simply 
$$
\left( f - \mathbb{E}[\hat{f}] \right)^2,
$$
which is identified as the bias term.
The second term is identified as the variance of $\hat{f}$, the variance term.
The third term cancels to zero:
$$
\begin{align}
\mathbb{E} \left[ \left(f - \mathbb{E}[\hat{f}] \right)\left(\hat{f} - \mathbb{E}[\hat{f}] \right) \right]  & = \mathbb{E}\left[f \hat{f} - \hat{f} \mathbb{E}[ \hat{f}] - f \mathbb{E}[\hat{f}] + \mathbb{E}[\hat{f}]^2 \right]  \\
 & = f \mathbb{E}[\hat{f}] - \mathbb{E}[\hat{f}]^2 - f \mathbb{E}[\hat{f}] + \mathbb{E}[\hat{f}]^2 \\
 & = 0.
\end{align}
$$
This completes the decomposition.