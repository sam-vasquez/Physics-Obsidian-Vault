---
tags:
  - ML
---
# Simple gradient descent
Gradient descent is one of the simplest iterative algorithms for minimizing a constraint function with respect to some parameters, such as the loss function in [[Linear Regression]].

The method is simple: iteratively adjust the parameters in the direction where the gradient of the cost function is largest and negative. With some careful choice of step size, this ensures that the algorithm approaches a local minimum of the cost function. 

Let $\mathcal{L}\left[ w^i \right]$ be the function we are attempting to optimize. We can descend along its gradient starting from some initial point with the following approximation:
$$
\Delta \mathcal{L} \approx \partial_{w_i} \mathcal{L} \cdot \Delta w_i.
$$
$\Delta \mathcal{L}$ should be negative to call this "descent". We are free to choose $\Delta w_i$ at each iteration step.

The simplest GD algorithm would choose $\Delta w_i$ at each iteration according to the following equations:
$$
\Delta w_i = \eta \partial_{w_i} \mathcal{L}[w_i]
$$
$$
w_i \rightarrow w_i - \Delta w_i
$$
for some "learning rate" hyper-parameter $\eta > 0$.

# Learning rate and convergence
For small $\eta$, the algorithm can be painfully slow and computationally expensive, and often gets stuck in local minima, but at the same time $\eta$ must be small enough for this first-order approximation to remain valid and return legitimate results.

GD algorithms often include dynamics for $\eta$, such as a decay rate such as power law or exponential decay that applies at long times, or a "momentum". 

It is possible to predict an "optimal" $\eta$ for some functions by thinking about the second derivative behavior of the cost function. One can show that if we were to use the same $\eta$ for all parameters, convergence requires that 
$$
\eta < \frac{2}{\lambda_{\textrm{max}}}
$$
where $\lambda_{\textrm{max}}$ is the largest singular value of the Hessian. Meanwhile, convergence time scales with 
$$
\kappa = \frac{\lambda_{\textrm{max}}}{\lambda_{\textrm{min}}}.
$$
TODO: See LeCun, Yann, Léon Bottou, Genevieve B Orr, and KlausRobert Müller (1998b), “Eﬃcient backprop,” in Neural networks: Tricks of the trade (Springer) pp. 9–50.

While this is very helpful for analyzing the convergence of our GD algorithm, unfortunately the Hessian is very expensive to calculate, and it's not particularly practical to keep track of second derivatives to inform our learning rate dynamically. 

## Stronger assumptions
It is often possible to prove many stronger statements about convergence rates if we can assume stronger conditions on the function.

For example, if the function is assumed to be strongly convex and Lipschitz smooth, then gradient descent converges linearly with a fixed step size.

See convex optimization for more details.

# Extensions
There are many shortcomings of this version of gradient descent, but there are just as many algorithms developed to address these shortcomings.

Of course we encounter a difficulty that the local minimum might not be a global minimum. When the cost function is a convex function, the local minimum is guaranteed to be a global minimum.

When the result of gradient descent is not guaranteed to be a local minimum, one common method to overcome local minima is to introduce some kind of stochasticity, inspired by thermal fluctuations that allow the algorithm to thermally tunnel through energy barriers to reach a more global minima. 

The gradient can be rather expensive to compute, simply because of the large datasets we often encounter and the fact that we have to iterate through every data point at each step of the GD algorithm. A good solution to this is to only calculate the gradients using smaller subsets of data called "mini batches". This also happens to introduce stochasticity into the algorithm. These two facts together are the basis of [[Stochastic Gradient Descent]].

It is helpful to include an inertia term that serves as a memory of the direction the gradient descent algorithm is moving in parameter space. This helps the algorithm avoid getting stuck following stochastic noise. See [[Gradient Descent with Momentum]] for details.