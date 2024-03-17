---
tags:
  - ML
  - Course
---
ML for many-body physics

First lecture: motivation

Applications of ML:
ChatGPT
PDE solving
Prediction models
generating synthetic data
computer vision
medical imaging
art
games
**Many body systems and phases of matter**

What's the difference between ML and AI?
Deep learning subset neural networks subset ML subset AI. 

ML trains computers to detect and characterize features from data

Many-body physics wants to predict macroscopic phenomena (features) from microscopic quantities (data)

System of many particles interacting. Describes, eg, high-T superconductors, fractional quantum hall effect. Not necessary for effective single quasiparticle theories, eg, electron conduction in most metals and semiconductors
Exponentially difficult to solve
Computational methods:
Monte Carlo - we'll use this to generate our training data lol
Tensor networks
exact diagonalization
density functional theory
**Machine Learning**

In all cases, we start from a physical theory, use it to generate data describing the system.
In machine learning, we start from data and learn about macroscopic features of that data, often phases.

ML algorithm categories
supervised learning - dataset $\mathcal{D} = \{(\vec{x}, \vec{y})\}$
	$\vec{x}$ are datapoints, $\vec{y}$ are labels
	Task: fit some $f(\vec{x})$ to $\vec{y}$. 
	different function classes
	regression: labels are continuous
	classification: labels are discrete
unsupervised learning - dataset $\mathcal{D} = \{x\}$ unlabelled
	Task: extract meaningful features from the dataset to efficiently represent the data's underlying probability distribution $P_\text{data}(x)$.
	Might not be able to label with any knowledge of the distribution
Reinforcement learning (not covered)
	Task: given an "environment", take an action such that the resulting "reward" is maximized.

Course outline:
Supervised Learning (1-7)
Unsupervised Learning (8,9,12 - generative modelling)
Ethics (11)
Research in physics and ML

SL Examples:
Example 1: 1D regression
	Take a dataset for points in $\mathbb{R}^2$, determine a curve that fits the data
Example 2: Classifying handwritten digits
	Dataset: data are handwritten images, labels are corresponding numbers
	Goal: learn to determine the label corresponding to a new handwritten image
Example 3:
	Simple classical Ising model
	Hamiltonian $H= -J \sum_{<ij>} s_i s_j$, spin up or down (0 or 1)
	Undergoes temperature phase transition in $d\geq 2$
Example 3a: Phase classifier
	In 2D, phase transition at $T/J = T_c/J \approx 2.269$
	Below is ferromagnetic phase, above is paramagnetic phase
	Dataset: data is the spin configuration, label is FM or PM. 
	Homework 1 - study this with a neural network
Example 3b: Regression for the Ising Hamiltonian
	Dataset: data is the spin configuration, label is energy
	Pretend we are given the dataset and don't know what Hamiltonian generates these energies
	Assume a general model $H_\text{reg} = -\sum_i \sum_j J_{ij} s_i s_j$
	Use the model to determine the matrix $J_{ij}$.

# Linear Regression
arxiv: 1803.08823 section VI. machine learning methods in physics review
U Toronto CSC 411, lec 6. general ML course
linear regression, gradient descent, k-nearest neighbors

Goal: fit $f(x)$ to $y$ with $f(x) = \sum_j w_j x_j$, $w_j$ are fitting parameters

Notation: 
	$M$ is number of datapoints in $\mathcal{D}$
	$x_j^{(i)}$ is the jth elem of sample i
	matrix $x\in \mathbb{R}^{M*d_x}$, with $x_{ij} = x_j^{(i)}$
	similar with $y$

Values of the fitting parameters are determined by minimizing
$$
\mathcal{L} \equiv \sum_i \left( \sum_j w_j x_j^{(i)} - y^{(i)} \right)^2 = || Xw - y ||_2^2
$$
For linear regression, can get an exact solution for $w$. $w = \left( X^T X \right)^{-1}X^T y, f(x) = w^Tx$.
Note: exact solution assumes $X^TX$ is invertible. Infinitely many solutions for rank($X$) $< d_x$

Alternatively, use eg gradient descent to optimize $w$.
Why use GD when we have an exact solution?
Matrix multiplication and inversion can be inefficient in general.
Many later algorithms are built on GD.

Gradient Descent
Used when we want to minimize some constraint equation wrt parameters $w$.
$$
\frac{\partial y}{\partial w_i} = 0
$$

Approximate a local change in $\mathcal{L}$ according to its gradient at some starting point, iterate until minimum reached.
$$
\Delta \mathcal{L} \approx \nabla_w \mathcal{L} \cdot \Delta w
$$
We want $\Delta \mathcal{L} < 0$ and we are free to choose $\Delta w$ at each iteration step.
First attempt:
$$
\Delta w = -\eta \nabla_w \mathcal{L}
$$
$$
w_i \rightarrow w_i - \eta \frac{\partial \mathcal{L}}{\partial w_i}
$$
$\eta > 0$ is the "learning rate". We get to choose this manually when defining the problem (hyperparamter). 

If $\eta$ is too small, can take forever and get stuck in a local minimum.
$\eta$ must be small enough for the first-order approx to be valid.

Constant $\eta$ might be too restrictive.
Ideas for the next attempt: add "momentum" and a decay rate to $\eta$. 

# K-Nearest Neighbors
Now our $f(x)$ 

Step 1: calculate the distance $d(x,x^{(i)})$ for all $x^{(i)} \in \mathcal{D}$.
Step 2: Find the k nearest neighbors of $x$.
	Store them as $x^{*(j)}$, and corresponding labels as $y^{*(j)}$.
Step 3a: If classification: any new x sufficiently close maps to corresponding y. $f(x)$ is the majority member of $\{ y^{*(j)}\}$
Step 3b: If regression: average of that.

hyperparameter is $k$
How to choose $k$?
Taking too many can just make it basically the same dataset, too few isn't going to be very accurate

There are no fitting parameters

what if k could be a fitting parameter?

# Other
Decision trees (not covered)
Logistic regression (not covered)
Support vector machines (might cover?)
Neural Networks (next lecture)

Even though we use NN so much in research, it's easier to physically interpret the starting simple algorithms.

Neural networks can be very flexible, whereas these simple algs might fail for certain function classes.

