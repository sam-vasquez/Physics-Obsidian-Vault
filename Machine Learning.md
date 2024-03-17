---
tags:
  - ML
---
Machine learning is the discipline that studies how to solve computationally
difficult problems for impossibly-large sets of data by training the computer to
detect simplifying patterns in the data.

Machine learning can be quite useful for physics. It mainly finds its
application in the study of [[Many-Body Systems|many-body systems]], where it
can be quite difficult to extract macroscopic quantities from the underlying
particle description, even computationally.

Machine learning can be loosely divided into three sub-disciplines: 
- [[Supervised Learning]]
- Unsupervised Learning 
- Reinforcement Learning

Supervised learning starts with a dataset that is labeled by the user, and the
model is trained to fit a function of some general form to the data such that it
outputs the labels. Upon training, the model is in theory capable of labeling
new unlabeled data according to its learned fitting function, extrapolating the
training set to a larger set of real data. When the labels are continuous, this
is usually called Regression. When the labels are discrete, this is usually
called Classification.

Unsupervised learning starts with a dataset that is unlabeled. The model is
trained to detect patterns in the data that aims to represent the data's
underlying probability distribution.

Reinforcement learning starts with an "environment", and tries to take different
actions to find an action that maximizes its reward function. This attempts to
model actual human learning, but it can be quite difficult to prepare an
appropriate environment and define a good reward function.