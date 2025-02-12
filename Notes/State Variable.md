---
tags:
  - Stat-Mech
---
In thermodynamics, a state variable is any macroscopic property of a thermodynamic system that can be measured at a specific time without needing to care about how the system became the way it is.

For example, energy and position are intrinsic properties of the system's state at a specific time. They are properties that can be measured immediately. They are state variables. However, work is not a state variable. It's a function of energy over time, it's calculated via a path-dependent work integral, it's necessarily sensitive to previous states of the system and cannot be directly measured.

State variables can be written in terms of each other, and functions can be defined in terms of state variables. However, you can't write a state variable in terms of non-state variables. There is no expression $U = U(W)$ for example.

There is a fewest number of functions needed to fully determine all other experimentally measurable properties of the system. For a manifold of equilibrium states at a given energy, there are at least two degrees of freedom, [[Entropy]] and [[Temperature]], related to heat transfer, plus other pairs of variables related to various kinds of work. A common additional pair is pressure and volume to represent mechanical work. 

State variables can be either [[Extensive and Intensive State Variables|extensive or intensive]]. The conjugate variables defining the degrees of freedom of the equilibrium state manifold always comes in pairs of extensive and intensive variables, such that work can always be written as $\delta W = \sum_i f_i \;dX_i$ for intensive $f$ and extensive $X$. For proof, look up Euler's theorem for homogeneous functions.