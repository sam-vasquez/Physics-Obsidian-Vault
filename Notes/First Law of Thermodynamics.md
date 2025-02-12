---
tags:
  - Stat-Mech
---
The first law of thermodynamics is simply a statement of energy conservation.

In classical thermodynamics, a distinction is made between energy that is added to systems via work, and energy that is added via any other process, called "heat transfer".

This is stated simply as 
$$
dU = \delta Q + \delta W,
$$
where $\delta$ denotes an inexact differential.

The work differential can be decomposed as a sum of products of canonical [[State Variable|state variables]],
$$
\delta W = \sum_i f_i\; dX_i,
$$
with $f$ a set of [[Extensive and Intensive State Variables|intensive]] variables and $X$ a set of extensive variables.

It can be proven with the [[Second Law of Thermodynamics]] that the heat differential can be written in terms of [[Entropy]] and [[Temperature]] as
$$
\delta Q = T dS,
$$
see the proof [[Heat-Entropy Relation|here]].

Therefore, the first law can be rewritten as the [[Thermodynamic Identity]],
$$
dU = TdS + f_i \; dX_i.
$$

In statistical thermodynamics, the first law is instead formulated in terms of fixed energy. The dynamical system evolves along its **energy surface**, the set of configurations in phase space $\{q_i, p_i\}$ with fixed energy $\mathcal{H}(q_i,p_i) = E$.  