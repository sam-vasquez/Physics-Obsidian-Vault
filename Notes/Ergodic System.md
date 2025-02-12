---
tags:
  - Stat-Mech
---
For some classical observable $A$, and some initial point in phase space $(r_0^N,p_0^N)$, denote its phase space trajectory
$$
A(t) \equiv A(t|r^N,p^N),
$$
$$
A(0) \equiv A(r_0^N, p_0^N).
$$
At a given time $t$, the [[Ensemble Average]] of $A(t)$ is
$$
\langle A(t) \rangle = \int dr^N \int dp^N f(r^N,p^N) A(t|r^N,p^N).
$$
Conversely, the time-averaged value of a single trajectory of $A(t)$ is
$$
\langle A \rangle = \frac{1}{T} \int_0^T dt A(t).
$$
An ergodic system is one for which these two averages coincide for sufficiently large $T$.
It can also be shown that the ensemble average of $\delta A(t) = A(t) - \langle A \rangle$ is zero, $\langle \delta A(t) \rangle = 0$.

This condition is identified with a system where all accessible states are equiprobable over large periods of time. 

It is typically taken as a postulate of equilibrium statistical mechanics that every thermal system is ergodic. See [[Ergodic Hypothesis]].
Note: There is an analogous result in [[Markov Chains]] theory.

In particular, [[Liouville's Theorem]] implies that a uniform distribution in phase space will remain uniform at all times. 
Note: There is an analogous result in [[Markov Chains]] theory, in particular, that [[Symmetric Markov Chain has Uniform Stationary Distribution]].