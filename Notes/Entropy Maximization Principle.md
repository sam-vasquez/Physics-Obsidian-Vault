---
tags:
  - Stat-Mech
  - todo
---
With energy and all other extensive variables equal, the equilibrium state of a thermodynamic system is the state with all internal configurations such that [[Entropy]] is maximized. 

**Proof:**

Consider a composite system of two subsystems. (This can be extended to as many subsystems as we wish.) Let this system initially be at equilibrium with entropy $S = S(U,X_i)$.

Imagine that there is some internal constraint $Y$ that controls the individual values of the internal extensive variables, while leaving the total values constant. For example, $Y$ could be the position of an internal wall partitioning the two subsystems, such that variations in the internal constraint will change the individual subvolumes but not the total volume. 

Make a change to this internal constraint, and do so reversibly. Do so in such a way that the total energy is also unchanged, $\Delta U = 0$. Work was surely done to change the internal constraint, so there must have been a flow of heat to counter this change, and therefore a change in entropy $S(U,X_i) \rightarrow S'(U,X_i; Y)$.

Now, return the internal constraint to its original value, but do so non-reversibly, and adiabatically. There should be no heat transferred, no work done, and the system should keep all of its total values of $X_i$. Entropy can still change, and ending in the same equilibrium state with all the same values of $U$ and $X_i$, the system should return to its original value $S(U,X_i)$. 

The second law dictates that because this process was non-reversible and adiabatic, $S$ must be greater than $S'$. That is,
$$
S(U,X_i) > S(U,X_i; Y).
$$
In other words, the equilibrium value of the internal constraint is that which maximizes entropy.