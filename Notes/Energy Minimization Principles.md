---
tags:
  - Stat-Mech
  - todo
---
The [[Entropy Maximization Principle]] implies an energy minimization principle. Maximizing entropy for fixed energy is equivalent to minimizing energy for fixed entropy.
TODO: How?

The free energy is minimized. If a system is placed in thermal equilbrium with a large bath, we have
$$
\Delta S_{\textrm{bath}} = \frac{\Delta E_{\textrm{bath}}}{T} = \frac{- \Delta E_{\textrm{sys}}}{T}.
$$
The composite system would maximize its entropy, so change in total entropy $\Delta S_{\textrm{sys}} + \Delta S_{\textrm{bath}}$ is positive. So this implies that 
$$
\Delta S_{\textrm{sys}} - \frac{\Delta E_{\textrm{sys}}}{T} \geq 0,
$$
which rearranges to $\Delta F \leq 0$.
This shows that at a given temperature, the free energy is what's minimized. At low temperatures, the energy dominates and is minimized, at high temperatures, the entropy dominates and is maximized. This is demonstrated by the example at [[Chain of Two-Level Systems (Statistical Mechanics)]].

A similar construction shows that Gibbs free energy is minimized, $\Delta G|_{T,p} \leq 0$.

Minimized free energy implies isothermal [[Compressibility]] is positive. 
$$
\left( \frac{\partial p}{\partial V} \right)_T = - \frac{\partial^2 F}{\partial V^2} 
$$


Reference: Chandler ch 2.2.