---
tags:
  - Stat-Mech
---
Consider a composite system that is in thermodynamic equilibrium.

To first order, the temperature, pressure, chemical potential stay the same. At fixed entropy, energy is minimized, other energies are minimized. All as a consequence of [[Energy Minimization Principles]].

In particular, thermal equilibrium says $T_1 = T_2$, $\left( \frac{\partial  U_1}{\partial S_1} \right)_{V,n} = \left( \frac{\partial U_2}{\partial S_2} \right)_{V,n}.$

To second order, the equilibrium should stay stable. The stability condition is that the second derivative is positive,
$$
\left( \frac{\partial^2 U_1}{\partial S_1^2} \right)_{n,V} = \left( \frac{\partial T_1}{\partial S_1} \right)_{n,V} > 0.
$$
This implies $0 < \left( \frac{\partial  T}{\partial S} \right)_{n,V} = \frac{T}{T \left( \frac{\partial  S}{\partial T} \right)_{n,V}} = \frac{T}{C_V} \implies C_V > 0$.
A positive [[Heat Capacity]] is a necessary condition for energy stability. Indeed, if it were negative, it would imply that a hot reservoir could drive a cold reservoir colder and colder.

A similar construction would show that free energy stability depends on 
$$
\left( \frac{\partial^2 F}{\partial V^2} \right)_{T,n} = - \left( \frac{\partial p}{\partial V} \right) > 0.
$$
This implies [[Compressibility]] is positive, therefore things get smaller when you apply pressure. See homework exercises.

Note that each stability condition involves derivatives of conjugate variables. There are others, as well. Unlike [[Maxwell Relations]], which relate derivatives by mixing conjugate variables.

Source: Chandler 2.2
