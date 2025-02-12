---
tags:
  - Stat-Mech
  - todo
---
The Van der Waals model introduces a rudimentary approximation of particle interactions to the classical ideal gas.

In an ideal gas, energy quantities extensive in $N$ and logarithmic in number density. With interactions enabled, mean field theory, expect scale as $\rho^2 V = N \rho$, which is extensive in $N$. Mean field theory writes $E_{MF}(\rho) = - a \rho^2 N$. 
In van der waals, implements this through a physical volume per particle, $V \rightarrow  V - N b$. That makes
$$
F = N \left[ k_B T \ln \left( \frac{\rho \lambda^3}{1-\rho b} \right) - k_B T - \rho a \right].
$$
Neglecting internal degrees of freedom of course. 
We showed in homework that
$$
p = - \rho^2 a + \rho \frac{k_B T}{1 - \rho b}.
$$
$$
\mu = \frac{G}{N} = \frac{F + pV}{N} = k_B T \ln \left( \frac{\rho \lambda^3}{1 - \rho b} \right) - 2 \rho a + k_B T \frac{\rho b}{1 - \rho b}.
$$

Above critical temperature, $k_B T_C = \frac{8a}{27b}$, there are no bound states, it's full of particles, everything is a gas, below, it's on the right, it can be a liquid/solid. See $\Delta G/N$ vs $\rho b$ plot in github.

Now plot isotherms in p-V plot. Take pressure, volume, and temperature as values relative to the critical temperature. You get $\left( p_r + \frac{3}{v_f^2}  \right) (3 v_r - 1) = 8 T_r$. 