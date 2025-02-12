---
tags:
  - Stat-Mech
---
Internal energy is a thermodynamic potential.
$$
dE = T dS - pdV + \sum_i \mu_i dN_i.
$$
It increases due to heat added to the system, and decreases due to work done by the system. When the work is done through mechanical means, inducing a change in volume, this is defined by $-p dV$.

In statistical mechanics, it is connected to the microstates as the [[Ensemble Average]] of microstate total energy with respect to the [[Canonical Ensemble]] partition function:
$$
\langle E \rangle = \frac{1}{Z}\int dE \; E e^{-\beta E} = - \frac{1}{Z}\frac{\partial }{\partial \beta} \int dE e^{ -\beta E } = -\frac{1}{Z}\frac{\partial Z}{\partial \beta} = -\frac{\partial }{\partial \beta} \ln Z.
$$
Its statistical fluctuations are also defined by the partition function:
$$
\begin{align*}
\langle (\delta E)^2 \rangle &= \langle E^2 \rangle - \langle E \rangle^2 \\
&= \frac{1}{Z} \frac{\partial^2 Z}{\partial \beta^2} - \frac{1}{Z^2} \left( \frac{\partial Z}{\partial \beta} \right)^2 \\
&= \frac{\partial^2}{\partial \beta^2} \ln Z.
\end{align*}
$$
The statistical fluctuations are related to the [[Heat Capacity]] at constant volume: $\langle (\delta E)^2 \rangle = k_B T^2 C_V$.

Todo: Reread argument from Landau: $\frac{\delta E}{E} \propto \frac{1}{\sqrt{ N }}$. Strongly peaked around the mean.