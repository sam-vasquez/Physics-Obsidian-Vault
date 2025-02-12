---
tags:
  - Stat-Mech
---
[[Mean Field Theory]] of the [[Ising Model]]. It predicts a spontaneous magnetization, and exact critical exponents for 4D and above. (This motivates sketchy $4-\epsilon$ dimension proofs of 3D results).

$$
\mathcal{H}_{IM} = - J \sum_{(ij)} s_i s_j - \mu H \sum s_i,
$$
Expressing the Ising model Hamiltonian as 
$$
\mathcal{H} = -\mu \sum_i H_i(m) s_i,
$$
the coefficients are determined by 
$$
H_i = -\frac{1}{\mu} \frac{\partial }{\partial s_i} \left( \mathcal{H}_{IM} \right) = H + \frac{J}{\mu} \sum_{\langle j \rangle}^q s_j,
$$
where $\langle j \rangle$ denotes the $q = 2^d$ nearest neighbors of $s_j$ (the coordination number).

The mean-field approximation replaces the value of all $q$ nearest neighbors with $\langle s_i \rangle \equiv m$, that is,
$$
m = \langle s_i \rangle = \sum_{s_i} \frac{1}{Z} e^{ -\beta (-\mu H_i s_i) } s_i,
$$
$$
H_i \rightarrow H + \frac{J}{\mu} q m.
$$
todo: write Hamiltonian.
$$
\mathcal{H}_{MT} = -\mu H \sum_i s_i - J 
$$
This gives a transcendental self-consistency equation for $m$:
$$
m = \frac{e^{ \beta (\mu H + J q m) } - e^{ - \beta (\mu H + Jqm) }}{e^{ \beta(\mu H + Jqm) } + e^{ -\beta (\mu H + Jqm) }} = \tanh (\beta \mu H + \beta J qm).
$$
Todo: attach plot from whiteboard. (See also Pathria Figure 12.6)3

With the external field turned off, $H=0$:
$$
m = \tanh(\beta J q m).
$$
Using the plot as guidance, it has nonzero solutions when the lines $y = m$ and $y = \tanh(\beta J q m)$ intersect. This is only when $\beta J q > 1$. So the **critical temperature** is $T_c = \frac{Jq}{k_B}$.

| Dim | $T_C$ Predicted ($j/k_B$) | $T_C$ True ($J/k_B$) |
| --- | ------------------------- | -------------------- |
| 1   | 2                         | 0                    |
| 2   | 4                         | 2.269                |
| 3   | 6                         | 4.513                |
| 4   | 8                         | 6.68                 |
In the limit of high dimension, the result for critical temperature becomes exact.

For $T < T_C$, the solution $m=0$ is unstable, while the nonzero solutions are stable, indicating a physical distinction in phases and thus a phase transition. See https://www.pas.rochester.edu/~stte/phy418S24/units/unit_4-3.pdf

Very close to and below the critical temperature $T_C = \frac{Jq}{k_B}$, the magnetization is small. The tanh can be expanded to third order in $\beta$:
$$
\begin{align*}
m &\approx \beta J qm + \beta \mu H - \frac{1}{3} (\beta J q m + \beta \mu H)^3 \\
&= \frac{T_c}{T} m + \beta \mu H - \frac{1}{3} \left( \frac{T_c}{T} m + \beta \mu H\right)^3
\end{align*}
$$
Further assuming $H$ is small, the cubic can be expanded to leading order in $H$:
$$
m \approx \frac{T_C}{T} m - \frac{1}{3} \left( \frac{T_C}{T} \right)^3 m^3 + \beta \mu H - \left( \frac{T_C}{T} \right)^2 m^2 \beta \mu H.
$$
Rearranging:
$$
\left( 1 - \frac{T_C}{T} \right)m + \frac{1}{3} \left( \frac{T_C}{T} \right)^3 m^3 = \beta \mu H \left( 1 - \left( \frac{T_C}{T} \right)^2 m^2 \right).
$$
Move the right side term to the left and expand to third order in $m$ via $(1-x)^{-1} \approx 1+x$:
$$
\left( 1 - \frac{T_C}{T} \right)m + \frac{1}{3}\left( \frac{T_C}{T} \right)^3 m^3 + \left( \frac{T_C}{T} \right)^2 \left( 1 - \frac{T_C}{T} \right) m^3 = \beta \mu H.
$$
Introduce the reduced temperature $\tau$ by expanding to first order around $T = T_C$, using $1 - \frac{T_C}{T} \approx \frac{T-T_C}{T_C} = \tau$.
$$
\tau m + \frac{1}{3} ( 1 - \tau )^3 m^3 + (1-\tau)^2 \tau m^3 = \beta \mu H.
$$
Finally, expand to first order in $\tau$:
$$
\tau m + \frac{1}{3}m^3 = \beta \mu H.
$$
This lets us define the critical exponents.

Magnetization at $H=0$:
$$
\tau m + \frac{1}{3}m^3 = 0 \implies m = 
\begin{cases}
\pm \sqrt{ 3 \tau } & T < T_C \\
0 & T > T_C
\end{cases}
$$
Thus $\beta^- = \frac{1}{2}$ and $\beta^+ = 0$.
However, Onsager's exact solution gives $\beta = \frac{1}{8}$.

Reference: https://www.pas.rochester.edu/~stte/phy418S24/units/unit_4-4.pdf