---
tags:
  - todo
  - Stat-Mech
---
Transfer matrix method by induction.

$$
Z = \textrm{Tr} (T^N),
$$
$$
\bra{ \sigma_i } T \ket{ \sigma_j } = e^{ K_p \delta_{\sigma_i\sigma_j} }
$$
$e^{ K_P }$ on diagonal, 1 on off-diagonals. ($K_P = \beta J_P$)
For $q=3$: $\lambda = y-1,y-1,y+2$. $(y = e^{ K_p })$
Can prove by induction on $q$ that eigenvalues of $T$ are $y-1$ with multiplicity $q-1$ and $y+q-1$ with multiplicity 1, giving
$$
Z = \lambda_{max}^N + (q-1)\lambda_{min}^N = (q+v_p)^N + (q-1)v_p^N,
$$
$v_p = y-1$.
$$
\beta\frac{F}{N} = \ln (q + e^{ K_P } - 1).
$$
$$
\frac{U}{N} = -\frac{\partial f}{\partial \beta} = -J_P \left( \frac{y}{y+q-1} \right).
$$
As $T\rightarrow\infty$, $k_P \rightarrow0$, $U/N \rightarrow -J_P/q$.
As $T\rightarrow 0$, $J>0$, $U/N \rightarrow -J_P$.

Specific heat $\bar{C} = \frac{\partial \bar{U}}{\partial T} = \frac{k_B k_P^2 (q-1) e^{ k_p }}{(e^{ k_p } + q - 1)^2}$.

