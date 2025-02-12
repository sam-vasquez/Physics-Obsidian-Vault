---
tags:
  - Stat-Mech
---
During any reversible process, the inexact differential of the heat of a thermodynamic system is given as a differential of entropy as
$$
\delta Q|_{rev} = T dS,
$$
where $T$ is a strictly positive value corresponding to the inverse of the derivative of entropy with respect to energy (holding all other variables constant), $\left(\frac{\partial S}{\partial U}\right)_{\mathbf{X}} \equiv \frac{1}{T}$.

**Proof:**
Let $S$ be the [[Entropy]] function proposed in the classical thermodynamic statement of the [[Second Law of Thermodynamics]].

Expand the differential of $S$ in terms of internal energy and all state variables $X_i$:
$$
dS = \left( \frac{\partial S}{\partial U} \right)_{\mathbf{X}} dU + \left( \frac{\partial S}{\partial X_i} \right)_U dX_i.
$$

Applying the [[Notes/First Law of Thermodynamics]],
$$
\begin{align}
dS  & = \left( \frac{\partial S}{\partial U} \right)_{\mathbf{X}} \left[ \delta Q|_{rev} + f_i \; dX_i \right] + \left( \frac{\partial S}{\partial X_i} \right)_U dX_i \\
 & = \left( \frac{\partial S}{\partial U} \right)_{\mathbf{X}} \delta Q|_{rev} \; + \left[ \left( \frac{\partial S}{\partial U} \right)_{\mathbf{X}} f_i  \; + \left( \frac{\partial S}{\partial X_i} \right)_U \right] dX_i.
\end{align}
$$
This expression must be true for any process, so suppose that our process is both [[Reversible Thermodynamic Process|reversible]] and [[Adiabatic Process|adiabatic]]. Since the process is adiabatic, $\delta Q = 0$. Since the process is also reversible, $dS = 0$. Therefore, the expression in brackets must be $0$.

Denote $\left(\frac{\partial S}{\partial U}\right)_{\mathbf{X}} \equiv \frac{1}{T}$. The first condition for entropy given by the second law is that $S$ is a monotonic increasing function of energy, therefore $T > 0$. 

Return to the expression for $dS$.
$$
\begin{align}
   dS &= \frac{1}{T} dU + \left( \frac{\partial S}{\partial X_i} \right)_U dX_i  \\
 & = \frac{1}{T} dU - \frac{1}{T} f_i \; dX_i  \\
 & = \frac{1}{T} dU - \frac{1}{T} \delta W  \\
 & = \frac{1}{T} \delta Q|_{rev}.
\end{align}
$$

Now we finally have an expression for heat transfer during reversible processes, thanks to the second law: $\delta Q|_{rev} = T dS$.

Note: it is important to specify that this relation only holds for [[Reversible Thermodynamic Process|reversible]] processes. If the system ever entered a non-equilibrium state during the process, that change wouldn't be describable in terms of integration over differential forms in the state variables, and this analysis wouldn't be accurate. In particular, this is why this statement is consistent with the Second Law of Thermodynamics: entropy is allowed to increase for non-reversible adiabatic processes despite heat staying constant, because in non-reversible processes entropy isn't necessarily coupled to heat.

Note: it is possible to prove all of this without the first law using a purely geometric argument. But I don't want to do that.