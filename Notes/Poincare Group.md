---
tags:
  - todo
---
The Poincare group is the group of coordinate transformations that leave the Minkowski metric invariant. This includes the Lorentz transformations together with translations. Their most general form is therefore $x^\mu \rightarrow {\Lambda^\mu}_\nu x^\nu + a^\mu$, where $\Lambda$ satisfies $\eta_{\mu\nu} {\Lambda^\mu}_\lambda {\Lambda^\nu}_\rho = \eta_{\lambda\rho}$.

#### Infinitesimal Transformations

#### Generators and Algebra
Consider an infinitesimal Lorentz transformation, ${\Lambda^\mu}_\nu \simeq \delta^\mu_\nu + {\epsilon^\mu}_\nu$. 
To first order in $\epsilon$, the condition $\eta_{\mu\nu} {\Lambda^\mu}_\lambda {\Lambda^\nu}_\rho = \eta_{\lambda\rho}$ becomes $\epsilon_{\mu\nu} = -\epsilon_{\nu\mu}$, making $\epsilon$ a $d\times d$ antisymmetric tensor with $d(d-1)/2$ independent components. Therefore, there are that many independent generators of the Lorentz algebra. Combine this with the generators of translations to get $d(d+1)/2$ generators of the Poincare algebra. 

> [!thm] Poincare Generators
> The generators of the Poincare Lie algebra are as follows:
> $$
> \begin{align}
> P_\mu &= -i\partial_\mu \\
> M_{\mu\nu} &= -i\left( x_\nu \partial_\mu - x_\mu \partial_\nu \right).
> \end{align}
> $$
> Proof:

^ceba13

> [!thm] Poincare Lie Algebra
> The non-trivial commutation relations among the generators are
> $$
> \begin{align}
> \left[ P_\rho, M_{\mu\nu} \right] &= i(\eta_{\rho\mu}P_\nu - \eta_{\rho\nu}P_\mu) \\
> \left[ M_{\mu\nu}, M_{\rho\sigma} \right] &= i(\eta_{\nu\rho}M_{\mu\sigma} + \eta_{\mu\sigma}M_{\nu\rho} - \eta_{\mu\rho}M_{\nu\sigma} - \eta_{\nu\sigma}M_{\mu\rho}).
>\end{align}
> $$
> Proof:

^7c067f

It happens to be the case that the Poincare group is a subgroup of the [[Conformal Group]], and its Lie algebra is a subset of the [[Conformal Group#^6b2799|Conformal Lie algebra]].
The Poincare group is also an Inonu-Wigner contraction of the de Sitter group and also the anti de Sitter group in the limit of vanishing cosmological constant.

