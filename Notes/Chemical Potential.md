---
tags:
  - Stat-Mech
---
The chemical potential is an [[Extensive and Intensive State Variables|extensive]] variable that couples to work done by adding and removing particles from a thermodynamic system. 

In the usual internal energy formulation, this comes up in 
$$
dU = TdS - p dV + \mu dn,
$$
chemical potential $\mu = \left( \frac{\partial U}{\partial n} \right)_{S,V}$.

However, this isn't easy to imagine computing in reality. 

A more useful expression comes from the [[Free Energy]]:
$$
dF = -S dT - p dV+ \mu dn,
$$
$$
\mu = \left( \frac{\partial  F}{\partial n_i} \right)_{T,V}.
$$
And also from the [[Gibbs Free Energy]]:
$$
dG = -S dT + V dp + \mu dn_i,
$$
$$
\mu = \left( \frac{\partial G}{\partial n_i} \right)_{T,p}.
$$
In fact, since $G$ is only extensive in $N$, it can be written as
$$
G = \mu(p,T) N.
$$

When a composite system is not yet in chemical equilibrium, the difference in chemical potentials drives particle number transport:

TODO: why is $\delta U = 0$ if the system is not in equilibrium?

$\delta U = 0$, volume is internally constrained, temperature and pressure are both already equal, entropy can and will still go up.
$$
0 = \Delta U = T \Delta S + \mu_1 \Delta n_1 + \mu_2 \Delta n_2
$$
$$
\Delta S = \left( \frac{\mu_1}{T} - \frac{\mu_2}{T} \right) \Delta n_1 > 0.
$$
If $\mu_1 > \mu_2$, $\Delta n_1 < 0$.
If $\mu_1 < \mu_2$, $\Delta n_1 > 0$.

It is generally observed that at high temperatures, the chemical potential becomes large and negative. This is because $\mu = \frac{G}{N}$, where $G = U + pV - TS$, entropy itself becomes large at high temperatures, and the $TS$ term dominates. 

Chemical potential usually written as a sum of contributions from different sectors of Hamiltonian. See Stat Mech problems 30, etc. Todo, locate these problems.

Chemical potential of an ideal solution can be written in terms of its concentration inside the standard background. Todo, derive from sackur tetrode.
$$
\mu_i = \mu^\circ + k_B T \ln\left( \frac{\rho_i}{\rho^\circ} \right).
$$
Non-ideal behavior of the solution is described via the activity coefficient $\gamma(\rho_i,T)$, where $\gamma = 1$ defines the ideal solution.
$$
\mu_i = \mu^\circ + k_B T \ln\left( \gamma_i \frac{\rho_i}{\rho^\circ} \right).
$$
Writing the condition for chemical equilibrium of a reaction with coefficients $v_i$ as  
$$
\prod_i \lambda_i^{v_i} = 1,
$$
absolute activity $\lambda_i = e^{ \beta \mu_i }$, we get $e^{ \beta (\mu_i - \mu^\circ) } = \gamma \frac{\rho_i}{\rho^\circ} = \lambda_i$. The activity coefficient characterizes the non-linear relationship between the concentration of the species and its chemical activity.
When the interaction terms in the thermodynamic potential, in practice usually the free energy, are isolated, $F = F_{\textrm{ideal}} + F_{\textrm{non-ideal}}$, the activity coefficient is simply the chemical potential contribution from that term:
$$
k_B T \ln(\gamma_i) = \left( \frac{\partial F_{\textrm{non-ideal}}}{\partial N_i} \right)_{T,V}.
$$
See Stat mech problem 30 for demonstration with an electrolytic solution.