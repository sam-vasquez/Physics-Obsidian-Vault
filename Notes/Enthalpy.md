---
tags:
  - Stat-Mech
---
Enthalpy is a thermodynamic potential that is a [[Legendre Transformations|Legendre transformation]] of internal energy $U$ to replace state variables $(S,V,n)$ with conjugate variables $(S,p,n)$.

The motivation is for studying systems with constant pressure where work can be done by varying the volume. For example, liquid-to-gas phase transitions can be modelled by keeping constant pressure and/or particle number, and when the liquid's volume expands when going to gas, it will have done work against the pressure that internal energy would not be sufficient to track.

The [[Thermodynamic Identity]] for enthalpy $H(S,p,n)$ is given as
$$
dH = T dS + V dp + \mu_i \; dn_i.
$$
Note that when pressure is held constant, $dH|_p = TdS = \delta Q$. Enthalpy therefore becomes a convenient way to understand heat transfer. Compare with [[Free Energy]] as a measure of work extractable at constant temperature.
Standard tables exist for the enthalpy change of various chemicals at various pressures. 

This relation between enthalpy and heat also means that the [[Heat Capacity]] at constant pressure can be expressed as 
$$
c_p = \left( \frac{\delta W}{\partial T} \right)_p = \left( \frac{\partial H}{\partial T} \right)_p.
$$
