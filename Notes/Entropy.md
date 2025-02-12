---
tags:
  - Stat-Mech
---
Entropy is a measure of statistical disorder that characterizes the number of possible microstates of the system in an extensive manner. As a quantity, entropy grows proportional to $N$ as $N \rightarrow \infty$.

As a fundamental assumption of the [[Second Law of Thermodynamics]], [[Adiabatic Process|adiabatic processes]] always increase the entropy of a system, or it stays the same when [[Reversible Thermodynamic Process|reversible]]. In classical thermodynamics this is most notably supported by the [[Entropy Maximization Principle]] that says that internal constraints on a composite system tend toward increasing entropy to a maximum. 

According to the [[Third Law of Thermodynamics]], when the system is at its ground state energy, there is only one or a finite number number of microstates. The entropy is therefore either zero, or constant and can be taken as zero.

Entropy as a derivative of thermodynamic potentials can be expressed as 
$$
S = -\left( \frac{\partial F}{\partial T} \right)_{V,n},
$$
and has Maxwell relations:
$$
\begin{align}
\left( \frac{\partial S}{\partial V} \right)_{T,N} &= \left( \frac{\partial p}{\partial T} \right)_{V,N} \textrm{(from F)} \\
\left( \frac{\partial S}{\partial N} \right)_{T,V} &= - \left( \frac{\partial \mu}{\partial T} \right)_{V,N} \textrm{(from F)}  \\

\end{align}
$$

In the statistical mechanics picture, entropy is a measure of information of the underlying state occupation statistics. If the probability of an energy state $E_i$ being occupied is $p_i$, the entropy is 
$$
S = - k_B \sum_i p_i \ln p_i.
$$
In the [[Microcanonical Ensemble]], all probabilities are $p_i = \frac{1}{\Omega}$, so the entropy is related to the number of microstates as
$$
S = k_B \ln \Omega.
$$
This is justified by recognizing that the number of microstates is increasing with energy and multiplicative. This also justifies the fact that entropy is typically increasing with temperature, when more states become thermally accessible.
