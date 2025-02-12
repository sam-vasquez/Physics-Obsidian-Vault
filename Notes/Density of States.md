---
tags:
  - Stat-Mech
---
The **density of states** $\Omega(E)$ is a measure of the number of available microstates with energy $E$ in a thermodynamic system. 

In continuous classical systems, the density of states is defined as 
$$
\Omega(E) = \prod_i \int \frac{dq_i dp_i}{2\pi} \delta(\mathcal{H}(q,p) - E).
$$

As a postulate of statistical mechanics, properties of macroscopic systems can be given by [[Ensemble Average]]s of certain quantities $A$ in this measure, defining the [[Microcanonical Ensemble]]. For example, for a continous classical system, 
$$
\langle A \rangle = \frac{1}{\Omega(E)} \prod_i \int \frac{dq_i dp_i}{2\pi} A(q,p) \delta(\mathcal{H}(q,p) - E).
$$
The density of states is a super-extensive quantity, that is, it grows as $e^{ cN }$ as $N \rightarrow \infty$. To get a quantity representing number of microstates that is properly extensive in $N$,  the [[Entropy]] of the system is defined as $S(E) = \ln \Omega(E)$. 

Consistent with the [[Third Law of Thermodynamics]], there is only one or a finite number of states at the ground state energy of the system.

As an example, the system of $N$ coupled classical harmonic oscillators can have its density of states written as 
$$
\Omega(E) = \prod_i \int \frac{dx_i dp_i}{2\pi} \delta\left( E - \sum_i \left[ \frac{p_i^2}{2m} + \frac{k x_i^2}{2} \right] \right).
$$
[[Density of States of Classical Harmonic Oscillators|It can]] [[Entropy of Classical Harmonic Oscillators|be shown]] that this computes to
$$
\Omega(E) = \left( 2 \sqrt{ \frac{m}{k} } E \right)^N \cdot \frac{1}{E} \cdot \mathcal{A}(2N),
$$
and the entropy of the system is 
$$
S(E) = \ln \left( \left( 2 \sqrt{ \frac{m}{k} } \right)^N \mathcal{A}(2N) \right) + (N-1) \ln E,
$$
where $\mathcal{A}(2N)$ is the area of the unit sphere in $2N$ dimensions. We see from this example that indeed the density of states scales as $e^{ N }$ and the entropy as $N$.

