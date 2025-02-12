---
tags:
  - Quantum
---
The Born-Oppenheimer approximation simplifies quantum chemistry by assuming that the total molecular wavefunction factors into separate electron and nuclear components, each of which satisfies a separated Hamiltonian.

The total molecular wavefunction satisfies the [[Time-Independent Schrodinger Equation]] 
$$
H_m \Psi(\vec{r}, \vec{R}) = E \Psi(\vec{r}, \vec{R}),
$$
where $\vec{r}$ denotes the electron positions, $\vec{R}$ the nuclei positions, and $H_m$ has the following form (neglecting spin interactions):
$$
H = - \sum_I \frac{1}{2m_I}\nabla_I^2(R) - \frac{1}{2} \sum_i \nabla_i^2(r) + \frac{1}{2} \sum_{IJ} \frac{q_I q_J}{r_{IJ}} - \sum_{Ii} \frac{q_I}{r_{Ii}} + \frac{1}{2} \sum_{ij} \frac{1}{r_{ij}},
$$
$I$ an index over nuclei, $i$ an index over electrons, and using [[Atomic Units]].

The Born-Oppenheimer approximation assumes that the wavefunction factors as 
$$
\Psi(\vec{r}, \vec{R}) = \psi_e(\vec{r}) \chi(\vec{R}),
$$
where $\psi_e$ satisfies its own Hamiltonian $H_e \psi_e(\vec{r}; \vec{R}) = \bar{V}(\vec{R}) \psi(\vec{r};\vec{R})$ with
$$
H_{e} = -\frac{1}{2} \sum_i \nabla_i^2(\vec{r}) + \frac{1}{2} \sum_{IJ} \frac{q_I q_J}{r_{IJ}} - \sum_{Ii} \frac{q_I}{r_{Ii}} + \frac{1}{2} \sum_{ij} \frac{1}{r_{ij}}.
$$
The electron energy $\bar{V}$ is referred to as the *adiabatic potential energy surface*. Adiabatic in the sense that the electron wavefunction adapts adiabatically to the configurations of the nuclei. Its dependence on $\vec{R}$ is called *parametric*, in that the nuclei coordinates can change the functional form of the wavefunction but do not appear explicitly in the functional expression. Analogy: a swarm of flies moving around a slow cow will never notice that the cow is moving.

The electron wavefunction can also include spin and vibrational degrees of freedom, with corresponding Hamiltonian terms. All that matters is it doesn't depend on internuclear interactions. 

The nuclear wavefunction is in turn solved by a kinetic nuclear Hamiltonian in the potential field of the *average electron positions*. The adiabatic potential energy surface $\bar{V}$ is combined with the internuclear interaction contributions to get the total *potential energy surface* $V(\vec{R})$, the total energy of the molecule as a function of its nuclear configuration at a particular time.
$$
\begin{align*}
H_{nuc} &= - \sum_I \frac{1}{2m_I} \nabla_I^2(R) + \frac{1}{2} \sum_{IJ} \frac{q_I q_J}{r_{IJ}} + \langle H_e \rangle \\
&= - \sum_I \frac{1}{2m_I} \nabla_I^2(R) + V(\vec{R}),
\end{align*}
$$
$$
H_{nuc} \chi_{nuc} = E \chi_{nuc}.
$$

This approximation works as long as the nuclei are slow and the molecular energies are gapped. It can be violated by strong correlations between electron and nuclear motion, in the form of non-adiabatic coupling or vibronic coupling. And it can be violated by small gaps that permit those strong couplings, such as is the case for certain state transitions in the form of nonadiabatic reaction dynamics, photochemistry excited states, and [[Jahn-Teller Theorem|Jahn-Teller systems]].