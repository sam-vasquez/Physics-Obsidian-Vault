---
tags:
  - Stat-Mech
---
The 1D [[Ising Model]] has Hamiltonian
$$
\mathcal{H} = - J \sum_{(ij)} s_i s_j - \sum H \mu s_i,
$$
$(ij)$ a sum over nearest neighbors. With periodic boundary conditions, this is equivalent to
$$
\mathcal{H} = -J \sum s_i s_{i+1} - H \mu \sum s_i.
$$
(In the thermodynamic limit, the periodic boundary should have vanishing impact, since the domain wall is just a single spin out of $N$ spins.)

The canonical ensemble partition function is 
$$
Z = \sum_{s_1} \ldots \sum_{s_N} e^{ \sum \left[  k s_i s_{i+1} + \frac{1}{2} h (s_i + s_{i+1}) \right]},
$$
where $k \equiv \beta J$ and $h \equiv \beta \mu H$.

See Phy 540 exercise 40 for some exact results with just 3 spins. 


Otherwise, results are obtained using the transfer matrix method. Note that there are four contributions possible per pair of spins $s_i$ and $s_{i+1}$. If both are up, they contribute $e^{ k+h }$ to the partition function. If mixed, $e^{ -k }$. If both down, $e^{ k-h }$. This yields the transfer matrix:
$$
P = \begin{pmatrix} 
e^{ k+h } & e^{ -k } \\
e^{ -k } & e^{ k - h }
\end{pmatrix},
$$
with elements $\bra{ s_i } P \ket{ s_j }$ for each pair of spins in the partition function exponential.
The partition function is rewritten as
$$
Z = \sum_{s_1} \ldots \sum_{s_N} \bra{ s_1 } P \ket{ s_2 } \bra{ s_2 } P \ket{ s_3 } \ldots \bra{ s_N } P \ket{ s_1 }.
$$
All sums aside from $s_1$ can be taken over each resolution of identity to remove each of those sums. 
$$
Z = \sum_{s_1} \bra{ s_1 } P \sum_{s_2} \ket{ s_2 } \bra{ s_2 }\ldots P \ket{ s_1 } = \sum_{s_1} \bra{ s_1 } P^N \ket{ s_1 } = Tr(P^N).
$$
$P$ is real and symmetric, so it can be diagonalized by an orthogonal matrix. The resulting diagonal matrix trivializes the calculation:
$$
Z = \textrm{Tr } D^N = \lambda_+^N + \lambda_-^N.
$$
The eigenvalues of $P$ are $\lambda = e^k \cosh(h) \pm \sqrt{ e^{-2k} + e^{2k} \sinh^2(h) }$.

In thermodynamic limit, the smaller eigenvalue contributes negligibly.
$$
\frac{1}{N} \ln Z \approx \ln \lambda_+.
$$
Pulling out the factor of $e^k$ from $\lambda$, the free energy is
$$
\frac{F}{N} = -k_B T \ln \lambda_+ = -J - k_B T \ln \left[  \cosh(h) + \sqrt{ e^{-4k} + \sinh^2(h) } \right].
$$
When all interactions are turned off, ($h$ and $k$ ($J$) all go to zero), $F/N$ goes to $-k_B T \ln(2)$, as expected from $F(E=0) = -TS$. When only the external field is turned off ($h$ goes to zero), $F/N$ goes to $-k_BT \ln (2 \cosh(k))$.

The energy is 
$$
\begin{align*}
\frac{E}{N} &= -\frac{\partial }{\partial \beta} \ln \lambda_+ \\
&= -\left(
\frac{\partial h}{\partial \beta} \frac{\partial
}{\partial h} + \frac{\partial k}{\partial \beta}
\frac{\partial }{\partial k} \right) \ln \lambda_+ \\
\end{align*}
$$
Todo: simplify.
Todo: Heat capacity, derivation in lecture 5/6.

Using convention $dE = TdS + HdM$, the magnetization is $M = -\left( \frac{\partial F}{\partial G} \right)_T$. This derivative comes out to $\frac{M}{N} = \frac{\mu\sinh(h)}{\sqrt{ e^{ -4k } + \sinh^2(h)} }$, derivation shown in lecture 6. 
When $H$ goes to 0, $M$ goes to 0 for all finite $T$. It only has nonzero spontaneous magnetization at $T=0$.

[[Magnetic Susceptibility]] is 
$$
\chi = \left( \frac{\partial M}{\partial H} \right)_T = \frac{\beta e^{ -4k} \cosh h}{(\sinh^2 h + e^{ -4k })^{3/2}}.
$$
Zero-field susceptibility is $\chi = \beta e^{ 2k }$. Diverges exponentially as $T \rightarrow 0$, indicating a 1-sided phase transition at $T=0$. This is an essential singularity of a second derivative of free energy, and discontinuous singularity in first derivative $M_{H=0}$. This is a peculiar kind of non-analytic behavior among phase transitions.
Nonzero field high temperature: As $T\rightarrow\infty$, $\bar{\chi} \rightarrow \beta$, (Pierre) Curie law. 
Nonzero field low temperature: $T\rightarrow 0$.
$J > 0$: $\bar{\chi} \rightarrow 0$. External field biases the spins toward ferromagnetic order, which approaches a complete ordering at zero temperature. Increasing $H$ wouldn't have any additional ordering effect.
$J < 0$: Competing interactions and frustration, the spin-spin interaction prefers antiferromagnetic spins, magnetic field prefers ferromagnetic order. Which one dominates? See hw.

The (connected) spin-spin correlation is
$$
\langle \sigma_i \sigma_j \rangle - \langle \sigma \rangle^2 = \langle \sigma_i \sigma_j \rangle - m^2.
$$
$$
Z = \sum_{\{\sigma_i\}} e^{ -\beta \mathcal{H} } = \sum_\sigma \prod_{ij} e^{ k \sigma_i \sigma_j }
$$
High temperature expansion:
$$
e^{ k \sigma_i \sigma_j } = 1 + k\sigma_i \sigma_j + \frac{1}{2} (k \sigma_i \sigma_j)^2 +\ldots = 1 + k \sigma_i \sigma_j + \frac{1}{2} k^2 +\ldots
$$
$$
e^{ k\sigma_i \sigma_j } = \cosh k + \sigma_i \sigma_j \sinh k
$$
$$
Z = \sum_\sigma \prod_{ij} \cosh k (1 + v \sigma_i \sigma_j),
$$
$$
Z = (\cosh k)^{e(\Lambda)} \sum_\sigma \prod_{ij} (1 + v\sigma_i \sigma_j ),
$$
where $e(\Lambda) = \frac{\Delta}{2} n(\Lambda)$, $\Delta$ the degree of each site in $\Lambda$. That is, number of edges is coordination number times half the number of sites. So $e(\Lambda) = n$ for a 1D periodic lattice, as expected. 

Example with 3 spins:
$$
Z_3 = (\cosh k)^3 \sum_{\sigma} (1 + \tanh k \sigma_1 \sigma_2)(\ldots)
$$
$$
= (\cosh k)^3 \sum_\sigma \left[ 1 + \tanh k (\sigma_1 \sigma_2 + \sigma_2 \sigma_3 + \sigma_3 \sigma_1) + (\tanh k)^2 (\sigma_1 \sigma_2 \sigma_2 \sigma_3 + cycl.) + (\tanh k)^3 \sigma_1 \sigma_2 \sigma_2 \sigma_3 \sigma_3 \sigma_1 \right]
$$
$$
= 2^3 (\cosh k)^3 (1 + (\tanh k)^3) = (2 \cosh k)^3 + (2 \sinh k)^3 = \lambda_+^3 + \lambda_-^3.
$$
Proceed to calculate 2-spin correlation (See lecture 6). $\langle \sigma_0 \sigma_l \rangle = v^l$, $l \geq 0$. $|\langle \sigma_0 \sigma_l \rangle|$ is a monotonically decreasing function of $l$ the distance between the two spins in lattice units, which makes sense that separating the spins decreases mutual influence. Also monotonically decreasing function of temperature.

Correlation length $\xi$. Take $J > 0$ wlog. For $T \neq T_C$, a connected correlation function in general decays exponentially as the distance between spins gets large: $\langle \sigma_0 \sigma_r \rangle \sim e^{ -r/\xi }$. So
$$
v^r = e^{ -r/\xi } \implies \xi = \frac{1}{\ln (1/v)}.
$$
Since $v$ less than 1 and decreases with increasing temperature, $\xi$ decreases. As $T$ increases, there is greater thermal disorder, correlations die off faster at distance. 

As $T$ goes to zero, or $k$ goes to infinity:
Convenient to write 
$$
\frac{1}{v} = \frac{\cosh k}{\sinh k} = \frac{1 + e^{ -2k }}{1 - e^{ -2k }} \approx (1 + e^{ -2k })(1+e^{ -2k }) \approx 1 + 2e^{ -2k }.
$$
$$
\ln(1/v) \approx \ln (1 + 2e^{ -2k }) \approx 2e^{ -2k }
$$
$$
\xi \approx \frac{1}{2} e^{ 2k }.
$$
Correlation length diverges exponentially as $T\rightarrow 0$, an essential singularity.

Another way to calculate $\chi$:
$$
\beta^{-1} \bar{\chi} = \lim \frac{1}{N} \sum_{ij} \langle \sigma_i \sigma_j \rangle,
$$
$ij$ over all sites. Since the lattice is homogeneous, 
$$
\sum_{ij} \langle \sigma_i \sigma_j \rangle = \sum_j \langle \sigma_0 \sigma_j \rangle,
$$
$j \in \mathbb{Z}$.
Since $\langle \sigma_0 \sigma_{j} \rangle = v^{|j|}$, $\beta^{-1} \bar{\chi} = \frac{1+v}{1-v} = e^{ 2k }$, as we derived before. 







When nearest-neighbor interactions are turned off ($k$ goes to zero), $M = N \mu \tanh(\beta \mu H)$. This replicates ideal results. Todo: Locate and attach that result. Todo: insert plots. See whiteboard for plots.

