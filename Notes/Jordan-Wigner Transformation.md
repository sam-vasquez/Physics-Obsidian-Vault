---
tags:
  - Stat-Mech
---
The Jordan-Wigner transformation is a mapping between spin operators and fermionic second quantized operators. The mapping from spins to fermions is useful in the study of spin chains, yielding exact results in 1D, and the inverse mapping from fermions to spins is useful in the implementation of fermion Hamiltonians onto quantum computer hardware for simulation.

Quantum spins can be difficult to study in spin chains because second quantized operators don't have simple fermionic or bosonic statistics. 

The transformation effectively works by treating the down and up states as empty and occupied fermion states,
$$
\ket{ \uparrow } \equiv f^\dagger \ket{ 0 }, \ket{\downarrow} \equiv \ket{ 0 }.
$$
The mapping would have to be through operators $f$ and $f^\dagger$ that satisfy
$$
f \ket{ \uparrow } = \ket{ \downarrow }, f \ket{ \downarrow } = 0.
$$
$$
f^\dagger \ket{ \downarrow } = \ket{ \uparrow }, f^\dagger \ket{ \uparrow } = 0.
$$

These can be realized with 
$$
f = \bar{X} - i\bar{Y}, f^\dagger = \bar{X} + i\bar{Y},
$$
that is 
$$
\bar{X} = \frac{1}{2}(f^\dagger + f), \bar{Y} = \frac{1}{2i} (f^\dagger - f),
$$
which can best be worked out from their matrix representations
$$
f = 
\begin{pmatrix} 
0 & 0 \\
1 & 0
\end{pmatrix},
f^\dagger = 
\begin{pmatrix} 
0 & 1 \\
0 & 0
\end{pmatrix}.
$$
$\bar{Z}$ can be written as 
$$
\bar{Z} = \frac{1}{2} \left[ \ket{ \uparrow }\bra{ \uparrow } - \ket{ \downarrow }\bra{ \downarrow } \right] = f^\dagger f - \frac{1}{2}.
$$
For a single spin, these operators $\{\bar{X}, \bar{Y}, \bar{Z} \}$ defined by $f$ and $f^\dagger$ satisfy the same algebra as the Pauli matrices $\{X,Y,Z\}$, and can define a mapping from a single fermion to a single spin.

For several spins, however, we find that independent operators do not commute the way that independent Pauli matrices do. Independent spin operators are meant to commute, but independent fermion operators anticommute.

Jordan and Wigner found a way to correct this in 1D with a phase factor in the definition of each spin,
$$
\bar{X}_j + i \bar{Y}_j = f_j^\dagger e^{ i \phi_j },
$$
$$
\bar{X}_j - i \bar{Y}_j = f_j e^{ -i \phi_j },
$$
where $\phi_j$ contains the sum over all fermion occupancies to the left of $j$:
$$
\phi_j = \pi \sum_{l<j} f_l^\dagger f_l.
$$



Reference:
http://home.uchicago.edu/dtson/phys411/Jordan-Wigner.pdf