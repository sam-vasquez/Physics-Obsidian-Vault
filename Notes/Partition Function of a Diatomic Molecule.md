---
tags:
  - Stat-Mech
---
The statistical mechanics of chemical reactions requires diatomic molecules at minimum. 

Consider two nuclei separated by distance $\vec{R}$. 

Make the [[Born-Oppenheimer Approximation]]. 

Make the **rigid rotor-harmonic oscillator approximation**, that is, assume that the vibrational and rotational wavefunctions can be separated, with the vibrational wavefunction being that of a simple harmonic oscillator: 
$$
\chi(\vec{R}) = \psi_{\textrm{trans}} \psi_{\textrm{rot}} \psi_{\textrm{vib}} = \psi_{\textrm{trans}} \psi_{\textrm{rot}} \psi_{\textrm{SHO}}.
$$
The eigenvalues of the corresponding Hamiltonians are known to be,

so the partition function is 
$$
\begin{align}
Z_1 &= \sum_k e^{ -\beta \frac{\hbar^2 k^2}{2m} } \sum_i g_i e^{ -\beta E_i } \sum_\nu e^{ -\beta (\nu + 1/2)\hbar \omega} \sum_J g_{IJ} e^{ -\beta B J (J + 1) }  \\
&= \left( \frac{V}{\lambda_{\textrm{th}}^3} \right)
\left( g_0 e^{ -\beta E_0 } \right) 
\left( e^{ -\beta \hbar \omega / 2 } \frac{1}{1 - e^{ -\beta \hbar \omega }} \right) 
\left( \sum_J g_{IJ} e^{ -\beta B J (J + 1) } \right).
\end{align}
$$
The last term depends on whether the nuclei are distinguishable or indistinguishable.

If distinguishable:
The two nuclei will have a degeneracy $g_{IJ} = (2I_A + 1)(2I_B + 1)(2J + 1)$, from the spin degeneracies of each individual nucleus and the 