---
tags:
  - QFT
---
The chiral anomaly is an example of an [[Anomaly]] in certain fermion theories such as [[QED]] and [[QCD]], where the axial current is conserved in the classical action but not the quantized action due to the group measure. This explicitly breaks chiral symmetry.

In general, coupling fermions to a vector gauge potential breaks chiral symmetry.

### Chiral Anomaly of QED$_2$

QED in [[1+1D Spacetime]], also known as the [[Schwinger Model]], has a simple chiral anomaly.

The massless action is 
$$
S = \int d^2x i \bar{\psi} \not \partial \psi = \int d^2x i \psi^\dagger \gamma^0 (\gamma^0 \partial_t \psi - \gamma^1 \partial_x \psi).
$$

With choice of gamma matrix representations 
$$
\gamma^0 = \sigma^1, \gamma^1 = i \sigma^2,\gamma^5 = -\gamma^0 \gamma^1 = \sigma^3,
$$
$$
S = \int d^2x i \psi^\dagger (\partial_t - \gamma^5 \partial_x)\psi.
$$

The fermion is decomposed into chiral components by projectors
$$
\psi_{\pm} = \frac{1}{2} (1 \pm \gamma^5) \psi,
$$
giving 
$$
\psi = 
\begin{pmatrix}
\psi_+\\
\psi_-
\end{pmatrix} =  
\begin{pmatrix}
\chi^+ \\
\chi^-
\end{pmatrix}.
$$
The Dirac fermion essentially decomposes as two separable Weyl (chiral) fermions.

In terms of the chiral fermions, the action becomes
$$
S = \int d^2x i \chi_+^\dagger \partial_- \chi_+ + i\chi_- \partial_+ \chi_-.
$$
$(\partial_\pm = \partial_t \pm \partial_x)$
The equations of motion are $\partial_- \chi_+ = \partial_+ \chi_- = 0$. The general solutions are $\chi_-(t-x)$ and $\chi_+(t+x)$. This indicates that the left-handed massless fermions are constrained to null trajectories moving leftward, and the right-handed massless fermions move rightward.

With the inclusion of the mass term, $m \bar{\psi} \psi = m \psi^\dagger \sigma_1 \psi = m\chi_+^\dagger \chi_- + m\chi_-^\dagger \chi_+$, the equations of motion become $i \partial_- \chi_+ = m \chi_-$ and $i \partial_+ \chi_- = m \chi_+$. That is, $-\partial_-\partial_+ \chi_- = -(\partial_t^2 - \partial_x^2)\chi_- = -\square \chi_- = m^2 \chi_-$ and $-\square \chi_+ = m^2 \chi_+$.
The solutions are 


### Chiral Anomaly of QED$_4$

The (massless) fermion action in QED is
$$
S = \int d^nx i \bar{\psi} \not D \psi.
$$
The action has two global symmetries: 
Vector rotation, $\psi \rightarrow e^{ i\alpha } \psi$, with current $j^\mu = \bar{\psi} \gamma^\mu \psi$.
Axial rotation, $\psi \rightarrow e^{ i\alpha\gamma^5 }\psi$, with current $j_A^\mu = \bar{\psi} \gamma^\mu \gamma^5 \psi$.

In the classical theory, $j_A^\mu$ is conserved. But in the quantum theory,
$$
\partial_\mu j_A^\mu = \frac{e^2}{16 \pi^2} \epsilon^{\mu\nu\rho\sigma} F_{\mu\nu} F_{\rho\sigma}.
$$
See the following two derivations:
Proof by Triangle Diagrams [[Chiral Anomaly Derivation (Triangle Diagrams)]]
Proof by Measure Transformation [[Chiral Anomaly Derivation (Measure Transformation)]]




Reference: https://www.damtp.cam.ac.uk/user/tong/gaugetheory/3anom.pdf