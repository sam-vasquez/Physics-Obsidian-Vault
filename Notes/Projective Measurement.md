---
tags:
  - Quantum-Information
---
A projective measurement is a technique developed for quantum state measurement with respect to an operator $\mathcal{O}$ that leaves the output in the eigenstate. This depends on $\mathcal{O}$ being unitary and Hermitian, and it is a special case of the [[Hadamard Test]] which applies to unitary only.

The measurement is done using the following circuit:
![[Projective Measurement Circuit.png|300]] 
The action of the unitary gates is
$$
\begin{align}
(H \otimes I) CNOT_{\mathcal{O}} (H \otimes I) (\ket{ 0 } \otimes \ket{ \psi })
&= (\ket{ + }\bra{ + }\otimes I + \ket{ - } \bra{ - } \otimes \mathcal{O})(\ket{ 0 } \otimes \ket{ \psi })  \\
 & = \frac{1}{\sqrt{ 2 }} \ket{ + } \otimes \ket{ \psi } + \frac{1}{\sqrt{ 2 }} \ket{ - } \otimes (\mathcal{O}\ket{ \psi }).
\end{align}
$$
A measurement in the 0/1 basis results in outcome 0 on the ancillary bit with probability $\frac{1}{2}\bra{ \psi }(1+\mathcal{O})\ket{ \psi }$, and outcome 1 with probability $\frac{1}{2}\bra{ \psi }(1-\mathcal{O})\ket{ \psi }$. In doing so, the state collapses to $\frac{1}{2}(1\pm \mathcal{O})\ket{ \psi }$. This is a projection of the state onto the space of eigenvectors of the operator $\mathcal{O}$.
The expectation value of the ancillary bit outcome is 
$$
\begin{align}
1 \cdot \frac{1}{2}\bra{ \psi }(1+\mathcal{O})\ket{ \psi } - 1\cdot\frac{1}{2}\bra{ \psi }(1-\mathcal{O})\ket{ \psi }
&= \bra{ \psi } \mathcal{O} \ket{ \psi } = \langle \mathcal{O} \rangle.
\end{align}
$$
This is therefore an indirect measurement of the operator that doesn't collapse the state.  
Similar gates can be constructed for operators with different spectra by constructing their eigenspace [[Projectors]] and working backwards.

When $\ket{ \psi }$ is already an eigenvector of $\mathcal{O}$ with eigenvalue 1, the measurement is guaranteed to be 0, and the state remains as $\ket{ \psi }$. [[Quantum Error Correction]] takes advantage of this by defining codes via states that are the stabilizing group of various operators. Whenever an operator is measured and the measurement isn't 0, the state still projects back into the code space and can be corrected as necessary.