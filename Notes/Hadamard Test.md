---
tags:
  - Quantum-Information
---
The Hadamard test is a generalization of [[Projective Measurement]]. The only difference is that the operator to be measured is not necessarily unitary, and therefore the probabilities of outcome don't simplify the same way.

The output states are still $\frac{1}{2} (1 \pm \mathcal{O})\ket{ \psi }$.

The probability of measuring 0 on the ancillary bit is now $\frac{1}{4}\bra{ \psi }(1+\mathcal{O}^\dagger)(1+\mathcal{O}^\dagger)\ket{ \psi }$, and the probability of 1 is $\frac{1}{4}\bra{ \psi }(1-\mathcal{O}^\dagger)(1-\mathcal{O}^\dagger)\ket{ \psi }$. 
The expectation value of the ancillary bit is now
$$
1 \cdot \frac{1}{4}\bra{ \psi }(1+\mathcal{O}^\dagger)(1+\mathcal{O})\ket{ \psi } - 1\cdot\frac{1}{4}\bra{ \psi }(1-\mathcal{O}^\dagger)(1-\mathcal{O})\ket{ \psi }
= \frac{1}{2} \bra{ \psi } (\mathcal{O}^\dagger + \mathcal{O}) \ket{ \psi } = \Re(\langle \mathcal{O} \rangle).
$$
This is therefore an indirect measurement of the operator that doesn't collapse the state.  
