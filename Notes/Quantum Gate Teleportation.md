---
tags:
  - Quantum-Information
---
See also, review paper by the professor: T.-C. Wei, Quantum spin models for measurement-based quantum computation

Consider an arbitrary qubit state $\ket{ \psi }_1 = a \ket{ \downarrow } + b \ket{ \uparrow }$, and a second qubit in the state $\ket{ + }_2 = \frac{1}{\sqrt{ 2 }}(\ket{ \downarrow } + \ket{ \uparrow })$. 

Consider a Controlled-Z gate $CZ_{mn} \equiv \ket{ 0 }\bra{ 0 }_m \otimes I_n + \ket{ 1 }\bra{ 1 }_m \otimes Z_n$. 
Apply the gate to the two qubits:
$$
\ket{ \psi } \otimes \ket{ + } \rightarrow a \ket{ 0 } \otimes \ket{ + } + b \ket{ 1 } \otimes Z \ket{ + } = a \ket{ 0 } \otimes \ket{ + } + b \ket{ 1 } \otimes\ket{ - }.
$$

Perform a measurement of the observable $\hat{O}(\xi) = (\cos{\xi} X + \sin{\xi} Y) \otimes I$. The eigenstates of this observable are $\ket{ \pm \xi } \equiv (\ket{ 0 } \pm e^{ i \xi } \ket{ 1 })/\sqrt{ 2 }$ with eigenvalues $\pm1$. This gives (TODO: prove this)
$$
\ket{ \psi' }_2 = \braket{ \pm \xi | \psi,+ } \sim H e^{ i \xi Z/2 } Z^s \ket{ \psi }_2.
$$
Depending on the first input qubit, we teleport it's state to the second output qubit, up to a unitary gate $H e^{ i \xi Z/2 } Z^s$ that depends on the measurement of the first qubit in the $\ket{ \pm \xi }$ basis.
