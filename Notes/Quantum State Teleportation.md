---
tags:
  - Quantum-Information
---
Qubit entanglement can be exploited to copy
the state of one qubit to the state of the other
qubit (or rather, it "cuts" the state, the
original state is destroyed in accordance with the
no-cloning theorem). 

Suppose Alice has an arbitrary and unknown state
$\ket{ \psi }_1 = a \ket{\downarrow} + b \ket{\uparrow}$ of qubit 1 at A, which shares
entanglement with qubits 2 and 3 in some Bell state $\ket{ \Phi }_{23}$.

For example, consider the Bell state $\ket{ \Phi^+ } = (\ket{ \uparrow\uparrow } + \ket{ \downarrow\downarrow })$.

$$
\begin{align*}
\ket{ \psi }_1 \otimes \ket{ \Phi^+ }_{23} &= \frac{1}{\sqrt{ 2 }}(a \ket{ \downarrow } + b \ket{ \uparrow })_1 \otimes (\ket{ \downarrow\downarrow } + \ket{ \uparrow\uparrow })_{23} \\
&= \frac{1}{\sqrt{ 2 }} (a\ket{ 000 } + b \ket{ 100 } + a \ket{ 011 } + b \ket{ 111 }) \\
&= \frac{1}{\sqrt{ 2 }} \left(  a\frac{\sqrt{ 2 }}{2}(\ket{ \Phi^+ } - \ket{ \Phi^- })\ket{ 0 } + a\frac{\sqrt{ 2 }}{2}(\ket{ \Psi^+ }- \ket{ \Psi^- } )\ket{ 1 } + b \frac{\sqrt{ 2 }}{2} \left(   \ket{ \Phi^+ } + \ket{ \Phi^- }\ket{ 1 } + b \frac{\sqrt{ 2 }}{2}(\ket{ \Psi^+ } + \ket{ \Psi^- })\ket{ 0 }   \right) \right) \\
&= \frac{1}{2} \left( \ket{ \Phi^+ }\otimes(a \ket{ 0 } + b \ket{ 1 }) + \ket{ \Phi^- } \otimes (-a \ket{ 0 } + b \ket{ 1 }) + \ket{ \Psi^+ } \otimes (a \ket{ 1 } + b \ket{ 0 }) + \ket{ \Psi^- } (-a \ket{ 1 } + b \ket{ 0 }) \right) \\
& = \frac{1}{2} \left( \ket{ \Phi^+ } \otimes \ket{ \psi } + \ket{ \Phi^- } \otimes Z \ket{ \psi } + \ket{ \Psi^+ } \otimes X \ket{ \psi } + \ket{ \Psi^- } \otimes iY\ket{ \psi } \right)
\end{align*}
$$

This entanglement of states transfers the information of state 1 to particle 3, but it depends on which Bell state basis it is measured in. 

Alice measures her particles 1 and 2. There are four cases that she can tell Bob to do:
- It is $\ket{ \Phi^+ }$: Please do nothing.
- It is $\ket{ \Phi^- }$: Please apply $Z$ to your particle.
- It is $\ket{ \Psi^+ }$: Please apply $X$ to your particle.
- It is $\ket{ \Psi^- }$: Please apply $-iY$ to your particle.

In every case, Bob's particle collapses to the same state that Alice's did.

A direct application is [[Entanglement swapping]].