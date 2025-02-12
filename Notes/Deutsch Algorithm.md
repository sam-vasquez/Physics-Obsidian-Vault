---
tags:
  - Quantum-Information
---
The Deutsch Algorithm is a proposed quantum algorithm that can solve a problem in faster than classical time.

Consider an unknown function $f: \{0,1\} \rightarrow \{0,1\}$ that maps single bits. 
The function is said to be either constant ($f(0) = f(1)$) or balanced ($f(1) = (f(0) + 1) \mod 2$). That is, either both bits map to the same bit, or they map to different bits.
The goal is to determine whether the function is constant or balanced.

Classical computers would need two function evaluations to determine this. Quantum computers only need one.

Suppose we have a circuit with two registers, it takes two qubits $\ket{ x } \otimes \ket{ b }$ and outputs $\ket{ x } \otimes \ket{ f(x) + b }$. 
If we input $\ket{ - } = \ket{ 0 } - \ket{ 1 }$ (ignoring normalization) into the second register, we get 
$$
\begin{align}
\ket{ x }\otimes\ket{ - } &\mapsto \ket{ x } \otimes (\ket{ f(x) } - \ket{ f(x) + 1 })  \\
 &= \ket{ x } \otimes (-1)^{f(x)} (\ket{ 0 } - \ket{ 1 })  \\
 & = \ket{ x } \otimes (-1)^{f(x)} \ket{ - }.
\end{align}
$$
The first equality comes from recognizing that $f(x)$ is either 0 or 1.

Generalization: [[Deutsch-Josza Algorithm]] 

Reference:
http://insti.physics.sunysb.edu/~twei/Courses/Fall2024/PHY568/Unit01TheHistoryOfQ.pdf