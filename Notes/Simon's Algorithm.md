---
tags:
  - Quantum-Information
  - todo
---
Example of a [[Hidden Subgroup Problems]].

string $s$, $f: \{0,1\}^N \rightarrow X$. Hidden string $s$ such that $f(x) = f(y)$ iff $x=y$ or $x=y\oplus s$.
Recall from hw: Hadamard maps $\ket{ 0 }$ to $\sum_z \ket{ z }$, $\ket{s}$ to $\sum_z (-1)^{s\cdot z} \ket{ z }$, $(\ket{ x } + \ket{ x \oplus s })$ to $\sum_{\{s\}^\perp} (-1)^{x \cdot z} \ket{ z }$. 
Algorithm: prepare Hadamard state, apply $U_f: \ket{ x }\ket{ b } \rightarrow \ket{ x }\ket{ f(x) \oplus b }$, apply Hadamard to first register, measure and record value. If dim span of measurements equals $n-1$, linear algebra, otherwise start over. http://insti.physics.sunysb.edu/~twei/Courses/Fall2024/PHY568/Unit01TheHistoryOfQ.pdf