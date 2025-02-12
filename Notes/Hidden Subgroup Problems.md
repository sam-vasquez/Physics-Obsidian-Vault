---
tags:
  - Quantum-Information
  - todo
---
Let $f$ be a function from a finitely generated group $G$ to a finite set $X$ such that $f$ is constant on the [[Coset|cosets]] of a subgroup $K$, and distinct on each coset. Given a quantum [[Oracle Problems|oracle]] for performing the unitary transform $U \ket{ g } \ket{ h } = \ket{ g } \ket{ h \oplus f(g) }$, for $g \in G$, $h \in X$, $\oplus$ a binary operation on $X$, the *hidden subgroup problem* is to find a generating set for $K$.

Example: [[Simon's Algorithm]], with $G = (\{0,1\}^n, \oplus)$, $X$ any finite set, $K = \{0,s\}$ for some bitstring $s$, $f(x) = f(x \oplus s)$. 

Reference: [[(2010) Quantum Computing and Quantum Information - Nielsen, Chuang]] Sec 5.4.3