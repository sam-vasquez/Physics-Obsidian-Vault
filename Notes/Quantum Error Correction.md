---
tags:
  - Quantum-Information
---
Quantum error correction encodes quantum states in such a way that errors can be detected by [[Projective Measurement]] with respect to stabilizing operators, and therefore corrected without collapsing the effective wavefunction. It mainly takes advantage of the group theory of the Pauli group, in particular many results use the fact that Pauli group elements always either commute or anticommute.

A subgroup of the Pauli group defines the stabilizer group. The vector space left invariant by this stabilizer group is a $2^{d-k}$ dimensional space, $d$ the dimension of the Pauli group, $k$ the dimension of the stabilizer group. The basis of this space forms the states of $d-k$ logical qubits. Logical operators are represented by [[Coset|cosets]] of the stabilizer group in its normalizer. The normalizer happens to be equivalent to the centralizer. Errors that can be corrected are represented by cosets of the normalizer in the Pauli group. 

Caveat: cannot correct errors that are inside the normalizer but not the stabilizer.

https://qubit.guide/14.4-logical-operators



Qudit error correction:
[[(2011) Standard form of qudit stabilizer groups]]
[[(1998) Fault-Tolerant Quantum Computation with Higher-Dimensional Systems]]


