---
tags:
  - Quantum-Information
  - todo
---
Generalization of [[Deutsch Algorithm]].
Function promised to be balanced or constant.
Oracle function is implemented by noting that $f(0\ldots0) = 1$ implies an $X$ gate, $f(0\ldots1\ldots0) = 1$ implies a $CNOT$ gate, $f(0\ldots1\ldots1\ldots0) = 1$ implies a $CCNOT$ gate, and so on. This is because a controlled $NOT$ gate implements $(a_1\cdot\ldots\cdot a_n) + b$.

See [[(2010) Quantum Computing and Quantum Information - Nielsen, Chuang]] 1.4.4