---
tags:
  - Quantum-Information
  - todo
---
Oracle problems are a class of problems that are among the first to be provably superior on quantum computers compared to classical computers.

They are somewhat contrived, but they generally involve probing a black box "oracle" that acts on bits and returns another set of bits, often with some promised condition satisfied by the output. The aim is to determine the function with as few probes as possible.

The simplest example is the [[Deutsch Algorithm]]. A function maps one bit to one bit, and it suffices to determine whether the function is constant $(f(x) = f(x+1))$ or balanced $(f(x) = f(x+1) + 1)$. A classical computer has to determine this with two function calls, a quantum computer can do it with one.

An immediate extension is the Deutsch-Josza algorithm. A function maps $N$ bits to $N$ bits, and this time it is promised that the function is either constant (all inputs map to same value) or balanced (evaluates to 0 for exactly half of all inputs, 1 for the other half). 

In the Bernstein-Vazirani algorithm, a function of $N$ bits to $N$ bits is promised to be linear, $f(x) = a \cdot x$ for some unknown $a$.

Simon's algorithm: [[Simon's Algorithm]] 