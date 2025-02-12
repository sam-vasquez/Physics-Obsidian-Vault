---
tags:
  - Graph-Theory
---
The chromatic polynomial $P(G,q)$ is the number of ways of assigning $q$ colors to the vertices of $G$ such that no two adjacent vertices have the same color. Called a proper $q$-coloring of $G$.
Minimum required number of colors is chromatic number $\chi(G)$.

$P(G,q)$ always contains a factor of $q$. Also, since it vanishes for $q=1$, also always contains a factor of $q-1$. If $G$ contains any triangular subgraph, it contains a factor of $q-2$.

$P(G,q)$ for a bipartitie graph always has $P(G,q=2) = 2$. There is a lower bound on $P(G,q)$.

Math application: The map coloring problem and the four-color theorem.

Practical application: the frequency allocation problem. With $n$ radio broadcast transmission stations, each one represented by a vertex, with edges defined for two transmitters if they are closer than a certain specified distance from each other. Assign frequencies from a set of $q$ values to these transmitters, subject to the condition that adjacent stations must use different frequencies. This is a coloring problem, and the number of ways is $P(G,q)$.