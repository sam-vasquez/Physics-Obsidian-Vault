---
tags:
  - Graph-Theory
  - todo
---
Consider coloring a bipartite graph with $q$ colors. The number of ways of achieving this is [[Chromatic Polynomial]] $P(G_b,q)$.

Assign one color to all vertices in $G_1$, which can be done in any of $q$ ways. Then, independently for each vertex in $G_2$, assign a color from the remaining $q-1$ colors.

Due to the correspondance with the $T=0$ antiferromagnetic [[Potts Model]], this coloring minimizes the energy. Ends up with nonzero ground state entropy, constituting a counterexample to the [[Third Law of Thermodynamics]].

Example: ice. tetragonal bonding, hydrogen bonds have dipoles constrained by local net electric neutrality. represent it with a square directed graph. For a given vertex, $4C2=6$ allowed configurations. Approximate number of configurations using number of arrow configurations $2^{n(E)}$ constrained via $\frac{6}{2^4} = \frac{3}{8}$. $Z \approx 2^{n(E)} \left( \frac{3}{8} \right)^{n(V)}$, 

Residual entropy per molecule is measured to be $S_0 = 0.41 k_B$, corresponds to $\Omega = 1.5$. 