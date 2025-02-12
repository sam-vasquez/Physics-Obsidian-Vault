---
tags:
  - Graph-Theory
---
A graph $G$ is defined by its set of vertices (sites) $V = \{v_1,\ldots,v_n\}$ and its set of edges (bonds) $E$.

$k(G)$ denotes the number of connected components of $G$. When this matches $n$, $G$ is a [[Connected Graph]].

Examples: 
The empty graph $E_N$ with $N$ vertices and 0 edges.
The line graph $L_N$ with $N$ vertices and $N-1$ edges connecting vertices $v_i$ and $v_{i+1}$.
The circuit graph $C_N$, a periodic version of the line graph.
The complete graph $K_N$, which connects every pair of vertices with a single edge.

Two vertices are adjacent if they are connected by at least one edge. On a regular lattice, one generally restricts to the case where adjacent vertices are connected by only one edge.

The degree (also called coordination number) $\Delta_v$ of a vertex is the number of vertices that connects to it. A graph is $\Delta$-regular if all vertices have the same degree. 
For a $\Delta$-regular lattice graph, $n(E) = \frac{\Delta}{2} n(V)$.
The thermodynamic limit considers an effective degree $\Delta_{\textrm{eff}} = \lim_{n(V)\rightarrow \infty} \frac{2n(E)}{n(V)}$.

A 1D lattice with periodic boundary conditions is a 2-regular graph, and equivalent to $C_N$. Other 1D lattices are effectively 2-regular.
A 2D square lattice with toroidal boundary conditions is a 4-regular graph, other 2D square lattices are effectively 4-regular.

A cycle in $G$ is a path on adjacent edges that forms a circuit. A tree graph is a connected graph with no cycles. Tree graphs are unique for $n(V) = 1,2,3$, but not unique for $n(V) \geq 4$.

A spanning subgraph $G' = (V,E')$ is a graph containing all vertices but only a subset of edges of $G$. 
A connected spanning subgraph which is cycle-free (a tree) is called a spanning tree. One that is cycle-free but can be disconnected is a spanning forest.

A graph is planar if it is embeddable in a plane without intersecting edges. The planar dual $G^*$ of planar graph $G$ is a graph with one vertex for every simple cycle of $G$, and edges $E^*$ connecting them through every edge $E$.

