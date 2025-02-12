---
tags:
  - Graph-Theory
---
The Tutte polynomial of a [[Graph]] $G$ is 
$$
T(G,x,y) = \sum_{G' \subseteq G} (x-1)^{k(G')-k(G)}(y-1)^{c(G')} = \sum_{G' \subseteq G} (x-1)^{k(G') - k(G)}(y-1)^{e(G')+k(G')-n(G')},
$$
$k(G')$ number of components, $c(G')$ number of linearly independent cycles.

Provides a way of counting various types of subgraphs. 

Although always a polynomial, often the case for many graphs that it can also be expressed as a function with a $1/(x-1)$ prefactor that cancels to achieve the polynomial. That form is often preferred for being closed-form.

Setting $x=1$ restricts to only connected subgraphs, because $k(G') = k(G)$ is necessary for nonzero contribution of $G'$. Similarly, setting $y=1$ restricts to cycle-free subgraphs. 

Setting both restricts to spanning trees, and $T(G,1,1) = N_{ST}(G)$. Setting $x=2$ relaxes connectedness, and $T(G,2,1)$ counts the number of spanning forests. Setting $y=2$ relaxes cycle-free and $T(G,1,2)$ counts the number of connected spanning subgraphs of $G$.

Special case of Tutte polynomial is [[Chromatic Polynomial]], 
$$
P(G,q) = (-q)^{k(G)}(-1)^{n(G)}T(G,x=1-q,y=0).
$$
