---
tags:
  - Stat-Mech
  - todo
---
The $q$-Potts model is a lattice model of classical spins, in which the spins $\sigma_i$ take on values $\sigma_i \in \{1,\ldots,q\}$. It has interactions
$$
\mathcal{H}_P = -J_P \sum_{ij} \delta_{\sigma_i \sigma_j}.
$$
The 2-Potts model is isomorphic to the [[Ising Model]], with energies scaled by $2J_I = J_P$ and the ground state energy shifted by a constant.

General identity:
$$
e^{ k_p\delta_{\sigma_i \sigma_j} } = 1 + v_p \delta_{\sigma_i \sigma_j} = 1 + (e^{ k_p} - 1) \delta_{\sigma_i \sigma_j},
$$
$$
Z = \sum_{\sigma_i} \prod_{ij} (1 + v_p \delta_{\sigma_i \sigma_j}).
$$
Analogous to Ising model decomposition.
Known as character decomposition.

Can be decomposed as a sum over [[Graph|spanning subgraphs]] 
$$
Z(G,q,v) = \sum_{G' \subseteq G} q^{k(G')}v^{e(G')}.
$$
called Fortuin-Kasteleyn cluster formula. 
Sketch of proof: With circuit graph $C_3$,
$$
Z(C_3) = \sum_{\sigma_i} \prod_{e_{ij}} (1 + v \delta_{\sigma_i \sigma_j})
$$
There are four contributing terms:
1, corresponding to $G'$ with three disjoint vertices, no edges. $e(G') = 0$, $k(G') = 3$, can be achieved in $q^3$ ways.
$v(\delta_{\sigma_1\sigma_2} + \ldots)$, corresponding to the three $G'$s with one edge and one disjoint vertex, $e(G') = 1$, $k(G') = 2$. The first term contributes if $\sigma_1 = \sigma_2$, achievable in $q^2$, same for the other two contributions. So these terms contribute $3q^2v$.
$v^2(\delta_{\sigma_1\sigma_2}\delta_{\sigma_2\sigma_3} + \ldots)$, corresponding to $G'$s with two edges, $e(G') = 2$, $k(G') = 1$. For the first term to contribute, all $\sigma$s must be equal, with $q$ ways of choosing them. Same for other two terms. Total contribution is $3qv^2$.
$v^3(\delta_{\sigma_1\sigma_2}\delta_{\sigma_2\sigma_3}\delta_{\sigma_3\sigma_1})$, corresponding to $G' = G$. Contributes $qv^3$.
Combining, $Z = q^3 + 3q^2v + 3qv^2 + qv^3 = (q+v)^3 + (q-1)v^3$.
Can generalize to
$$
Z(C_n) = (q+v)^n + (q-1)v^n,
$$
matching the transfer-matrix method from [[Exact Solution of 1D Potts Model]]. This is stronger, since this method doesn't require integer $q$.

Since $k(G') \geq 1$ and $e(G') \geq 0$, the cluster representation shows that $Z(G,q,v)$ is a polynomial in $q$ and $v$. 

The cluster formula allows formal generalization of $q$ from integers to nonnegative real numbers. This is a Gibbs measure. But the generalization only applies for FM coupling. Can also generalize $q$ to complex numbers to discuss zeros of $Z$.

In the antiFM model, limit $T\rightarrow 0$, $k\rightarrow -\infty$, $v=e^k-1 \rightarrow -1$, the only spin configurations that contribute to $G$ are thse for which adjacent spins have different values, otherwise the partition function diverges. So the $T=0$ limit of the partition function is the [[Chromatic Polynomial]] $Z(G,q,-1) = P(G,q)$.
Note: $Z$ always contains an overall factor of $q$, therefore also true for $P(G,q)$.

Also close connection to [[Tutte-Whitney Polynomial]]. 
Proof: let $x = 1+\frac{q}{v}$, $y=v+1$. So $q = (x-1)(y-1)$.
Using $n(G') = n(G)$, 
$$
T(G,x,y) = (x-1)^{-k(G)}(y-1)^{-n(G)} \sum_{G' \subseteq G} q^{k(G')}v^{e(G')},
$$
last factor is $Z$,
$$
Z(G,q,v) = (x-1)^{k(G)}(y-1)^{n(G)} T(G,x,y).
$$
Since $Z(C_n,q,v) = (q+v)^n + (q-1)v^n$, get that 
$$
T(C_n,x,y) = \frac{x^n + (xy-y-x)}{x-1}.
$$
Using $\frac{x^n-x}{x-1} = \sum_j^{n-1}x^j$ (for $n\geq 2$), get
$$
T(C_n,x,y) = x + x^2 + \ldots + x^{n-1} + y.
$$


Physical realization of $q=3$: Krypton atoms adsorbed on graphite. They adsorb onto graphene faces. It's larger than the graphene faces. So it maps to a triangular lattice. Adjacent face adsorbtion is not preferred. Get a $q=3$ Potts model on a triangular lattice. 
