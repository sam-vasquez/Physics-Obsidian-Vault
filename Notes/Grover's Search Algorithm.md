---
tags:
  - Quantum-Information
---
Grover's search algorithm is a quantum search algorithm with quadratic speedup. If there are $N$ items and $M$ solutions, a classical algorithm completes in $O(N/M)$ operations, compared with Grover's search completing in $O(\sqrt{ N/M })$ operations. It can be applied to problems as simple as searching for items in a list, or as general as any problem that involves combing through a collection of items/routes to find solutions. 

Consider a search space of $N = 2^n$ elements, stored in $n$ bits, with $M$ solutions. Represent the search problem by an [[Oracle Problems|oracle]] $f(x)$, $0 \leq x \leq N$ an integer, such that $f(x) = 1$ is $x$ is a solution and $f(x) = 0$ otherwise. Its unitary implementation is 
$$
\ket{ x } \ket{ q } \mapsto \ket{ x }\ket{ q \oplus f(x) },
$$
where $\oplus$ denotes addition modulo 2. When $\ket{ q } = \ket{ - }$, the action is
$$
\ket{ x }\ket{ - } \mapsto (-1)^{f(x)} \ket{ x } \ket{ - },
$$
through a phase kickback mechanism very similar to that of the [[Deutsch Algorithm]], making the effective action
$$
\ket{ x } \mapsto (-1)^{f(x)} \ket{ x }.
$$
Explicitly, this is
$$
\mathcal{O} = \sum_x (-1)^{f(x)} \ket{ x } \bra{ x } = I - 2 \sum_{x \textrm{ marked}} \ket{ x } \bra{ x }.
$$

The search algorithm proceeds as follows:
1. Begin with the equal superposition state 
$$
\ket{ s } = \ket{ ++\cdots+ } = \frac{1}{N^{1/2}} \sum_{x=0}^{N-1} \ket{ x },
$$
the superposition of qubit binary representations of all integers $x$. 
1. Apply the oracle $\mathcal{O}$.
2. Apply the Hadamard transform $H^{\otimes n}$.
3. Shift the phase of every state except $\ket{ 0 }$ by $-1$.
4. Apply the Hadamard transform $H^{\otimes n}$.
5. Repeat steps 2-5.
The combined effect of steps 3-5 is to apply the operator
$$
\mathcal{U} = 2 \ket{ s } \bra{ s } - I.
$$
This is called a "reflection about the mean", because its action on a general state $\ket{ \alpha } = \sum_k \alpha_k \ket{ k }$ is
$$
\alpha_k \mapsto -\alpha_k + 2 \langle \alpha \rangle.
$$

The Grover algorithm can be sumarized as a single unitary $\mathcal{G} = \mathcal{U} \mathcal{O}$, which corresponds to a rotation in the space spanned by the starting vector and the vector consisting of a superposition of all solutions to the search problem. The angle satisfies
$$
\sin \theta = \frac{2 \sqrt{ M (N-M) }}{N}.
$$
See NC 6.1.3.

The algorithm can also be [[Adiabatic Implementation of Grover's Search Algorithm||implemented on]] an [[Adiabatic Quantum Computer]].

Reference: Nielson & Chuang Ch. 6
See also http://insti.physics.sunysb.edu/~twei/Courses/Fall2024/PHY568/Unit04GrindingGatesInQC.pdf
The basic algorithm is described in Section 6.1. 
In Section 6.2 we derive the algorithm from another point of view, based on the quantum simulation algorithm of Section 4.7. 
Three important applications of this algorithm are also described: 
quantum counting in Section 6.3, 
speedup of solution of NP-complete problems in Section 6.4, 
and search of unstructured databases in Section 6.5. 
One might hope to improve upon the search algorithm to do even better than a square root speedup but, as we show in Section 6.6, it turns out this is not possible.
We conclude in Section 6.7 by showing that this speed limit applies to most unstructured problems.