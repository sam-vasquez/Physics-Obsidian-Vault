---
tags:
  - Reference
---
### Single Dimensional Case


### Multi-Dimensional Case
If $g(v) = f(Av)$ for some function $g: \mathbb{R}^m \rightarrow \mathbb{R}^n$ and $f: \mathbb{R}^l \rightarrow \mathbb{R}^n$, then $\nabla g(v) = A^T \nabla f(Av)$.

**Proof**
Let $a$ be an index over $\mathbb{R}^n$, $i,j,k$ be indices over $\mathbb{R}^m$, and $\mu$ an index over $\mathbb{R}^l$.
$$
\begin{align*}
    [ \nabla g(v) ]_{ia} = \partial_i g_a(v_j) &= \partial_i f_a(A_{\mu k}v_k) \\
                        &= \partial_i(A_{\mu k} v_k) (\partial f(Av)	)_{\mu a} \\
                        &= A_{\mu k} \delta_{ik} (\partial f	)_{\mu a} \\
                        &= A_{\mu i} (\partial f	)_{\mu a} \\
                        &= (A^T)_{i \mu} (\partial f	)_{\mu a} \\
                        &= [A^T \nabla f]_i{}_a.
\end{align*}
$$
