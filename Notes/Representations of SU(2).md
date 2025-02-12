---
tags:
  - Rep-Theory
  - todo
---
Todo: Put my representation theory notes here. 

Georgi does this in the following way:

Diagonalize the operator $J_3$. Denote the state corresponding to highest eigenvalue $j$ as $J_3\ket{ j,\alpha } = j \ket{ j,\alpha }$. 

Define $J^\pm = (J_1 \pm i J_2)/\sqrt{ 2 }$.
$$
\left[ J_3, J^\pm \right] = \pm J^\pm
$$
$$
\left[ J^+, J^- \right] = J_3.
$$
If $J_3 \ket{ m,\alpha } = m \ket{ m,\alpha}$, then
$$
J_3 J^\pm \ket{ m,\alpha } = J^\pm J_3 \ket{ m,\alpha } \pm J^\pm \ket{ m,\alpha } = (m \pm 1) J^\pm \ket{ m,\alpha }.
$$
So $m\pm 1$ is the $J_3$ eigenvalue of state $J^\pm \ket{ m,\alpha }$ for all $m$. Equivalently, $J^\pm \ket{ m,\alpha } = N^\pm_m(\alpha) \ket{ m\pm 1,\alpha' }$ for some normalization $N^\pm_m$ and some possible change to $\alpha$.

In particular, since j is the highest $J_3$ eigenvalue, we must have $J^+ \ket{ j } = 0$. Choosing $\braket{ j,\alpha | j,\beta } = \delta_{\alpha\beta}$, we can determine the other $N_m$ through induction. 
Start with $j-1$. We have
$$
J^- \ket{ j,\alpha } = N^-_j(\alpha) \ket{ j-1, \alpha' },
$$
and its overlap with $J^- \ket{ j,\beta } = N_j^-(\beta) \ket{ j-1, \beta' }$ is 
$$
\begin{align}
& N_j^{-*}(\beta)N^-_j(\alpha)\braket{ j-1,\beta' |j-1,\alpha'}  \\
&= \bra{ j,\beta' } J^+ J^- \ket{ j,\alpha' }  \\
&= \bra{ j, \beta' } \left[ J^+, J^- \right] \ket{ j,\alpha' } \\
&= \bra{ j,\beta' } J_3 \ket{ j,\alpha' } = j \delta_{\alpha'\beta'}.
\end{align}
$$
It was used that $J^-J^+ \ket{ j,\alpha' } = 0$. 
We now have $N^-_j(\alpha)^2 = j$, and can choose $N^-_j(\alpha) = N^-_j = \sqrt{ j }$. 
Then,
$$
\begin{align}
&J^+ \ket{ j-1,\alpha } = J^+ \left(\frac{1}{\sqrt{ j }} J^- \ket{ j,\alpha } \right)  \\
&=\frac{1}{\sqrt{ j }} J^+ J^- \ket{ j,\alpha } = \frac{1}{\sqrt{ j }} \left[ J^+, J^- \right] \ket{ j,\alpha } = \frac{1}{\sqrt{ j }} J_3 \ket{ j,\alpha } \\
&= \frac{j}{\sqrt{ j }}\ket{ j,\alpha } = \sqrt{ j } \ket{ j,\alpha } \equiv N^+_{j-1}(\alpha) \ket{ j, \alpha' }.
\end{align}
$$
We find that $N^+_{j-1}(\alpha) = N_j$ and $J^+ \ket{ j-1, \alpha } = N_j \ket{ j, \alpha }$. Note that it turns out to not depend on nor change $\alpha$.

Now, look at $j-2$. Again, we have
$$
J^- \ket{ j-1,\alpha } = N^-_{j-1}(\alpha) \ket{ j-2, \alpha' }.
$$
Again, we find
$$
\begin{align}
& N_{j-1}^{-*}(\beta)N^-_{j-1}(\alpha)\braket{ j-2,\beta' |j-2,\alpha'}  \\
&= \bra{ j-1,\beta' } J^+ J^- \ket{ j-1,\alpha' }  \\
&= \bra{ j-1, \beta' } \left(  \left[ J^+, J^- \right] + J^- J^+\right)\ket{ j-1,\alpha' } \\
&= \bra{ j-1,\beta' } J_3 \ket{ j-1,\alpha' } + \bra{ j,\beta' } N_j^2 \ket{ j,\alpha } \\
&= (j-1)\delta_{\alpha'\beta'} + j\delta_{\alpha'\beta'},
\end{align}
$$
and we get $N^{-2}_{j-1}(\alpha) = N_{j-1}^2 = 2j-1$.

It will be the case that by continuing the process we get a recursion relation between each $N_{j-k}$ and can solve it in terms of $j$ and $j-k$.

If we choose $\ket{ j-1,\alpha }$ orthonormal, then $N_j(\alpha)^2 = j$ and we can choose $N_J(\alpha) = \sqrt{ j }$.