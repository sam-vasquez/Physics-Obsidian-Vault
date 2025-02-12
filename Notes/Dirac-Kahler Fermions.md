---
tags:
  - Rep-Theory
---
Solutions to geometric analogue of Dirac equation.

Can be defined on any manifold. 

For fields $\Phi$ that are a linear combination of all differential forms of all degrees in $N$ dimensions,
$$
\Phi = \phi^{(0)} + \phi^{(1)}_{\mu_1} dx^{\mu_1} + \ldots + \phi^N_{\mu_1\ldots\mu_N} dx^{\mu_1} \wedge \ldots \wedge dx^{\mu_N},
$$
the Dirac-Kahler equation replaces the Laplacian operator with the Laplace-de Rham operator $(d - *d*)$, $d$ the [[Exterior Derivative]], $*$ the [[Hodge Star Operator]], yielding
$$
(d - *d* + m)\Phi = 0.
$$

The Dirac-Kahler fermions can be very naturally discretized through the correspondence between exterior algebras and simplicial complexes.  


In flat 3+1D, the [[Staggered Fermions]] transform in the representation
$$
\left[ \left( \frac{1}{2}, 0 \right) \oplus \left( 0,\frac{1}{2} \right) \right] \otimes \left[ \left( \frac{1}{2}, 0 \right) \oplus \left( 0,\frac{1}{2} \right) \right] 
$$
$$
= 2(0,0) \oplus 2\left( \frac{1}{2},\frac{1}{2} \right) \oplus (1,0) \oplus (0,1).
$$
Objects in this representation can in general be written in the same form as $\Phi$. With 16 basis differential forms and 16 degrees of freedom in the fermions, the degrees of freedom can indeed be repackaged to show that
1. The staggered fermions satisfy the Dirac-Kahler equation
2. The Dirac-Kahler equation in flat 3+1D is equivalent to four copies of the ordinary Dirac equation.
This is the sense in which each staggered chiral fermion is equivalent to four (not sixteen) Dirac fermions.

