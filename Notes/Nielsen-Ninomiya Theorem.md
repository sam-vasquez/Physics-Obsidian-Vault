---
tags:
  - Quantum
---
Fermions put on a discrete lattice will always have equal numbers of left and right handed Weyl fermions. This is the result of extra poles in the Dirac propagator that cannot be made to disappear in the continuum limit. 



https://www.damtp.cam.ac.uk/user/tong/gaugetheory/4lattice.pdf gives two proofs: one directly from dispersion relations, the other from the Berry phase.

As a consequence, one of the following must fail:
- The propagator $D(k)$ is continuous (periodic) within the Brillouin zone. 
	- Violating this breaks locality. Bad.
- $D(k) \approx \gamma^\mu k_\mu$ for $k \ll 1/a$, such that the theory looks like a massless Dirac fermion when momentum is small.
- $D(k)$ has poles only at $k=0$. In other words, there are no [[Fermion Doubling|fermion doublers]].
- The theory preserves chiral symmetry, $\{\gamma^5,D(k)\} = 0$.
	- Violating this makes studying chiral gauge theory impossible. Even with QCD being non-chiral, we explicitly break chiral symmetry early and it makes studying the chiral anomaly difficult. These are Wilson fermions, and they lead to lots of spurious annoying terms.

It is often the fermion doublers that are permitted. They can be partially resolved using [[Staggered Fermions]].

Other approaches include domain wall fermions in 4+1D, and symmetric mass generation.