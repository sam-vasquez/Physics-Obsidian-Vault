---
tags:
  - Stat-Mech
---
Mean field theory is a general approach to the statistical mechanics of interacting systems, that makes the simplifying approximation that all interactions can be replaced with a pointwise coupling to a field representing the average interaction.

In other words, the goal is to replace a Hamiltonian of the form
$$
\mathcal{H} = -\sum_{ij} A_{ij} s_j - \sum_{ij} s_i s_j
$$
with a Hamiltonian of the form
$$
\mathcal{H}_{MF} = -\sum_{ij} \bar{A}_{ij}(\langle s \rangle) s_j.
$$

It can also identically be achieved by rewriting the Hamiltonian in terms of fluctuations from the mean, and assuming small fluctuations to neglect direct correlations between fluctuations.
$$
s_i s_j = (m + (s_i - m))(m + (s_j - m)) = (m^2 + m \delta s_i + m \delta s_j + \delta s_i \delta s_j).
$$
$$
s_i s_j \approx m^2 + m(s_i - m + s_j - m) = m(s_i + s_j) - m^2.
$$

That said, the fluctuations are really important, and neglecting them will give poor results. For example, compare the critical behavior of the [[Curie-Weiss Mean Field Theory]] to the [[Ising Model]].