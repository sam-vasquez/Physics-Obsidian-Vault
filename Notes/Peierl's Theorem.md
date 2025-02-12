---
tags:
  - Stat-Mech
---
Argument for why the 1D [[Ising Model]] (with $H=0$) can be expected to not have a phase transition whereas the 2D model does. It comes down to analyzing the changes in energy, entropy, and free energy necessary to form domain walls between distinct phases.

The [[Exact Solution of 1D Ising Model]] treats it with periodic boundary conditions, that is, a circle with $N$ spins on it. Assuming only two distinct phases, a circle supports exactly one type of domain wall: two spins. There are $N(N-1)$ choices of these spins. The energy required to form this domain wall would be $4J$, from antialignment of the boundary with its 4 neighboring spins. The change in free energy would be  
$$\Delta F = \Delta E - T \Delta S = 4 J- k_B T \ln (N(N-1)) \approx 4J - 2k_B T  \ln N.$$
This suggests a critical temperature of $T_C = \frac{2J}{k_B \ln N}$ where $\Delta F$ goes from positive to negative. This temperature notably goes to $0$ in the thermodynamic limit, while $\Delta F$ goes negative, and thus by [[Energy Minimization Principles]] it is always favorable to form domain walls, any region with a distinct magnetization will never remain permanently magnetized.

However, in 2D, the domain walls are an entire line cutting the plane. There are many ways to draw such a line. Difficult to quantify due to working with Manhatten distance, but it can be estimated. There are $2N^{1/2}$ choices of starting points, and 3 choices for each turn, so the number of different lines of length $L$ should be roughly $\Omega_L\sim 2N^{1/2}3^L$. The length can be expected to be roughly $L\sim N^{1/2}$. 
The energy of formation of a line of length $L$ is $\Delta E = 2JL$. So the free energy of formation is expected to be roughly
$$
\Delta F = 2 J L - k_B T \ln \Omega_L = 2 J L - k_B T L \ln(3) - 1/2 k_B T \ln N - k_B T \ln(2).
$$
This time it's the $L\sim N^{1/2}$ interaction term that dominates rather than the $\ln N$ entropy term. The critical temperature is at $T_C = \frac{1}{k_B}\frac{2J}{\ln(3)} \approx 1.82$ $J/k_B$. Exact result is 2.269 $J/k_B$.