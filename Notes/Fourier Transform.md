---
tags:
  - Reference
---
In terms of the angular frequency $\omega = 2\pi \xi$, the Fourier transform is
$$
\hat{f}(\omega) = \int_{-\infty}^\infty dt \; f(t) e^{ -i \omega t },
$$
$$
f(t) = \frac{1}{2\pi}\int_{-\infty}^\infty d\omega \; \hat{f}(\omega) e^{ i\omega t }.
$$
Sometimes the factor of $2\pi$ is split between $\hat{f}$ and $f$ to restore symmetry.

See Woit chapter 11 for the representation theory perspective.


Notable properties:
Each derivative $\frac{d}{dt}$ of $f(t)$ multiplies by a factor of $i\omega$: $\dot{f}(t) = i\omega f(t)$.

[[Parseval's Theorem]]
[[Convolution Theorem]]