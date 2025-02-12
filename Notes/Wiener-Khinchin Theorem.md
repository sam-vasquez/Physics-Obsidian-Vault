---
tags:
  - Stat-Mech
---
The Wiener-Khinchin theorem effectively defines the [[Fourier Transform]] of a stochastic process, and in doing so defines the [[Power Spectral Density]] as the Fourier transform of the [[Classical Time Correlation Function]]:

$$
S_A(\omega) = \int dt e^{i\omega t} \langle \delta A(0) \delta A(t) \rangle.
$$

Proof:
Define $\delta A^{(T)}(t) = \delta A(t) W_T(t)$, $W_T(t)$ the square function centered on $t=0$. We can integrate this over infinity for simplicity.
By Parseval's theorem,
$$
\lim_{t\rightarrow\infty} \int_{-\infty}^\infty dt |\delta A^{(t)} |^2 = \lim_{T\rightarrow\infty} \int_{-\infty}^\infty \frac{d\omega}{2\pi} |\delta A^T (\omega)|2
$$
$$
P = \int_{-\infty}^\infty \frac{d\omega}{2\pi} \lim_{T\rightarrow\infty} \frac{1}{T} |\delta A^T(\omega)|^2.
$$
$$
\delta A^T(\omega) = \int_{-\infty}^\infty dt e^{ i\omega t } \delta A^T(t)
$$

If $\delta A^T(t)$ is real, then $\delta A^T(-t) \rightarrow (\delta A^T(\omega))^*$.
By conv theorem, $(\delta A^T (\omega))^2 = (\delta A(\omega))^* \delta A(\omega)$
$$
= \int d\tau e^{i \omega \tau} \int dt \left( \delta A^T(t-\tau) \right)^* \delta A^T(t)
$$
Remembering that $S_A(\omega) = \lim_{T\rightarrow\infty} \frac{1}{T} |\delta A^T \omega|^2$
$$
= \int d\tau e^{i\omega \tau} \lim \int dt \delta A(t - \tau) \delta A(t) (W_T(t))^2.
$$
That last term is the step function, integrates to 1. The rest of the second integral is the correlation function,  $\langle \delta A(0) \delta A(\tau) \rangle$, since ergodic principle makes this time average the ensemble average. So the power spectrum is
$$
S_A(\omega) = \int dt e^{i\omega t} \langle \delta A(0) \delta A(t) \rangle.
$$
The correlation function is what gives the power spectrum fluctuations, it's just the fourier transform of the correlation function. Two sided PSD.

When negative frequencies aren't physical, such as in a circuit, you identify the positive frequencies with the negative frequencies, and use the fact that the correlation function is real and even, to get the two-sided power density function
$$
S_A(\omega) = 2 \int_0^\infty dt \cos(\omega t) \langle \delta A(0) \delta A(t) \rangle.
$$
$$
\langle \delta A(0) \delta A(t) \rangle = \int \frac{d\omega}{2\pi} e^{ -i \omega t } S_A(\omega).
$$
This is basically the convolution theorem, but we had to do all that math because of the subtleties of the non-integrable function.