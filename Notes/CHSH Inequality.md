---
tags:
  - Quantum-Information
---
The CHSH inequality is an inequality of expected values of random variables that can only be $\pm1$-valued. If $a,a',b,b'$ are four such random variables, then
$$
|\mathbb{E}(ab + ab' + a'b - a'b')| = |\sum_{aba'b'} \mathbb{P}(a,b,a',b')(ab+ab'+a'b-a'b')| \leq 2.
$$

It can be proven that classical local hidden-variable
theories obey this inequality, but quantum
mechanics does not, making a proof of [[Bell's Theorem]].

Let $\lambda$ be the hidden variable of a [[Hidden Variable Theory]]. 

Consider an experiment where a source emits pairs of photons, one at location $A$, one at location $B$. 

At each location, we make a measurement of the photon's helicity state with respect to some measurement axis $a$ and $b$. Assume that the outcome of measurement depends on the hidden variable.

If locality holds, then the expected value of all measured outcomes is an average over the hidden variable,
$$
\mathbb{E}_L(a,b) = \int d\lambda \; \mathbb{P}(\lambda) A(a,\lambda) B(b,\lambda),
$$
where $A(a,\lambda)$ and $B(b,\lambda)$ denote the measured values $\pm1$ of $A$ and $B$ in the $a$ and $b$ basis, which according to the hidden local variable theory, is not random as quantum mechanics suggests, but rather is a function of an unknown variable $\lambda$ with an unknown but stationary probability distribution $\mathbb{P}$. Locality requires that $A$ does not depend on $b$, and $B$ does not depend on $a$, although they can depend on the same $\lambda$. (The $L$ subscript denotes that this is an expected value of measurements according to the local variable theory.)

$\mathbb{E}_L(a,b)$ matches the form of the terms in the CHSH inequality for every $a$ and $b$, so $\mathbb{E}_L$ satisfies the CHSH inequality. If the hidden local variable theory exists, then any experiment that measures our photon states with respect to different axes $a, a',b,b'$ sufficiently many times should have $|\mathbb{E}_Q(ab + a'b + b'a - a'b')| = |\mathbb{E}_L(ab + a'b + b'a - a'b')| \leq 2$.

According to quantum mechanics, the expected value of measurements of the two particle state $\ket{ \psi }$ in the $a \otimes b$ axis is
$$
\mathbb{E}_Q(a,b) = \bra{ \psi } (\vec{\sigma}\cdot\vec{a} \otimes \vec{\sigma} \cdot \vec{b}) \ket{ \psi }.
$$ Define
$$
2B = \vec{\sigma}\cdot\vec{a} \otimes \vec{\sigma} \cdot \vec{b} + \vec{\sigma}\cdot\vec{a} \otimes \vec{\sigma} \cdot \vec{b}' + \vec{\sigma}\cdot\vec{a}' \otimes \vec{\sigma} \cdot \vec{b} + \vec{\sigma}\cdot\vec{a}' \otimes \vec{\sigma} \cdot \vec{b}'
$$
as the operator of interest. There exist four states ("Bell states") that violate the CHSH to a maximum extent:
$$
\ket{ \Phi^\pm } = \frac{1}{\sqrt{ 2 }} (\ket{ \uparrow\uparrow } \pm \ket{ \downarrow\downarrow }), \ket{ \Psi^\pm } = \frac{1}{\sqrt{ 2 }}(\ket{ \uparrow\downarrow }\pm\ket{ \downarrow\uparrow }).
$$
For $\ket{ \psi }$ equal to each of these four states, 
$$
\max_{a,a',b,b'} |\bra{ \psi } 2B \ket{ \psi }| = 2\sqrt{ 2 } \nleq 2.
$$
This is the "Tsirelson bound".

This inequality can be formulated as a two-player game:
A referee gives two bits $x$ and $y$ separately to Alice and Bob, without the other's knowledge. Alice and Bob have to each produce a bit respectively ($a$ and $b$) without communication. They win the game if
$$
x \times y = a \oplus b.
$$
($\times = \textrm{AND}$)
There are 8 winning cases out of 16 total possibilities for $x,y,a,b$. A random strategy would give a winning probability of 50%.
In classical theory, without proof, the maximum winning probability is 3/4: If Alice and Bob always output $a=b=0$, there are three out of four cases where $x \times y = 0$. 
In quantum mechanics, the maximum winning probability is $\frac{1}{2} + \frac{\sqrt{ 2 }}{4} \approx 0.8535$. If $x$ and $y$ are taken to represent two different measurement settings, and $a$ and $b$ are taken to represent two different outcomes, then the probability of winning is related to $B$ as
$$
P_{win} = \frac{1}{2} + \frac{\langle B \rangle}{4},
$$
maximized by the CHSH inequality violation.

This game is related to a so-called Popescu-Rohrlich box. See S. Popescu, Nonlocality beyond quantum mechanics (2014).