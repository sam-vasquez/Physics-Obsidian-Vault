---
tags:
  - Math
---
 If $S \subset \mathbb{Z}^+$ has $\gcd(S) = g_S$, then there is some integer $m_S$ such that for all $m \geq m_S$ the product $mg_S$ can be written as a linear combination of elements of $S$ with non-negative integer coefficients.

Note: The largest integer that cannot be written as a non-negative integer combination of elements of $S$ is usually called the Frobenius number.

**Proof (Partial)**
[[GCD of Set is Integer Combination of Set Elements|It is known]] that the gcd of a set $S$ can always be written as a linear combination of elements of $S$ with integer coefficients. Naturally, so can multiples of the gcd.
It is also easy to verify that every linear integer combination of elements of $S$ is a multiple of the gcd. 
Let $S = \{a,b\} \subset \mathbb{Z}^+$ have $\gcd(S) = g$. Given $k > 0$, we can write $kg = \alpha a + \beta b$ for some integers $\alpha,\beta$. Assume without loss of generality that $\alpha < 0$.
Note that $\alpha$ and $\beta$ are not unique. Transformations $\alpha' = \alpha + \gamma b$ and $\beta' = \beta - \gamma a$ define a new representation $kg = \alpha'a + \beta'b$ for every integer $\gamma$.
We can choose $\gamma'$ such that $0 \leq \alpha' < b$. Under what conditions on $k$ is $\beta' \geq 0$?
Because $\alpha' < b$, $\alpha' \leq b-1$ and 
$$
mg = \alpha'a + \beta'b \leq (b-1)a + \beta'b.
$$
Isolate $\beta'$ to get that
$$
\beta' \geq \frac{mg - (b-1)a}{b}.
$$
It is desired that $\beta' \geq 0$, or equivalently that $\beta' > -1$. Therefore it is demanded that $mg - (b-1)a > -b$. This comes out to $mg > ab - a -b$. 
Thus, for all larger integers $m$, we can always choose a linear combination of $a$ and $b$ that has positive integer coefficients.
It should be believable that this argument extends to all finite subsets of $\mathbb{Z}^+$ by induction on the size of the set. I don't care about proving this. 