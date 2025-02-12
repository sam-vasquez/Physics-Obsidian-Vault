---
tags:
  - Math
---
 For integers $a,b$, the smallest positive linear combination of $a$ and $b$ equals $\gcd (a,b)$.

**Proof**
Suppose $d$ is the smallest positive linear combination of $a$ and $b$.
Write it as $d = \alpha a + \beta b$.
For some $q$ we can write $a = qd + r$ with $0 \leq r < d$.
Then we have
$$
r = a - qd = a - q(\alpha a + \beta b) = (1 - q \alpha)a - (q\beta)b.
$$
Therefore $r$ is also a linear combination of $a$ and $b$. But $r$ is smaller than $d$, and we said that $d$ is the smallest positive combination. So $r$ must be 0. Therefore $d | a$.
We can repeat this for $b$ to get that $d|b$. Therefore $d$ is a common divisor of $a$ and $b$.
Suppose that there were some other common divisor $c$. Then $c | (\alpha a + \beta b)$, so $c|d$, so $c\leq d$. Therefore $d$ is the greatest common divisor.
Note: This argument extends to sets of any size.


References:
https://www.math.umd.edu/~immortal/MATH406/lecturenotes/ch3-3.pdf