---
tags:
  - Probability-Theory
  - private
---
Question 1: Unfair coin (Easy)

You have a fair coin (one side heads, one side tails) and an unfair coin (both
sides tails). You pick one at random, flip it 5 times, and observe that it comes
up as tails all five times. What is the chance that you are flipping the unfair
coin?
Answer: $\frac{32}{33}$

Question 2: Minimum of Uniform Random Variables (Medium)

i) Let $X_i \sim$ Uniform(0, 1), $i = 1,2$, be two independent random variables.
What is the expected value of the minimum of $X_1$ and $X_2$?
Answer: $\frac{1}{3}$

ii) Now consider N independent random variables $X_i \sim$ Uniform(0,1), $i = 1,\cdots,N$.
What is the expected value of the minimum of all $X_i$?
Answer: $\frac{1}{N+1}$

iii) Write code to verify that this is the expected value of $\min{X_i}$. For
the cases of $N=2,3,5,10,50$, generate $N$ random numbers from a uniform
distribution, find their minimum value, repeat $10000$ times, and average the
minimum values. Plot these results together with your solution to part ii.

Question 3: 