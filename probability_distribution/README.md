# Probability distribution
Probability theory is important in solving pattern recognition problems. Here we summarize various probability distributions and their characteristics.

<br></br>

## Binomial distribution
The distribution of the number $m$ of observations for a given data set of size $N$, where $x=1$, is called the binomial distribution. To calculate the normalization factor, we consider the sum over all possible cases where $N$ coin tosses produce the table $m$ times, the binomial distribution is as follows. Denote the probability that x=1 by $\mu$.

$$
Bin(m|N,\mu)=\frac{N!}{(N-m)!m!}\mu ^m (1-\mu) ^{N-m}
$$

The mean and variance are as follows.

$$
E[m]=\sum_{m=0}^N mBin(m|N,\mu)=N\mu \\
var[m]=\sum_{m=0}^N (m-E[m])^2 Bin(m|N,\mu) = N\mu(1-\mu)
$$

You can draw the binomial distribution by running follow command.

```bash
python3 draw_binomial_distribution.py
```

The graph shows the results for varying values of $\mu$.

<img src='images/binomial_dist.gif' width='600'>

<br></br>

## Beta distribution

```bash
python3 draw_beta_distribution.py
```

<img src='images/beta_dist.gif' width='600'>

<br></br>

## References
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Python Plotting Binomial Distributions](https://www.anarchive-beta.com/entry/2022/01/14/073000)
