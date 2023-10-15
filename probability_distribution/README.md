# Probability distribution
Probability theory is important in solving pattern recognition problems. Here we summarize various probability distributions and their characteristics.

<br></br>

# Binary variable

## Binomial distribution
The distribution of the number $m$ of observations for a given data set of size $N$, where $x=1$, is called the binomial distribution. To calculate the normalization factor, we consider the sum over all possible cases where $N$ coin tosses produce the table $m$ times, the binomial distribution is as follows. Denote the probability that x=1 by $\mu$.

$$
Bin(m|N,\mu)=\frac{N!}{(N-m)!m!}\mu ^m (1-\mu) ^{N-m} \tag{1}
$$

The mean and variance are as follows.

$$
E[m]=\sum_{m=0}^N mBin(m|N,\mu)=N\mu \\
var[m]=\sum_{m=0}^N (m-E[m])^2 Bin(m|N,\mu) = N\mu(1-\mu) \tag{2}
$$

You can draw the binomial distribution by running follow command.

```bash
python3 draw_binomial_distribution.py
```

The graph shows the results for varying values of $\mu$.

<img src='images/binomial_dist.gif' width='600'>

<br></br>

## Beta distribution
The maximum likelihood estimate of the parameter $\mu$ of the binomial distribution is the fraction of observations in the data set for which $x=1$. But this method can be very over fitting when the data set is small. Therefore, it is necessary to introduce a prior distribution of the parameter $\mu$ in order to treat this problem in a Bayesian manner. The following beta distribution is chosen as the prior distribution.

$$
Beta(\mu|a,b)=\frac{\gamma(a+b)}{\gamma(a)\gamma(b)}\mu^{a-1}(1-\mu)^{b-1} \tag{3}
$$

$\gamma$ is the gamma function, and the coefficients of Eq(3) ensure that the beta distribution is normalized, and the following equation holds.

$$
\int_{1}^{0}Beta(\mu|a,b) \mathrm{d}\mu=1 \tag{4}
$$

The mean and variance of the beta distribution are given by

$$
E[\mu]=\frac{a}{a+b} \\
var[\mu]=\frac{ab}{(a+b)^2(a+b+1)} \tag{5}
$$

The parameters $a$ and $b$ are often referred to as hyperparameters because they determine the distribution of the parameter $\mu$. A graph can be created by running the following code when the hyperparameters are set to various values.

```bash
python3 draw_beta_distribution.py
```

<img src='images/beta_dist.gif' width='600'>

As the number of observations increases, the peaks of the distribution become sharper. This is also seen in the variance of the beta distribution, Eq(5), where the variance is zero if $a\rightarrow\infty$ or $b\rightarrow\infty$.

<br></br>

## Dirichlet distribution

<img src='images/dirichlet_dist.gif' width='600'>

<br></br>

## References
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Python Plotting Binomial Distributions](https://www.anarchive-beta.com/entry/2022/01/14/073000)
