# Probability distribution
Probability theory is important in solving pattern recognition problems. Here we summarize various probability distributions and their characteristics.

<br></br>

# Binary variable
Let us first consider the case where there is a single binary random variable $x\in{0,1}$. For example, if $x=1$ represents a table and $x=0$ represents a flip, the result of the coin toss is represented by this binary random variable $x$. The probability that $x=1$ is represented by the parameter $\mu$, where $\mu$ is the probability that the coin is skewed.

$$
p(x=1|\mu)=\mu \tag{1}
$$

However, $0\leq\mu\leq1$, from which $p(x=0|\mu)=1-\mu$. Therefore, the probability distribution on $x$ is as follows.

$$
Bern(x|\mu)=\mu^x (1-\mu)^{1-x} \tag{2}
$$

This is known as the Bernoulli distribution. The mean and variance of this distribution are as follows

$$
\begin{align*}
E[x]&=\mu \\
var[x]&=\mu(1-\mu) \tag{3}
\end{align*}
$$

## Binomial distribution
The distribution of the number $m$ of observations for a given data set of size $N$, where $x=1$, is called the binomial distribution. To calculate the normalization factor, we consider the sum over all possible cases where $N$ coin tosses produce the table $m$ times, the binomial distribution is as follows. Denote the probability that x=1 by $\mu$.

$$
Bin(m|N,\mu)=\frac{N!}{(N-m)!m!}\mu ^m (1-\mu) ^{N-m} \tag{4}
$$

The mean and variance are as follows.

$$
\begin{align*}
E[m]&=\sum_{m=0}^N mBin(m|N,\mu)=N\mu \\
var[m]&=\sum_{m=0}^N (m-E[m])^2 Bin(m|N,\mu) = N\mu(1-\mu) \tag{5}
\end{align*}
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
Beta(\mu|a,b)=\frac{\gamma(a+b)}{\gamma(a)\gamma(b)}\mu^{a-1}(1-\mu)^{b-1} \tag{6}
$$

$\gamma$ is the gamma function, and the coefficients of Eq(6) ensure that the beta distribution is normalized, and the following equation holds.

$$
\int_{1}^{0}Beta(\mu|a,b) \mathrm{d}\mu=1 \tag{7}
$$

The mean and variance of the beta distribution are given by

$$
\begin{align*}
E[\mu]&=\frac{a}{a+b} \\
var[\mu]&=\frac{ab}{(a+b)^2(a+b+1)} \tag{8}
\end{align*}
$$

The parameters $a$ and $b$ are often referred to as hyperparameters because they determine the distribution of the parameter $\mu$. A graph can be created by running the following code when the hyperparameters are set to various values.

```bash
python3 draw_beta_distribution.py
```

<img src='images/beta_dist.gif' width='600'>

As the number of observations increases, the peaks of the distribution become sharper. This is also seen in the variance of the beta distribution, Eq(8), where the variance is zero if $a\rightarrow\infty$ or $b\rightarrow\infty$.

<br></br>

# Multivalued variable
Binary variables can be used to describe quantities that take one of two possible values. However, it is often necessary to deal with discrete variables that take one of $K$ possible states that are mutually exclusive. Here, we use the 1-of-K encoding method. In this method, the variable is represented by a K-dimensional vector $x$ such that one of the elements $x_k$ is $1$ and all the rest are $0$. For example, there is a variable that can be in $K=6$ different states, and the observed value $x$ of this variable that happens to be in the state $x_3=1$ is represented as follows.

$$
x=(0,0,1,0,0,0)^\intercal \tag{9}
$$

Such a vector satisfies $\sum_{k=1}^K x_k=1$. If we denote the probability that $x_k=1$ by the parameter $\mu_k$, the distribution of $x$ is given by

$$
p(x|\mu)=\prod_{k=1}^K\mu_k^{x_k} \tag{10}
$$

In order for the parameter $\mu_k$ to represent probability, with $\mu=(\mu_1,...,\mu_K)^\intercal$, the parameter $\mu_k$ must satisfy $\mu_k\geq0$ and $\sum_a\mu_k=1$.

## Dirichlet distribution
Under the given conditions of the parameter $\mu$ and the total number of observations $N$, we consider the simultaneous probability of $m1,...,m_K$. This takes the following form and is called a multinomial distribution.

$$
Multi(m_1,m_2,...,m_K|\mu,N)=\frac{N!}{m_1!m_2!...m_K!}\prod_{k=1}^K\mu_k^{x_k} \tag{11}
$$

Here, as can be seen from the multinomial distribution form of Eq(11), its conjugate distribution is as follows.

$$
p(\mu|\alpha)\propto\prod_{k=1}^K\mu_k^{\alpha_k-1} \tag{12}
$$

$0\leq\mu_k\leq1$ and $\sum_k\mu_k=1$. where $\alpha$ denotes $(\alpha_1,...,\alpha_K)^\intercal$.

Normalizing this distribution yields the following

$$
Dir(\mu|\alpha)=\frac{\gamma(\alpha_0)}{\gamma(\alpha_1)...\gamma(\alpha_K)}\prod_{k=1}^K\mu_k^{\alpha_k-1} \tag{13}
$$

This is called as Dirichlet distribution. $\gamma(x)$ is the gamma function and $\alpha_0$ is as follows

$$
\alpha_0=\sum_{k=1}^K \alpha_k \tag{14}
$$

You can draw the Dirichlet distribution by running follow command.

```bash
python3 draw_dirichlet_distribution.py
```

The drawing shows what happens when the parameter $\alpha_1$ is varied.

<img src='images/dirichlet_dist.gif' width='600'>

<br></br>

## Gaussian distribution

You can draw the Gaussian distribution by running follow command.

```bash
python3 draw_gaussian_distribution.py
```

<img src='images/gaussian_dist.gif' width='600'>

<br></br>

## References
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Python Plotting Binomial Distributions](https://www.anarchive-beta.com/entry/2022/01/14/073000)
- [Python Plotting Dirichlet Distributions](https://www.anarchive-beta.com/entry/2022/10/19/120500)
