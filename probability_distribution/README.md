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
The Gaussian distribution, also called the normal distribution, is widely used as a model for the distribution of continuous variables.
If there is only one variable, it can be written as follows. $\mu$ is the averange and $\sigma^2$ is the variance.

$$
\mathcal{N}(x|\mu,\sigma^2)=\frac{1}{(2\pi\sigma^2)^{1/2}}e^{-\frac{1}{2\sigma^2}(x-\mu)^2} \tag{15}
$$

You can draw the Gaussian distribution by running follow command. Under a Gaussian distribution, the probability density of the random numbers created gradually follows a Gaussian distribution.

```bash
python3 draw_gaussian_distribution.py
```

<img src='images/gaussian_dist.gif' width='600'>

<br></br>

### Multivariate Gaussian distribution

The multivariate Gaussian distribution for a vector $x$ in dimension $D$ is as follows.

$$
\mathcal{N}(x|\mu,\sigma)=\frac{1}{(2\pi)^{D/2}} \frac{1}{|\sigma|^{1/2}} e^{-\frac{1}{2}(x-\mu)^\intercal\sigma^{-1}(x-\mu)} \tag{16}
$$

$\mu$ is the average in dimention $D$, $\sigma$ is the covariance matrix in dimention $D \times D$ and $|\sigma|$ is the determinant of $\sigma$

The Gaussian distribution depends on $x$ through the following quadratic form, which appears in the exponential part.

$$
\triangle^2=(x-\mu)^\intercal\sigma^{-1}(x-\mu) \tag{17}
$$

This quantity $\triangle$ is called the **Mahalanobis distance** from $\mu$ to $x$. This would be the Euclidean distance if $\sigma$ were a unit matrix. The density of the Gaussian distribution is constant on the surface where the value of this quadratic form is constant in $x$-space.


You can draw the multivariate Gaussian distribution by running follow command.

```bash
python3 draw_multivariate_gaussian_distribution.py
```

<img src='images/multi_gaussian.png' width='600'>

<img src='images/multi_gaussian_top.png' width='600'>

<br></br>

## Conditional Gaussian distribution
An important property of the multivariate Gaussian distribution is that if the simultaneous distribution of two sets of variables follows a Gaussian distribution, then given one set of variables, the conditional distribution of the other set will also be Gaussian.
Let $x$ be a D-dimensional vector following a Gaussian distribution $\mathcal{N}(x|\mu,\sigma)$, and partition this vector $x$ into two mutually prime subsets $x_a$ and $x_b$.

$$
x=\begin{pmatrix}
x_a \\
x_b \\
\end{pmatrix} \tag{18}
$$

The corresponding partitioning of the mean vector $\mu$ is also defined.

$$
\mu=\begin{pmatrix}
\mu_a \\
\mu_b \\
\end{pmatrix} \tag{19}
$$

And the covariance matrix $\sigma$ is given in the same way.

$$
\sigma=\begin{pmatrix}
\sigma_{aa} & \sigma_{ab} \\
\sigma_{ba} & \sigma_{bb} \\
\end{pmatrix} \tag{20}
$$

By the way, it is often more convenient to consider the inverse of the covariance.

$$
A\equiv\sigma^{-1} \tag{21}
$$

This is called the precision matrix. The partitioned precision matrix is as follows.

$$
A=\begin{pmatrix}
A_{aa} & A_{ab} \\
A_{ba} & A_{bb} \\
\end{pmatrix} \tag{22}
$$

Let's start by finding an expression for the conditional distribution $p(x_a|x_b)$: as long as we fix $x_b$ at the observed value and normalize the obtained expression to be a legitimate probability on $x_a$, we can find the conditional distribution using the simultaneous distribution $p(x)=p(x_a, x_b)$ from the multiplication theorem of probability. Instead of doing this normalization explicitly, the solution can be efficiently obtained by considering the quadratic form of the exponential part of the Gaussian distribution in Eq(17) and finding the normalization coefficient at the end of the calculation.

We can get below equation by using Eq(18), Eq(19) and Eq(22).

$$
-\frac{1}{2}(x-\mu)^\intercal\sigma^{-1}(x-\mu)=-\frac{1}{2}(x_a-\mu_a)^\intercal A_{aa}(x_a-\mu_a)-\frac{1}{2}(x_a-\mu_a)^\intercal A_{ab}(x_b-\mu_b) \\
-\frac{1}{2}(x_b-\mu_b)^\intercal A_{ba}(x_a-\mu_a)-\frac{1}{2}(x_b-\mu_b)^\intercal A_{bb}(x_b-\mu_b) \tag{23}
$$

If we view this equation as a function of $x_a$, we see that the corresponding conditional distribution $p(x_a|x_b)$ is also Gaussian, since it is also quadratic in form. Since the characteristics of a Gaussian distribution are completely determined by the mean and covariance matrices, we use Eq(23) to obtain the mean and covariance expressions for $p(x_a|x_b)$.

The exponential part of the general Gaussian distribution $\mathcal{N}(x|\mu,\sigma)$ can be written in a straightforward manner by noting that it can be written as

$$
-\frac{1}{2}(x-\mu)^\intercal\sigma^{-1}(x-\mu)=-\frac{1}{2}x^\intercal \sigma^{-1} x + x^\intercal \sigma^{-1}\mu+const \tag{24}
$$


You can draw the conditional Gaussian distribution by running follow command.

```bash
python3 draw_conditional_gaussian_distribution.py
```

<img src='images/conditional_gaussian_dist.png' width='600'>

<br></br>

## References
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Python Plotting Binomial Distributions](https://www.anarchive-beta.com/entry/2022/01/14/073000)
- [Python Plotting Dirichlet Distributions](https://www.anarchive-beta.com/entry/2022/10/19/120500)
- [Python Plotting Gaussian Distributions](https://www.anarchive-beta.com/entry/2022/01/31/180000)
- [The world's easiest way to use np.meshgrid (mesh grid)](https://disassemble-channel.com/np-meshgrid/)
