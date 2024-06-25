# Kernel method
Many parametric linear models can be rewritten in the form of an equivalent dual representation, prediction is also made by using kernel function defined with training data as the center. In a model based on a mapping $\phi(x)$ to a predefined nonlinear feature space, kernel function is given by following relationship.

$$
k(x,x\prime)= \phi(x)^\intercal \phi(x\prime) \tag{1}
$$

You can plot kernel functions by running following command.

```bash
python3 draw_kernel_function.py
```

Bottom row shows kernel function $k(x,x\prime)$. From the oeft, the plots show a polynomial function, a Guassian distribution, and a logistic sigmoid function.

<img src="images/kernel_function.png" width='1200'>

<br></br>

## Nadaraya-Watson model
Kernel regression model can be motivated in terms of kernel density estimation. Let the training set be $(x_n, t_n)$, and to estimate the joint distribution $p(x,t)$, we use the Parzen windows density estimation as follows.

$$
p(x,t)=\frac{1}{N}\sum_{n=1}^N f(x-x_n, t-t_n) \tag{2}
$$

$f(x,t)$ is the density function that composes $p(x,t)$, one centered on each data point.
To calculate regression function $y(x)$, we need to calculate the conditional expectation of target variable conditioned on input variables, which is given by following equation.

$$
\begin{align*}
y(x)&=E[t|x]=\int_{-\infty}^{\infty} tp(t|x)dt \\
&=\frac{\int tp(x,t)dt}{\int p(x,t)dt} \\
&=\frac{\sum_{n} \int tp(x-x_n, t-t_n)dt}{\sum_{m} \int p(x-x_m, t-t_m)dt} \tag{3}
\end{align*}
$$

We can get the following equation by substituting variables.

$$
\begin{align*}
y(x)&=\frac{\sum_{n} g(x-x_n)t_n}{\sum_{m} g(x-x_m)} \\
&=\sum_{n} k(x, x_n)t_n \tag{4}
\end{align*}
$$

Here, $n,m=1,...,N$ and kernel function $k(x, x_n)$ is given by following equation.

$$
k(x, x_n)=\frac{g(x-x_n)}{\sum_{m} g(x-x_m)} \tag{5}
$$

And $g(x)$ is defined as follows.

$$
g(x)=\int_{-\infty}^{\infty} f(x,t)dt \tag{6}
$$

The result of Eq(3) is called as **Nadaraya-Watson model** or **Kernel regression**. If you use local kernel function, data points $x_n$ closer to data $x$ is are given more weight. Note that $k(x,x_n)$ is satisfied with sum constraint.

$$
\sum_{n=1}^N k(x,x_n)=1 \tag{7}
$$

You can try Nadaraya-Watson model by running following command.

```bash
python3 nadaraya_watson_model.py
```

Below graph shows Nadaraya-Watson model when isotropic Gaussin kernel is used for the trigonometric data. Each input data is the center of the isotropic Gaussian kernel.

<img src="images/nadaraya_watson.png" width='600'>

<br></br>

# References
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
