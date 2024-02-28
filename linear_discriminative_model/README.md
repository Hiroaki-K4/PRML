# Linear discriminative model
In this section, we describe the model solve classification problems. The purpose of classification is to assign a certain input vector $x$ to one of $K$ discrete classes $C_k$. Input space is separated to **decision region**. The boundary of this desicision region is called **decision boundary** or **decision surface**.

<br></br>

# Fisher's linear discriminant
We will introduce **Fisher's linear discriminant**, one of the linear discriminant models. We think about 2 class version. In this case, we get the D dimentional input vector and project to 1 dimentional by the following equation.

$$
y=w^\intercal x \tag{1}
$$

We set a cetertain threshold about $y$. If $y>-w_0$, then let y be class $C_1$, otherwise let be class $C_2$. Typically, the projection of 1 dimentional occurs a significant loss of information, so separated classes in the original D dimentional space may overlap greatly in the 1 dimentional space. However, we can choose the projection maximizes separation between classes by arranging elements of weight vector $w$. We think about 2 class problem has there are $N_1$ points of class $C_1$ and $N_2$ points of class $C_2$. The mean vectors of 2 classes are as follows.

$$
m_1=\frac{1}{N_1}\sum_{n\in C_1}x_n, \quad m_2=\frac{1}{N_2}\sum_{n\in C_2}x_n \tag{2}
$$

The most easy method to measure the class separation degree when projected onto $w$ is to check the separation degree between means of classes. It means that choosing $w$ maximizes following eqaution.

$$
m_2-m_1=w^\intercal(m_2-m_1) \tag{3}
$$

$$
m_k=w^\intercal m_k \tag{4}
$$

Here, $m_k$ is the mean of projected data from class $C_k$. However, this equation can be made to any large value by increasing $w$ value. To avoid this problem, we add constraints that $w$ has the unit length. We can get $w\propto (m_2-m_1)$ by using Lagrangian multipliers for solving constrained maximization problems. However, this approach has a problem there are still many areas of overlap. This is because covariance of off-diagonal elements of the class distribution is strong. In Fisher's approach, maximize separation degree between projected class means and minimize within-class variances at the same time. As a result, we can minimize the overlap between classes.

Eq(1) projects labeled datasets in the input space $x$ to labeled datasets in the input space $y$. That is, Within-class variance of the data projected from class $C_k$ is given as follows.

$$
s_k^2=\sum_{n\in C_k}(y_n-m_k)^2 \tag{5}
$$

Here, $y_n=w^\intercal x_n$ holds true. We define the total within-class variance for all datasets as $s_1^2+s_2^2$. Fisher's discrimination criteria is defined as the ratio between the within-class variance and the between-class variance.

$$
J(w)=\frac{(m_2-m_1)^2}{s_1^2+s_2^2} \tag{6}
$$

From Eq(1), Eq(4) and Eq(5), Fisher's discriminant criteria can be rewritten as follows.

$$
J(w)=\frac{w^\intercal S_B w}{w^\intercal S_W w} \tag{7}
$$

Here, $S_B$ is the **between-class covariacne matrix** and given as follows.

$$
S_B=(m_2-m_1)(m_2-m_1)^\intercal \tag{8}
$$

Thus, $S_W$ is the total **within-class covariance matrix** and given as follows.

$$
S_W=\sum_{n\in C_1}(x_n - m_1)(x_n - m_1)^\intercal + \sum_{n\in C_2}(x_n - m_2)(x_n - m_2)^\intercal \tag{9}
$$

By differentiating Eq(7) with respect to $w$, we find $J(w)$ is maximized when the following equation is satisfied.

$$
(w^\intercal S_B w)S_W w = (w^\intercal S_W w)S_B w \tag{10}
$$

From Eq(8), we find that $S_B w$ is always a vector with the same direction as $(m_2-m_1)$. Thus, only the direction of $w$ is important and there is no need to consider the length of that. We can ignore the scalar factors $(w^\intercal S_B w)$ and $(w^\intercal S_W w)$. We can get the relationship between $w$ and the difference of class means by multiplying both sides by $S_W^{-1}$.

$$
w\propto S_W^{-1}(m_2-m_1) \tag{11}
$$

Eq(11) is known as **Fisher's linear discriminant**.

We can try the Fishers linear discriminant by running the following command. The graph below shows that accurate linear identification is possible.

```bash
python3 fishers_linear_discriminant.py
```

<img src="images/fisher.png" width='800'>

<br></br>

# Reference
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
