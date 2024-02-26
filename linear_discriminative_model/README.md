# Linear discriminative model
In this section, we describe the model solve classification problems. The purpose of classification is to assign a certain input vector $x$ to one of $K$ discrete classes $C_k$. Input space is separated to **decision region**. The boundary of this desicision region is called **decision boundary** or **decision surface**.

<br></br>

# Fisher's linear discriminant
We will introduce **Fisher's linear discriminant**, one of the linear discriminant models. We think about 2 class version. In this case, we get the D-dimentional input vector and project to 1 dimentional by the following equation.

$$
y=w^\intercal x \tag{1}
$$

We set a cetertain threshold about $y$. If $y>-w_0$, then let y be class $C_1$, otherwise let be class $C_2$. Typically, projection of 1 dimentional occurs a significant loss of information, so separated classes in the original D dimentional space may overlap greatly in 1 the dimentional space. However, we can choose the projection maximizes separation between classes by arranging elements of weight vector $w$. We think about 2 class problem has there are $N_1$ points of class $C_1$ and $N_2$ points of class $C_2$. The mean vectors of 2 classes are as follows.

$$
m_1=\frac{1}{N_1}\sum_{n\in C_1}x_n, \quad m_2=\frac{1}{N_2}\sum_{n\in C_2}x_n \tag{2}
$$

The most easy method to measure the class separation degree when projected onto $w$ is to check the separation degree between means of classes. It means that choosing $w$ maximizes following eqaution.

$$
m_2-m_1=w^\intercal(m_2-m_1) \tag{3}
$$

Here, $m_k$ is the mean of projected data from class $C_k$. However, this equation can be made to any large value by increasing $w$ value. To avoid this problem, we add constraints that $w$ is unit length. We can get $w\propto (m_2-m_1)$ by using Lagrangian multipliers for solving constrained maximization problems. However, this approach has a problem there are still many areas of overlap. This is because covariance of off-diagonal elements of the class distribution is strong. In Fisher's approach, maximize separation degree between projected class means and minimize intraclass variances at the same time. As a result, we can minimize the overlap between classes.

<br></br>

# Reference
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
