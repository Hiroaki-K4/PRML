# Neural network


## Tanh sigmoid function
Tanh sigmoid function is sometimes used as activation function of machine learning. It is one of hyperbolic functions.

$$
tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

### Hyperbolic functions
In mathematics, hyperbolic functions are analogues of the ordinary trigonometric functions, but defined using the hyperbola rather than the circle. Just as the points $(cos t, sin t)$ form a circle with a unit radius, the points $(cosh t, sinh t)$ form the right half of the unit hyperbola. Also, similarly to how the derivatives of $sin(t)$ and $cos(t)$ are $cos(t)$ and $–sin(t)$ respectively, the derivatives of $sinh(t)$ and $cosh(t)$ are $cosh(t)$ and $+sinh(t)$ respectively.

$$
sinh(x)=\frac{e^x-e^{-x}}{2}, \quad cosh(x)=\frac{e^x+e^{-x}}{2}, \quad tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

You can draw hyperbolic functions by running following command.

```bash
python3 draw_hyperbolic_functions.py
```

<img src="images/hyperbolic_functions.png" width='600'>

### Derivatives
Derivatives of hyperbolic functions are as follows.

$$
\begin{align*}
\frac{d}{dx}sinhx&=\frac{e^x+e^{-x}}{2}=cosh(x) \\
\frac{d}{dx}coshx&=\frac{e^x-e^{-x}}{2}=sinh(x) \\
\frac{d}{dx}tanhx&=\frac{d}{dx}\frac{sinh(x)}{cosh(x)}=\frac{cosh^2(x)-sinh^2(x)}{cosh^2(x)}=1-tanh^2(x) \\
\end{align*}
$$

### Comparison of tanh sigmod and sigmoid
Both tanh function and sigmoid function are S-shaped curve function, but while tanh function is a symmetrical S-shaped curve at the origin $(0, 0)$, sigmoid function is a symmetrical S-shaped curve at $(x,y)=(0,0.5)$. Furthermore, it is known that it is desirable for activation function to be symmetrical at the origin.

You can compare tanh and sigmoid function by running following command.

```bash
python3 compare_tanh_sigmoid_and_sigmoid.py
```

<img src="images/sigmoid_tanh.png" width='600'>

<br></br>

# References
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)
