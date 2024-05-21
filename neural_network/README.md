# Neural network


## Tanh sigmoid function

### Hyperbolic functions

$$
sinh(x)=\frac{e^x-e^{-x}}{2}, \quad cosh(x)=\frac{e^x+e^{-x}}{2}, \quad tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

### Derivatives

$$
\begin{align*}
\frac{d}{dx}sinhx&=\frac{e^x+e^{-x}}{2}=cosh(x) \\
\frac{d}{dx}coshx&=\frac{e^x-e^{-x}}{2}=sinh(x) \\
\frac{d}{dx}tanhx&=\frac{d}{dx}\frac{sinh(x)}{cosh(x)}=\frac{cosh(x)^2-sinh(x)^2}{cosh(x)^2}=1-tanh(x)^2 \\
\end{align*}
$$

### Comparison of tanh sigmod and sigmoid

# Reference
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions)
