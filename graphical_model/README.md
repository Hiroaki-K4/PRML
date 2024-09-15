# Graphical model
It is very useful to use the probabilistic graphic model, which are graphical representations of probability distributions, in the analysis. The graph consists of a collection of **vertexes** connected by **edges**. In the probabilistic graphic model, each vertex represents the stochastic variable and each edge represents stochastic relationships between these parameters.
Directed graph whose edges have a specific direction is useful to represent the causal relationship between stochastic variables. Undirected graph is useful to represent the loose bondage relationship between stochastic variables.

<br></br>

# Image denoising

```bash
python3 remove_noise_using_graphical_model.py
```

<img src="images/test_data.png" width='500'>

<img src="images/noise.png" width='500'>

<img src="images/denoise.png" width='500'>

<br></br>

# Reference
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
