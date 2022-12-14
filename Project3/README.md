# Project 3

## Dependencies

### Python

- [JAX](https://jax.readthedocs.io/en/latest/index.html) for automatic differentiation and numerical linear algebra.
- [NumPy](https://pypi.org/project/numpy/) for numerical linear algebra.
- [Matplotlib](https://pypi.org/project/matplotlib/) for plotting.

## Run the Code

**Note.** This code has only been tested on Ubuntu Linux. 

To train the neural network for solving the one-dimensional heat equation on the unit interval, run

```
$ python3 heat_eq_mlp.py
```

To train the neural network for obtaining and eigenvalue/eigenvector pair of a symmetric matrix, run

```
$ python3 symdiag_mlp.py
```

To generate the plots of the bias-variance tradeoff for polynomial fitting using OLS, Ridge and Lasso regression, run

```
$ python3 bias_variance_tradeoff.py
```