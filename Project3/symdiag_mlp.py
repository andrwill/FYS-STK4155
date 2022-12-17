import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import jit, grad, jacobian, vmap
from jax import random

from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController

from heat_eq_mlp import random_layers, sigmoid

def rayleigh_quotient(x, A):
    return x.T@A@x/(x.T@x)

def compute_max_eigval(A, num_epochs=1000, step_size=0.001):
    @jit
    def update(x, A, eta=0.001):
        return x + eta*grad(rayleigh_quotient)(x, A)

    key = random.PRNGKey(0)
    x = random.normal(key, (N,))
    x /= jnp.linalg.norm(x)
    for epoch in range(num_epochs):
        x = update(x, A, step_size)

    eigvec = x/jnp.linalg.norm(x)
    eigval = eigvec.T@A@eigvec

    return eigval, eigvec

def min_eigval(A, num_epochs=1000, step_size=0.001):
    @jit
    def update(x, A, eta=0.001):
        return x - eta*grad(rayleigh_quotient)(x, A)

    key = random.PRNGKey(0)
    x = random.normal(key, (N,))
    x /= jnp.linalg.norm(x)
    for epoch in range(num_epochs):
        x = update(x, A, step_size)

    eigvec = x/jnp.linalg.norm(x)
    eigval = eigvec.T@A@eigvec

    return eigval, eigvec

if __name__ == '__main__':
    key = random.PRNGKey(0)

    N = 6

    x0 = random.normal(key, (N,))

    A = random.normal(key, (N, N))
    eigvals = jnp.linalg.eigvalsh(A)
    max_eigval = jnp.max(eigvals)

    def rhs(x):
        return (x.T@x)*A@x - (x.T@A@x)*x

    vector_field = lambda t, x, args: rhs(x)
    term = ODETerm(vector_field)

    t0, t1, dt0 = 0.0, 10.0, 0.001

    ts = jnp.arange(t0, t1, dt0)
    solver = Tsit5()
    saveat = SaveAt(ts=ts)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    solution = diffeqsolve(
        term, 
        solver, 
        t0 = t0, t1 = t1, 
        dt0=dt0, 
        y0=x0,
        saveat=saveat,
        stepsize_controller = stepsize_controller    
    )

    ys = solution.ys

    @vmap
    def eigpair_loss(x):
        x /= jnp.linalg.norm(x)
        lmbda = x.T@A@x

        eigval_loss = abs(max_eigval- lmbda)
        eigvec_loss = jnp.linalg.norm(A@x - lmbda*x)

        return jnp.array([eigval_loss, eigvec_loss])

    @vmap
    def rayleigh_qoutient(x):
        return x.T@A@x/(x.T@x)

    eigpair_losses = eigpair_loss(ys)
    eigval_losses = eigpair_losses[:,0]
    eigvec_losses = eigpair_losses[:,1]
    plt.title('Eigenvalue Losses')
    plt.xlabel('$t$')
    plt.ylabel('$|\lambda_{\max} - \lambda|$')
    plt.ylim(0.0, 1.1*jnp.max(eigval_losses))
    plt.plot(ts, eigval_losses)
    plt.savefig('./figures/eigval_losses.pdf')
    plt.show()

    plt.title('Eigenvector Losses')
    plt.xlabel('$t$')
    plt.ylabel('$|\mathbf{A}\mathbf{x} - \lambda\mathbf{x}|$')
    plt.ylim(0.0, 1.1*jnp.max(eigvec_losses))
    plt.plot(ts, eigvec_losses)
    plt.savefig('./figures/eigvec_losses.pdf')
    plt.show()

    num_epochs = 1000
    lambda_rayleigh, x_rayleigh = compute_max_eigval(A, num_epochs)
    eigval_loss_rayleigh = abs(max_eigval - lambda_rayleigh)
    eigvec_loss_rayleigh = jnp.linalg.norm(A@x_rayleigh - lambda_rayleigh*x_rayleigh)

    print(f'|lambda_max - lambda| (Gradient Ascent ({num_epochs} epochs)): {eigval_loss_rayleigh}')
    print(f'|lambda_max - lambda| (Neural ODE): {eigval_losses[-1]}')
    print(f'|Ax - lambda*x| (Gradient Ascent ({num_epochs} epochs)): {eigvec_loss_rayleigh}')
    print(f'|Ax - lambda*x| (Neural ODE): {eigvec_losses[-1]}')
