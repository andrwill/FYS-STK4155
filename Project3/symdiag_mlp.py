import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import jit, grad, vmap, random

from pdemlp import random_layers, sigmoid

def single_predict(layers, x0):
    z = x0
    for w, b in layers[:-1]:
        z = sigmoid(jnp.dot(w, z) + b)

    w, b = layers[-1]
    x = jnp.dot(w, z) + b

    return x

predict = vmap(single_predict, in_axes=(None, 0))

def ode_loss(layers, A, x0):
    x = single_predict(layers, x0).T
    return jnp.linalg.norm((x.T@x)*A@x + (x.T@A@x)*x)

#grad_loss = grad(loss)

@jit
def loss(layers, A, x0, eps=1.0e-8):
    x = single_predict(layers, x0).T
    #x_norm = jnp.linalg.norm(x)
    if x_norm < eps:
        return 0.0
    #
    #x = x.at[:].set(x / x_norm)

    return -((x.T@A@x)/(x.T@x))**2


@jit
def update(loss, layers, A, x0, eta=0.001):
    grads = grad(loss)(layers, A, x0)
    return [(w-eta*dw, b-eta*db) for (w,b), (dw,db) in zip(layers, grads)]

if __name__ == '__main__':
    key = random.PRNGKey(0)

    n = 6

    A = random.normal(key, (n, n))
    A = 0.5*(A+A.T)

    exact_eigvals, exact_eigvects = jnp.linalg.eig(A)
    exact_eigvals, exact_eigvects = exact_eigvals.real, exact_eigvects.real

    nn_width = n # = x0.size + t.size
    nn_depth = 2
    layer_sizes = nn_depth*[nn_width]
    layers = random_layers(key, layer_sizes)

    x0 = jnp.eye(n)[0]

    num_epochs = 100000
    epochs = range(num_epochs)
    preds = []
    losses = []
    for epoch in epochs:
        layers = update(layers, A, x0)
        preds.append(single_predict(layers, x0))
        losses.append(loss(layers, A, x0))

    plt.title('Losses during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, losses)
    plt.show()

    eigvals, eigvecs = jnp.linalg.eig(A)
    eigvals, eigvecs = eigvals.real, eigvecs.real

    print(eigvals)

    predicted_eigvecs = predict(layers, x0)

    eigvec = single_predict(layers, x0)
    if not jnp.allclose(eigvec, 0.0):
        eigvec /= jnp.linalg.norm(eigvec)
    eigval = eigvec.T@A@eigvec

    print(f'Estimated eigenvector x = {eigvec}')
    print(f'Estimated eigenvalue a = {eigval}')
    
    print(f'|Ax - ax|: {jnp.linalg.norm(A@eigvec - eigval*eigvec)}')
    print(f'Distance from a to nearest eigenvalue: {jnp.min(jnp.abs(eigval - eigvals))}')
