import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import jit, grad, vmap, random

@jit
def sigmoid(x):
    return 1.0/(1.0 + jnp.exp(-x))

def random_layer(key, m, n, weight_scale=1.0e-2, bias_scale=1.0e-2):
    weight_key, bias_key = random.split(key)
    weights_shape = (m,n) if n != 1 else (m,) # Convert shape from (m,1) to (m,)
    random_weights = weight_scale*random.normal(weight_key, weights_shape)
    random_biases = bias_scale*random.normal(bias_key, (n,))
    
    return random_weights, random_biases

def random_layers(key, layer_sizes, weight_scale=1.0e-2, bias_scale=1.0e-2):
    keys = random.split(key, len(layer_sizes))
    layer_data = zip(keys, layer_sizes[:-1], layer_sizes[1:])
    
    return [ random_layer(k, m, n, weight_scale, bias_scale) for k,m,n in layer_data ]

def single_predict(layers, x, t):
    #z = jnp.array([x, t])

    num_repetitions = layers[0][0].shape[0] // 2
    
    z = jnp.vstack([x, t])
    z = jnp.repeat(z, num_repetitions)
    for w, b in layers[:-1]:
        z = sigmoid(jnp.dot(w, z) + b)

    w, b = layers[-1]
    nn = jnp.dot(w, z) + b
    return ((1.0 - t)*jnp.sin(jnp.pi*x) + (1.0 - x)*x*t*nn)[0] # Scalar output

du_dt = grad(single_predict, 2)
d2u_dx2 = grad(grad(single_predict, 1), 1)

predict = vmap(single_predict, in_axes=(None, 0, None))
predict = vmap(predict, in_axes=(None, None, 0))

def single_loss(layers, x, t):
    return (du_dt(layers, x, t) - d2u_dx2(layers, x, t))**2

vmap_single_loss = vmap(single_loss, in_axes=(None, 0, None))
vmap_single_loss = vmap(vmap_single_loss, in_axes=(None, None, 0))

def heat_eq_loss(layers, x, t):
    return jnp.mean(vmap_single_loss(layers, x, t))

def exact_solution(x, t):
    return jnp.exp(-jnp.pi**2 * t)*jnp.sin(jnp.pi*x)

exact_solution = vmap(exact_solution, in_axes=(0, None))
exact_solution = vmap(exact_solution, in_axes=(None, 0))

def l2_loss(layers, x, t):
    return jnp.mean(jnp.square(exact_solution(x, t) - predict(layers, x, t)))

def loss(layers, x, t):
    return heat_eq_loss(layers, x, t)

grad_loss = grad(loss)

@jit
def update(layers, x, t, step_size=0.01):
    grads = grad_loss(layers, x, t)
    return [(w-step_size*dw, b-step_size*db) for (w,b),(dw,db) in zip(layers, grads)]

def explicit_forward_euler(u0, T=0.01, h=0.01, dt=5.0e-5):
    if dt > 0.5*h**2:
        dt = 0.49*h**2

    x = jnp.arange(0.0, 1.0+h, h)
    t = jnp.arange(0.0, T+dt, dt)
    r = dt/h**2

    U = np.zeros((t.size, x.size))
    U[0] = u0(x)
    for k in range(t.size-1):
        u = U[k]
        U[k+1, 1:-1] = (1.0-2.0*r)*u[1:-1] + r*(u[:-2] + u[2:])
    
    return U

@jit
def u0(x):
    return jnp.sin(jnp.pi*x)

if __name__ == '__main__':
    key = random.PRNGKey(0)
    
    h = 0.1
    T = 1.0
    dt = 0.49*h**2
    x = jnp.arange(0.0, 1.0+h, h)
    t = jnp.arange(0.0, T+dt, dt)
    fe = explicit_forward_euler(u0, T, h, dt)

    plt.title('Mean Squared Error of Approximations')
    plt.xlabel('t')
    plt.ylabel('Mean squared error')
    plt.yscale('log')
    plt.plot(
        t,
        np.mean((fe - exact_solution(x, t))**2, axis=1), 
        label=f'Forward Euler (h = {h})'
    )

    h = 0.01
    T = 1.0
    dt = 0.49*h**2
    x = jnp.arange(0.0, 1.0+h, h)
    t = jnp.arange(0.0, T+dt, dt)
    fe = explicit_forward_euler(u0, T, h, dt)

    plt.plot(
        t, 
        np.mean((fe - exact_solution(x, t))**2, axis=1), 
        label=f'Forward Euler (h = {h})'
    )
    
    num_x_values = 128
    num_t_values = 128

    x = jnp.linspace(0.0, 1.0, num_x_values)
    t = jnp.linspace(0.0, 1.0, num_t_values)

    train_size = 0.75
    i_split = int(train_size*t.size)

    # Split `t` into training set and test set
    t_shuffled = random.permutation(key, t)
    t_train, t_test = t_shuffled[:i_split], t_shuffled[i_split:]

    batch_size = 16
    num_batches = t_train.size // batch_size + int(t_train.size % batch_size != 0)
    t_batches = [t_shuffled[batch_size*k:batch_size*(k+1)] for k in range(num_batches)]

    num_repetitions = 30
    num_layers = 4
    layer_sizes = num_layers*[2*num_repetitions] + [1]
    layers = random_layers(key, layer_sizes, 1.0, 1.0)

    exact = exact_solution(x, t)

    training_losses = []
    test_losses = []
    num_epochs = 2000
    epochs_to_plot = [500, 1000, 1500]
    epochs = range(num_epochs)
    for epoch in epochs:
        for t_batch in t_batches:
            layers = update(layers, x, t_batch)

        print(f'Completed epoch {epoch} of {num_epochs-1}')
        training_losses.append(loss(layers, x, t_train))
        test_losses.append(loss(layers, x, t_test))

        if epoch in epochs_to_plot:
            plt.plot(
                t,
                np.mean((predict(layers, x, t) - exact)**2, axis=1),
                label=f'Neural Network (epochs = {epoch})'
            )

    print(f'Losses after completing {num_epochs+1} epochs')
    print(f'Training loss: {training_losses[-1]}')
    print(f'Test loss: {test_losses[-1]}')
    plt.plot(
        t,
        np.mean((predict(layers, x, t) - exact)**2, axis=1),
        label=f'Neural Network (epochs = {epoch})'
    )
    plt.legend()
    plt.savefig('./figures/mean_squared_error_comparison.pdf')
    plt.show()

    """
    def plot_solutions(t0):
        h = x[1] - x[0]

        t0 = jnp.array([t0])
        plt.title(f'Solution at $t = {t0[0]}$')
        plt.ylim(0.0, 1.0)
        plt.xlabel('$x$')
        plt.plot(x, predict(layers, x, t0).flatten(), label='Neural Network')
        plt.plot(x, explicit_forward_euler(u0, t0[0], h), label='Forward Euler')
        plt.plot(x, exact_solution(x, t0).flatten(), label='Exact Solution', linestyle='dashed', linewidth=3.0)
        plt.legend()
        plt.savefig(f'./figures/solution_at_t{t0[0]:.1f}_{num_epochs}epochs.pdf')
        plt.show()

    plot_solutions(0.1)
    plot_solutions(0.3)
    plot_solutions(1.0)

    plt.title('Losses during training')
    plt.ylim(0.0, 1.1*max(max(training_losses), max(test_losses)))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, training_losses, label='Training loss')
    plt.plot(epochs, test_losses, label='Test loss')
    plt.legend()
    plt.savefig(f'./figures/losses_during_training_{num_epochs}epochs.pdf')
    plt.show()
    """