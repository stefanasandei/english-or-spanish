import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import catppuccin

from data_utils import get_dataset, get_data_params

# 1. hyperparameters
seed = 42
lr = 1
epochs = 10000
decay_step = epochs / 4 * 3
batch_size = 256

key = jax.random.key(seed)

# 2. data prep
data = get_data_params()
X, Y = get_dataset(seed)

n = int(0.8 * data["data_size"])

# X is (batch, word, char); chars are one-hot vectors
# Y is (batch); single int representing the label
# X.shape = ((71416, 10, 28)
# Y.shape = (71416,))

Xtr, Ytr = X[:n], Y[:n]
Xval, Yval = X[n:], Y[n:]

# stats for nerds
lossi = []

# 3. network params
n_input = data["vocab_size"] * data["max_chars_in_word"]  # 28 * 10
n_hidden = 100
n_output = data["num_classes"]  # 7

W1 = jax.random.uniform(key, (n_input, n_hidden)) * 0.05  # kaiming init
b1 = jnp.zeros((n_hidden))
W2 = jax.random.uniform(key, (n_hidden, n_output)) * 0.01
b2 = jnp.zeros((n_output))
parameters = [W1, b1, W2, b2]

# 4. forward and backward passes


@jax.jit
def forward(params: list[jax.Array], X: jax.Array) -> jax.Array:
    h = X.reshape(-1, n_input)

    # go until the layer before the last one
    for W, b in zip(params[:-2], params[1:-2]):
        hpreact = h @ W + b
        h = jnp.tanh(hpreact)

    logits = h @ params[-2] + params[-1]
    return logits


@jax.jit
def get_loss(params, X: jax.Array, y: jax.Array):
    logits = forward(params, X)

    # cross entropy loss
    counts = jnp.exp(logits)
    prob = counts / counts.sum(1, keepdims=True)
    loss = -jnp.mean(jnp.log(prob[jnp.arange(y.shape[0]), y]))

    return loss


# differentiate with respect to the first argument (model params)
get_grad = jax.grad(get_loss)


@jax.jit
def update(params: jax.Array, gradient: jax.Array) -> jax.Array:
    return [p - lr * grad for p, grad in zip(params, gradient)]


# 5. training loop
for epoch in range(epochs + 1):
    # mini-batch
    key, _ = jax.random.split(key)  # get a new key from jax
    ix = jax.random.randint(key, (batch_size,), 0, Xtr.shape[0])
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    loss = get_loss(parameters, Xb, Yb)
    lossi.append(loss)

    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss:.2f}")

    # learning rate decay
    if epoch == decay_step:
        lr /= 10

    # backward pass
    gradient = get_grad(parameters, Xb, Yb)
    parameters = update(parameters, gradient)

# 6. validation test
valid_loss = get_loss(parameters, Xval, Yval)
print(f"valid_loss={valid_loss:.3f}")  # best loss is 1.065

# 7. plot
lossi.pop()
lossi = jnp.mean(jnp.reshape(jnp.array(lossi), (-1, 100)), 1)

plt.style.use(["ggplot", catppuccin.PALETTE.mocha.identifier])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.plot(lossi)
plt.show()
