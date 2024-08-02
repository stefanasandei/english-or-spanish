import jax
import jax.numpy as jnp
import equinox as eqx

import matplotlib.pyplot as plt
import catppuccin

from data_utils import get_seq_dataset, get_data_params, encode_word

# 1. hyperparameters
seed = 42
lr = 0.005
epochs = 20000
decay_step = epochs / 4 * 3

key = jax.random.key(seed)

# 2. data prep
data = get_data_params()
X, Y = get_seq_dataset(seed)

n = int(0.8 * data["data_size"])

# X is a list of sentences, each sentence is (num_words, word, char); chars are one-hot vectors
# Y is (batch); single int representing the label
# X[0].shape = (16, 10, 28) (for example, one sentence with 16 words)
# Y.shape = (5801), len(X) = 5801

Xtr, Ytr = X[:n], Y[:n]
Xval, Yval = X[n:], Y[n:]

# stats for nerds
lossi = []

# 3. network params
n_input = data["vocab_size"] * data["max_chars_in_word"]  # 28 * 10
n_hidden = 128
n_output = data["num_classes"]  # 7


class RNN(eqx.Module):
    n_input: int

    Wi2h: jax.Array
    Wh2h: jax.Array
    bh: jax.Array
    Wh2o: jax.Array
    bo: jax.Array

    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        self.n_input = n_input

        self.Wi2h = jax.random.uniform(key, (n_input, n_hidden)) * 0.05  # kaiming init
        self.Wh2h = jax.random.uniform(key, (n_hidden, n_hidden)) * 0.01
        self.bh = jnp.zeros((n_hidden))
        self.Wh2o = jax.random.uniform(key, (n_hidden, n_output)) * 0.01
        self.bo = jnp.zeros((n_output))

    def __call__(self, X: jax.Array, hidden: jax.Array) -> tuple[jax.Array, jax.Array]:
        X = X.reshape(-1, self.n_input)

        hpreact = X @ self.Wi2h + hidden @ self.Wh2h + self.bh
        hidden = jnp.tanh(hpreact)
        out = hidden @ self.Wh2o + self.bo

        return out, hidden


model = RNN(n_input, n_hidden, n_output)

# 4. forward and backward passes


@eqx.filter_jit
def forward(model: eqx.Module, X: jax.Array) -> jax.Array:
    hidden = jnp.zeros((1, n_hidden))

    for i in range(X.shape[0]):
        logits, hidden = model(X[i], hidden)

    return logits


@eqx.filter_jit
def get_loss(model: eqx.Module, X: jax.Array, y: jax.Array):
    logits = forward(model, X)

    # cross entropy loss
    counts = jnp.exp(logits)
    prob = counts / counts.sum(1, keepdims=True)
    loss = -jnp.mean(jnp.log(prob[1, y]))

    return loss


# differentiate with respect to the first argument (model params)
get_grad = eqx.filter_grad(get_loss)

# 5. training loop
for epoch in range(epochs + 1):
    # one random example sentence
    key, _ = jax.random.split(key)  # get a new key from jax
    ix = jax.random.randint(key, (1,), 0, len(Xtr))[0]
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    loss = get_loss(model, Xb, Yb)
    lossi.append(loss)

    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss:.2f}")

    # learning rate decay
    if epoch == decay_step:
        lr /= 10

    # backward pass
    gradient = get_grad(model, Xb, Yb)
    updates = jax.tree_util.tree_map(lambda g: -lr * g, gradient)
    model = eqx.apply_updates(model, updates)

# warning: this seems to be broken lol
