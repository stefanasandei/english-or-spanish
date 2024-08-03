import jax
import jax.numpy as jnp
import equinox as eqx

import matplotlib.pyplot as plt
import catppuccin

from data_utils import get_seq_dataset, get_data_params

# 1. hyperparameters
seed = 42
lr = 0.005
epochs = 20000
decay_step = epochs / 4 * 3

key = jax.random.key(seed)

# 2. data prep
data = get_data_params()
X, Y = get_seq_dataset(seed)

n = int(0.8 * data["data_size_seq"])

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

# TODO: lstm \w flax, optix

model = None

# 4. forward and backward passes


@eqx.filter_jit
def forward(model: eqx.Module, X: jax.Array) -> jax.Array:
    pass


@eqx.filter_jit
def get_loss(model: eqx.Module, X: jax.Array, y: jax.Array):
    pass


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
        lr /= 5

    # backward pass
    gradient = get_grad(model, Xb, Yb)
    updates = jax.tree_util.tree_map(lambda g: -lr * g, gradient)
    # model = eqx.apply_updates(model, updates)

# 6. validation test
valid_loss = 0.0
for i in range(len(Xval)):
    valid_loss += get_loss(model, Xval[i], Yval[i])
valid_loss /= len(Xval)
print(f"valid_loss={valid_loss:.3f}")  # best loss is 0.611

# 7. plot loss
lossi.pop()
lossi = jnp.mean(jnp.reshape(jnp.array(lossi), (-1, 100)), 1)

plt.figure(figsize=(10, 6))
plt.style.use(["ggplot", catppuccin.PALETTE.mocha.identifier])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.plot(lossi)
plt.savefig("test_rnn")
