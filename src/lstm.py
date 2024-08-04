import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
import catppuccin

from data_utils import get_seq_dataset, get_data_params

# todo: not working, needs to match the torch lstm file
# 1. hyperparameters
seed = 42
lr = 0.005
epochs = 1000
decay_step = epochs // 2
batch_size = 64

key = jax.random.PRNGKey(seed)

# 2. data prep
data = get_data_params()
X, Y = get_seq_dataset(seed)

max_seq_length = max(len(seq) for seq in X)
max_feature_length = max(len(seq[0]) for seq in X)

padded_X = []
for seq in X:
    padded_seq = []
    for sub_seq in seq:
        padded_sub_seq = jnp.pad(
            sub_seq, (0, max_feature_length - len(sub_seq)), mode="constant"
        )
        padded_seq.append(padded_sub_seq)
    padded_seq = jnp.pad(
        jnp.array(padded_seq), ((0, max_seq_length - len(seq)), (0, 0)), mode="constant"
    )
    padded_X.append(padded_seq)

X = jnp.array(padded_X)
n = int(0.8 * data["data_size_seq"])
Xtr, Ytr = X[:n], Y[:n]
Xval, Yval = X[n:], Y[n:]

# stats for nerds
lossi = []

# 3. network params
n_input = data["vocab_size"] * data["max_chars_in_word"]  # 28 * 10
n_hidden = 128
n_output = data["num_classes"]  # 7


class LSTMModel(nn.Module):
    n_input: int
    n_hidden: int
    n_output: int

    def setup(self):
        self.lstm = nn.LSTMCell(features=self.n_hidden)
        self.dense = nn.Dense(features=self.n_output)

    def __call__(self, x):
        batch_size, seq_length, _ = x.shape
        hidden = self.lstm.initialize_carry(
            jax.random.PRNGKey(seed), (batch_size,), self.n_hidden
        )

        def lstm_step(carry, x):
            carry, y = self.lstm(carry, x)
            return carry, y

        _, lstm_outputs = jax.lax.scan(lstm_step, hidden, x)
        logits = self.dense(lstm_outputs[0])
        return logits


model = LSTMModel(n_input=n_input, n_hidden=n_hidden, n_output=n_output)


# Initialize the model
@jax.jit
def init_model(key, batch):
    return model.init(key, batch)


# Define the loss function
@jax.jit
def compute_loss(params, batch, targets):
    logits = model.apply(params, batch)
    one_hot_targets = jax.nn.one_hot(targets, num_classes=n_output)
    loss = optax.softmax_cross_entropy(logits, one_hot_targets).mean()
    return loss


# Define the training step
@jax.jit
def train_step(state, batch, targets):
    loss_fn = lambda params: compute_loss(params, batch, targets)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Initialize the training state
init_batch = jnp.zeros((1, max_seq_length, n_input))
params = init_model(key, init_batch)

tx = optax.adamw(learning_rate=lr)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Training loop
for epoch in range(epochs + 1):
    key, subkey = jax.random.split(key)
    ix = jax.random.randint(subkey, (batch_size,), 0, len(Xtr))
    Xb, Yb = Xtr[ix], Ytr[ix]

    state, loss = train_step(state, Xb, Yb)

    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss:.2f}")
    lossi.append(loss)

    if epoch == decay_step:
        state = state.replace(tx=optax.adamw(learning_rate=lr / 10))


# 5. validation dataset loss
@jax.jit
def evaluate_model(params, Xval, Yval):
    logits = model.apply(params, Xval)
    loss = compute_loss(params, Xval, Yval)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == Yval) * 100.0
    return loss, accuracy


Xval = jnp.array(Xval)
Yval = jnp.array(Yval)
valid_loss, accuracy = evaluate_model(state.params, Xval, Yval)
print(f"valid_loss={valid_loss:.2f}")  # best loss is 0.27
print(f"accuracy={accuracy:.2f}%")  # best accuracy is 95.26%

# 6. plot loss
lossi = jnp.array(lossi)
lossi = jnp.mean(lossi.reshape(-1, 2), axis=1)
plt.figure(figsize=(10, 6))
plt.style.use(["ggplot", catppuccin.PALETTE.mocha.identifier])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.plot(lossi)
plt.savefig("test_jax_lstm")
plt.show()

# Save model parameters
with open("lstm_params.npz", "wb") as f:
    f.write(state.params)
