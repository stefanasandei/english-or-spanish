import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
import catppuccin

from data_utils import get_seq_dataset, get_data_params

# 1. hyperparameters
seed = 42
lr = 0.005
epochs = 2000
batch_size = 64

key = jax.random.key(seed)

# 2. data prep
data = get_data_params()
X, Y = get_seq_dataset(seed)


def transform_X(X):
    """from list of sentences to (batch, sentence, word, char)"""
    # Determine the batch size
    batch_size = len(X)

    # Find the maximum length along the first dimension
    max_len = max(array.shape[0] for array in X)

    # Initialize the new array with zeros
    result = jnp.zeros((batch_size, max_len, 10, 28))

    # Copy the data from X into the result array
    for i, array in enumerate(X):
        length = array.shape[0]
        result = result.at[i, :length, :, :].set(array)

    return result


X = jnp.array(transform_X(X))
n = int(0.8 * data["data_size_seq"])
Xtr, Ytr = X[:n], Y[:n]
Xval, Yval = X[n:], Y[n:]

# stats for nerds
lossi = []

# 3. network params


class LSTM(nn.Module):
    sentence_size: int
    input_size: int
    hidden_size: int
    output_size: int

    def setup(self):
        self.lstm = nn.OptimizedLSTMCell(features=self.hidden_size)
        self.h2o = nn.Dense(self.output_size)

    def __call__(self, x):
        x = x.reshape(-1, self.sentence_size, self.input_size)
        batch_size, seq_length = x.shape[0], x.shape[1]
        carry = self.lstm.initialize_carry(
            key, (batch_size,))

        outputs = []
        for t in range(seq_length):
            carry, y = self.lstm(carry, x[:, t])
            outputs.append(y)

        output = outputs[-1]
        output = self.h2o(output)

        return output


input_size = data["max_chars_in_word"] * data["vocab_size"]
hidden_size = 128
output_size = data["num_classes"]

# Initialize the model
model = LSTM(sentence_size=data["max_words_in_sentence"], input_size=input_size, hidden_size=hidden_size,
             output_size=output_size)

# Create an example input to initialize the model parameters
ix = jax.random.randint(key, (batch_size,), 0, len(Xtr))
example_input = Xtr[ix]

# Initialize model parameters
variables = model.init(key, example_input)
params = variables['params']

# Define a simple optimizer and training state
optimizer = optax.adam(learning_rate=lr)
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=optimizer)

# 4. forward and backward passes


@jax.jit
def compute_loss(params, batch, targets):
    logits = model.apply({'params': params}, batch)
    one_hot_targets = jax.nn.one_hot(targets, num_classes=output_size)
    loss = optax.softmax_cross_entropy(logits, one_hot_targets).mean()
    return loss


@jax.jit
def train_step(state, batch, targets):
    def loss_fn(params):
        return compute_loss(params, batch, targets)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 5. training loop


for epoch in range(epochs + 1):
    key, _ = jax.random.split(key)
    ix = jax.random.randint(key, (batch_size,), 0, len(Xtr))
    Xb, Yb = Xtr[ix], Ytr[ix]

    state, loss = train_step(state, Xb, Yb)
    lossi.append(loss)

    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss:.2f}")

# 6. validation loss


@jax.jit
def evaluate_model(params, Xval, Yval):
    logits = model.apply({'params': params}, Xval)
    loss = compute_loss(params, Xval, Yval)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == Yval) * 100.0
    return loss, accuracy


valid_loss, accuracy = evaluate_model(state.params, Xval, Yval)
print(f"valid_loss={valid_loss:.2f}")  # best loss is 0.27
print(f"accuracy={accuracy:.2f}%")  # best accuracy is 95.26%

# 7. plot loss

lossi.pop()
lossi = jnp.mean(jnp.array(lossi).reshape(200, -1), axis=1)
plt.figure(figsize=(10, 6))
plt.style.use(["ggplot", catppuccin.PALETTE.mocha.identifier])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.plot(lossi)
plt.savefig("test_jax_lstm")
plt.show()
