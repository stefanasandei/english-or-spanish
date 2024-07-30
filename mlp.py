import jax
import jax.numpy as jnp

from data_utils import get_dataset, get_data_params

# 1. hyperparameters
seed = 42

key = jax.random.key(seed)

# 2. data prep
X, Y = get_dataset(seed)

n = int(0.8 * get_data_params()["data_size"])

Xtr, Ytr = X[:n], Y[:n]
Xval, Yval = X[n:], Y[n:]

# 3. network params

# 4. loss function

# 5. training loop

# 6. validation test
