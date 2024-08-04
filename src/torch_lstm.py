import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

import matplotlib.pyplot as plt
import catppuccin

from data_utils import get_seq_dataset, get_data_params

# 1. hyperparameters
seed = 42
lr = 0.005
epochs = 20000
decay_step = epochs / 4 * 3
batch_size = 32

torch.random.manual_seed(seed)

# 2. data prep
data = get_data_params()
X, Y = get_seq_dataset(seed)

X = pad_sequence([torch.tensor(x, dtype=torch.float32) for x in X], batch_first=True)
Y = torch.tensor(Y)

n = int(0.8 * data["data_size_seq"])

Xtr, Ytr = X[:n], Y[:n]
Xval, Yval = X[n:], Y[n:]

# stats for nerds
lossi = []

# 3. network params


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        batch_size = input.shape[0]

        # (batch, seq, feature)
        input = input.view(batch_size, -1, self.input_size)

        _, hidden = self.lstm(input)
        output = self.h2o(hidden[0])

        return output.view(batch_size, -1)


n_hidden = 128
lstm = LSTM(
    data["max_chars_in_word"] * data["vocab_size"], n_hidden, data["num_classes"]
)

# 4. training

optimizer = optim.Adam(lstm.parameters(), lr=lr)

lstm.train()
for epoch in range(epochs + 1):
    ix = torch.randint(0, len(Xtr), (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    logits = lstm(Xb)

    loss = F.cross_entropy(logits, Yb)
    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss.item():.2f}")
    lossi.append(loss.item())

    if epoch == decay_step:
        optimizer.param_groups[0]["lr"] = lr / 10

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. validation dataset loss
lstm.eval()
loss, accuracy = 0, 0

# cross entropy loss
logits = lstm(Xval)
loss = F.cross_entropy(logits, Yval)
counts = logits.exp()

# accuracy
prob = counts / counts.sum(1, keepdim=True)
label_index = torch.argmax(prob, 1)
accuracy = (label_index == Yval).sum() / Yval.shape[0] * 100.0

print(f"valid_loss={loss:.2f}")  # best loss is 0.27
print(f"accuracy={accuracy:.2f}%")  # best accuracy is 95.26%

# 6. plot loss
lossi.pop()
lossi = torch.tensor(lossi).view(-1, 100).mean(1)
plt.figure(figsize=(10, 6))
plt.style.use(["ggplot", catppuccin.PALETTE.mocha.identifier])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.plot(lossi)
plt.savefig("test_torch_lstm")
plt.show()

torch.save(lstm.state_dict(), "./lstm.torch")
