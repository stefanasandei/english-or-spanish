import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import catppuccin

from data_utils import get_seq_dataset, get_data_params, encode_word

# 1. hyperparameters
seed = 42
lr = 0.005
epochs = 20000
decay_step = epochs / 4 * 3

torch.random.manual_seed(seed)

# 2. data prep
data = get_data_params()
X, Y = get_seq_dataset(seed)

X = [torch.tensor(x, dtype=torch.float32) for x in X]
Y = torch.tensor(Y)

n = int(0.8 * data["data_size_seq"])

Xtr, Ytr = X[:n], Y[:n]
Xval, Yval = X[n:], Y[n:]

# stats for nerds
lossi = []


# 3. network params
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(-1, self.input_size)
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(data["max_chars_in_word"] * data["vocab_size"], n_hidden, data["num_classes"])

# 4. training
rnn.train()
for epoch in range(epochs + 1):
    ix = torch.randint(0, len(Xtr), (1,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    hidden = torch.zeros((1, n_hidden))
    for i in range(Xb.shape[0]):
        logits, hidden = rnn(Xb[i], hidden)

    loss = F.cross_entropy(logits, Yb)
    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss={loss.item():.2f}")
    lossi.append(loss.item())

    if epoch == decay_step:
        lr /= 10

    # backward pass
    rnn.zero_grad()
    loss.backward()
    for p in rnn.parameters():
        p.data += -lr * p.grad


# 5. validation dataset loss
rnn.eval()
loss = 0
for i in range(len(Xval)):
    Xb, Yb = Xval[i], Yval[i]

    hidden = torch.zeros((1, n_hidden))
    for i in range(Xb.shape[0]):
        logits, hidden = rnn(Xb[i], hidden)

    loss += F.cross_entropy(logits, torch.tensor([Yb]))
loss /= len(Xval)
print(f"val_loss={loss:.2f}")  # best loss is 0.54

# 6. plot loss
lossi.pop()
lossi = torch.tensor(lossi).view(-1, 1000).mean(1)
plt.figure(figsize=(10, 6))
plt.style.use(["ggplot", catppuccin.PALETTE.mocha.identifier])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.plot(lossi)
plt.savefig("test_torch_rnn")
plt.show()

torch.save(rnn.state_dict(), "./rnn.torch")
