import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_COINS = 10
SAMPLES = 20
np.random.seed(1)

class LSTMInvestor(nn.Module):
    def __init__(self, n_coins, hidden_dim, n_options):
        super(LSTMInvestor, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(n_coins, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, n_options)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, coins):
        coins = coins.view(-1, 1, len(coins))
        lstm_out, self.hidden = self.lstm(coins, self.hidden)
        lstm_out = lstm_out[-1]
        tag_space = self.hidden2tag(lstm_out.view(1, self.hidden_dim))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

def generate_coins(n, steps):
    data = np.zeros((n, steps))
    for coin in range(n):
        for step in range(steps):
            if step > 0:
                prev = data[coin, step - 1]
            else:
                prev = np.random.randint(0.01, 100)
            change = np.random.normal(0.0, 0.25, 1)
            data[coin, step] = prev + prev * change

    return np.clip(data, 0, None)

def normalize(arr):
    arr -= arr.mean()
    arr /= arr.std()
    return arr

def visualize_coins(data):
    for coin in data:
        plt.plot(coin)
    plt.savefig("coins.png")

model = LSTMInvestor(N_COINS, 16, N_COINS + 1)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300): 
    for i in range(1000):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        coins = generate_coins(10, 20)
        coins = normalize(coins)
        inputs = coins[:, :-5]
        outputs = coins[:, -1]

        best_final_input = inputs[:, -1].max()
        if outputs.max() < best_final_input:
            best_investment = -1
        else:
            best_investment = np.argmax(outputs)

        target = np.zeros((N_COINS + 1, ), dtype="int8")
        target[best_investment] = 1
        target = autograd.Variable(torch.LongTensor(target.tolist()))

        inputs = autograd.Variable(torch.Tensor(inputs))

        # Step 3. Run our forward pass.
        tag_scores = model(inputs)
        print tag_scores.size()
        print target.size()
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, target)
        loss.backward()
        optimizer.step()
