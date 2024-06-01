import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Create dataset
n = 2000
d = 1
t = 1
x_train = np.random.uniform(0, 1, (n, d)).astype(np.float32)
noise = np.random.uniform(-0.1, 0.1, (n, d)).astype(np.float32)
y_train = x_train + 0.3 * np.sin(2 * np.pi * x_train) + noise
x_train_inv = y_train
y_train_inv = x_train
x_test = np.linspace(-0.1, 1.1, n).reshape(-1, 1).astype(np.float32)
h = 15  # Dimension of hidden layer
# Initialize parameters
w1 = Variable(torch.randn(d, h) * np.sqrt(1 / d), requires_grad=True)
b1 = Variable(torch.zeros(1, h), requires_grad=True)
w2 = Variable(torch.randn(h, t) * np.sqrt(1 / h), requires_grad=True)
b2 = Variable(torch.zeros(1, t), requires_grad=True)
x = Variable(torch.from_numpy(x_train_inv))
y = Variable(torch.from_numpy(y_train_inv))


def forward(x):
    out = torch.tanh(x.mm(w1) + b1)
    out = out.mm(w2) + b2
    return out


def train():
    opt = optim.Adam([w1, b1, w2, b2], lr=0.09)

    # Optimize
    epochs = 2000
    for epoch in range(epochs):
        opt.zero_grad()
        out = forward(x)
        loss = F.mse_loss(out, y)
        if epoch % 100 == 0:
            print("Epoch: {0}/{1}, Loss: {2}".format(epoch, epochs, loss.item()))
        loss.backward()
        opt.step()


def main():
    plt.rcParams["figure.figsize"] = (7, 7)
    np.random.seed(314)
    torch.manual_seed(314)
    torch.cuda.manual_seed(314)

    train()

    # predict
    out = forward(torch.from_numpy(x_test))
    # plot
    plt.plot(x_train_inv, y_train_inv, "bo", alpha=0.5, markerfacecolor="none")
    plt.plot(x_test, out.data.numpy(), "r", linewidth=3.0)
    title = "Inverse problem with simple neural network($N={0}$)".format(n)
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
