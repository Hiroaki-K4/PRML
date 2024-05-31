import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

plt.rcParams["figure.figsize"] = (8, 8)
np.random.seed(314)
torch.manual_seed(314)
torch.cuda.manual_seed(314)


# generate data
n = 3000
d = 1
t = 1
x_train = np.random.uniform(0, 1, (n, d)).astype(np.float32)
noise = np.random.uniform(-0.1, 0.1, (n, d)).astype(np.float32)
y_train = x_train + 0.3 * np.sin(2 * np.pi * x_train) + noise
x_train_inv = y_train
y_train_inv = x_train
x_test = np.linspace(-0.1, 1.1, n).reshape(-1, 1).astype(np.float32)

h = 50  # Dimension of hidden layer
k = 4  # Mixing components
d_pi = k
d_sigmasq = k
d_mu = t * k

# Initialize parameters
w1 = Variable(torch.randn(d, h) * np.sqrt(2 / (d + h)), requires_grad=True)
b1 = Variable(torch.zeros(1, h), requires_grad=True)
w_pi = Variable(torch.randn(h, d_pi) * np.sqrt(2 / (d + h)), requires_grad=True)
b_pi = Variable(torch.zeros(1, d_pi), requires_grad=True)
w_sigmasq = Variable(
    torch.randn(h, d_sigmasq) * np.sqrt(2 / (d + h)), requires_grad=True
)
b_sigmasq = Variable(torch.zeros(1, d_sigmasq), requires_grad=True)
w_mu = Variable(torch.randn(h, d_mu) * np.sqrt(2 / (d + h)), requires_grad=True)
b_mu = Variable(torch.zeros(1, d_mu), requires_grad=True)


def predict(pi, mu):
    n, k = pi.shape
    _, kt = mu.shape
    t = int(kt / k)  # label dim
    _, max_component = torch.max(pi, 1)  # Get index of max mixture component
    out = Variable(torch.zeros(n, t))
    for i in range(n):
        for j in range(t):
            out[i, j] = mu[i, max_component.data[i] * t + j]

    return out


def forward(x):
    out = F.tanh(x.mm(w1) + b1)  # shape (n, d) -> (n, h)
    pi = F.softmax(out.mm(w_pi) + b_pi, dim=1)  # shape (n, h) -> (n, d_pi)
    sigmasq = torch.exp(out.mm(w_sigmasq) + b_sigmasq)  # shape (n, h) -> (n, d_sigmasq)
    mu = out.mm(w_mu) + b_mu  # shape (n, h) -> (n, d_mu)
    return pi, sigmasq, mu


def gaussian_pdf(x, mu, sigmasq):
    return (1 / torch.sqrt(2 * np.pi * sigmasq)) * torch.exp(
        (-1 / (2 * sigmasq)) * torch.norm((x - mu), 2, 1) ** 2
    )


def loss_fn(pi, sigmasq, mu, target):
    losses = Variable(torch.zeros(n))
    for i in range(k):
        likelihood_z_x = gaussian_pdf(target, mu[:, i * t : (i + 1) * t], sigmasq[:, i])
        losses += pi[:, i] * likelihood_z_x

    loss = torch.mean(-torch.log(losses))
    return loss


def train():
    opt = optim.Adam([w1, b1, w_pi, b_pi, w_sigmasq, b_sigmasq, w_mu, b_mu], lr=0.008)

    x = Variable(torch.from_numpy(x_train_inv))
    y = Variable(torch.from_numpy(y_train_inv))

    epochs = 1000
    for epoch in range(epochs):
        opt.zero_grad()
        pi, sigmasq, mu = forward(x)
        loss = loss_fn(pi, sigmasq, mu, y)
        if epoch % 100 == 0:
            print("Epoch: {0}/{1}, Loss: {2}".format(epoch, epochs, loss.item()))
        loss.backward()
        opt.step()


def main():
    # Training
    train()

    # Prediction
    pi, sigmasq, mu = forward(Variable(torch.from_numpy(x_test)))
    preds = predict(pi, mu)

    # Plot mixture density network
    plt.plot(x_train_inv, y_train_inv, "bo", alpha=0.5, markerfacecolor="none")
    plt.plot(x_test, preds.data.numpy(), "r.")
    title = "Mixture density network ($N={0}$)".format(n)
    plt.title(title)

    # Plot mixing components
    plt.figure(figsize=(8, 8))
    for i in range(pi.shape[1]):
        plt.plot(x_test, pi[:, i].detach().numpy())
    title = "Mixing components ($K={0}$)".format(k)
    plt.title(title)

    # Plot mu
    plt.figure(figsize=(8, 8))
    for i in range(mu.shape[1]):
        plt.plot(x_test, mu[:, i].detach().numpy())
    title = "$\mu$ of mixture density network"
    plt.title(title)


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
