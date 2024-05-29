import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

plt.rcParams["figure.figsize"] = (8, 8)
np.random.seed(42)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


# generate data
n = 2500
d = 1
t = 1
x_train = np.random.uniform(0, 1, (n, d)).astype(np.float32)
noise = np.random.uniform(-0.1, 0.1, (n, d)).astype(np.float32)
y_train = x_train + 0.3 * np.sin(2 * np.pi * x_train) + noise
x_test = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)


# plot
# fig = plt.figure(figsize=(8, 8))
# plt.plot(x_train, y_train, 'go', alpha=0.5, markerfacecolor='none')
# plt.show()

# define a simple neural net
h = 15
w1 = Variable(torch.randn(d, h) * np.sqrt(1 / d), requires_grad=True)
b1 = Variable(torch.zeros(1, h), requires_grad=True)
w2 = Variable(torch.randn(h, t) * np.sqrt(1 / h), requires_grad=True)
b2 = Variable(torch.zeros(1, t), requires_grad=True)


def forward(x):
    out = torch.tanh(x.mm(w1) + b1)  # a relu introduces kinks in the predicted curve
    out = out.mm(w2) + b2
    return out


# wrap up the data as Variables
x = Variable(torch.from_numpy(x_train))
y = Variable(torch.from_numpy(y_train))

# select an optimizer
# NOTE: there are a few options here -- feel free to explore!
# opt = optim.SGD([w1, b1, w2, b2], lr=0.1)
opt = optim.SGD([w1, b1, w2, b2], lr=0.09, momentum=0.9, nesterov=True)
# opt = optim.RMSprop([w1, b1, w2, b2], lr=0.002, alpha=0.999)
# opt = optim.Adam([w1, b1, w2, b2], lr=0.09)

# optimize
# 10000 for SGD, 2000 for SGD w/ nesterov momentum, 4000 for RMSprop, 800 for Adam
for e in range(2000):
    opt.zero_grad()
    out = forward(x)
    loss = F.mse_loss(
        out, y
    )  # negative log likelihood assuming a Gaussian distribution
    if e % 100 == 0:
        print(e, loss.item())
    loss.backward()
    opt.step()

# predict
out = forward(Variable(torch.from_numpy(x_test)))
# plot
fig = plt.figure(figsize=(8, 8))
plt.plot(x_train, y_train, "go", alpha=0.5, markerfacecolor="none")
plt.plot(x_test, out.data.numpy(), "r", linewidth=3.0)
plt.show()
