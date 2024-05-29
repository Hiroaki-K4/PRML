import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

import dataset
import prepare_dataset


class MixtureDensityNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 9),
        )
        self.device = device
        self.t_candidates = torch.arange(0, 1.01, 0.01).to(self.device)

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits

    def predict(self, logits):
        mu = logits[:3].to("cpu").detach().numpy().copy()
        var = torch.exp(logits[3:6]).to("cpu").detach().numpy().copy()
        mix_coef = logits[6:]
        mix_coef_softmax = torch.softmax(mix_coef, dim=-1)
        # idx = random.choices([i for i in range(3)], weights=mix_coef_softmax)
        max_value, idx = torch.max(mix_coef_softmax, dim=0)
        print(mix_coef_softmax, idx)
        pred_t = np.random.normal(mu[idx], np.sqrt(var[idx]))
        return pred_t

    def loss(self, logits, y, loss_fn):
        mu_0, mu_1, mu_2 = logits[0], logits[1], logits[2]
        var_0, var_1, var_2 = (
            torch.exp(logits[3]),
            torch.exp(logits[4]),
            torch.exp(logits[5]),
        )
        mix_coef = logits[6:]
        mix_coef_softmax = torch.softmax(mix_coef, dim=-1)

        normal_dist_0 = Normal(mu_0, torch.sqrt(var_0))
        normal_dist_1 = Normal(mu_1, torch.sqrt(var_1))
        normal_dist_2 = Normal(mu_2, torch.sqrt(var_2))
        loss = -torch.log(
            mix_coef_softmax[0] * normal_dist_0.log_prob(y).exp()
            + mix_coef_softmax[1] * normal_dist_1.log_prob(y).exp()
            + mix_coef_softmax[2] * normal_dist_2.log_prob(y).exp()
        )
        return loss


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        logits = model(X)
        loss = model.loss(logits, y, loss_fn)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            test_loss += model.loss(logits, y, loss_fn).item()

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")


def predict(model_path, test_data, device):
    model = MixtureDensityNetwork(device).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    xs = []
    ys = []
    preds = []
    for i in range(len(test_data)):
        with torch.no_grad():
            x, y = test_data[i][0], test_data[i][1]
            x = x.unsqueeze(0).to(device)
            logits = model(x)
            pred = model.predict(logits)
            xs.append(x.to("cpu").detach().numpy().copy())
            ys.append(y.to("cpu").detach().numpy().copy())
            preds.append(pred)

    plt.scatter(xs, ys, label="Input", color="green")
    plt.scatter(xs, preds, label="Prediction", color="blue")
    plt.legend()
    # title = "Mixed density network(N={0})".format(str(N))
    # plt.title(title)
    plt.show()
    # print(f'Predicted: "{predicted}", Actual: "{actual}"')


def main():
    random.seed(314)
    np.random.seed(314)

    train_N = 500
    x, y, input_x, input_y = prepare_dataset.create_dataset(train_N)
    training_data = dataset.MixtureDensityNetworkDataset(
        np.array(input_x), np.array(input_y)
    )
    test_N = 100
    x, y, input_x, input_y = prepare_dataset.create_dataset(test_N)
    test_data = dataset.MixtureDensityNetworkDataset(
        np.array(input_x), np.array(input_y)
    )
    batch_size = 1
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, t in test_dataloader:
        print(f"Shape of X [N]: {X.shape}")
        print(f"Shape of y: {t.shape} {t.dtype}")
        break

    device = "cpu"
    print(f"Using {device} device")

    mixture_density_model = MixtureDensityNetwork(device).to(device)
    print(mixture_density_model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(mixture_density_model.parameters(), lr=1e-3)

    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, mixture_density_model, loss_fn, optimizer, device)
        test(test_dataloader, mixture_density_model, loss_fn, device)
    print("Done!")

    model_path = "mixture_density_model.pth"
    torch.save(mixture_density_model.state_dict(), model_path)
    print("Saved PyTorch Model State to {0}".format(model_path))

    predict(model_path, test_data, device)


if __name__ == "__main__":
    main()
