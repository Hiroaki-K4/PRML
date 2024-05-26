import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

import dataset
import prepare_dataset


class MixedDensityNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 9),
        )
        self.device = device
        self.t_candidates = torch.arange(0, 1.01, 0.01).to(self.device)

    def forward(self, x):
        logits = self.linear_tanh_stack(x)

        mu_0, mu_1, mu_2 = logits[0], logits[3], logits[6]
        sigma_0, sigma_1, sigma_2 = (
            torch.exp(logits[1]),
            torch.exp(logits[4]),
            torch.exp(logits[7]),
        )
        mix_coef_sum = (
            torch.exp(logits[2]) + torch.exp(logits[5]) + torch.exp(logits[8])
        )
        mix_coef_0 = torch.exp(logits[2]) / mix_coef_sum
        mix_coef_1 = torch.exp(logits[5]) / mix_coef_sum
        mix_coef_2 = torch.exp(logits[8]) / mix_coef_sum

        normal_dist_0 = Normal(mu_0, sigma_0)
        normal_dist_1 = Normal(mu_1, sigma_1)
        normal_dist_2 = Normal(mu_2, sigma_2)
        values = (
            mix_coef_0 * normal_dist_0.log_prob(self.t_candidates).exp()
            + mix_coef_1 * normal_dist_1.log_prob(self.t_candidates).exp()
            + mix_coef_2 * normal_dist_2.log_prob(self.t_candidates).exp()
        )
        pred_t = torch.max(values, dim=0)[0].unsqueeze(0)
        return pred_t


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")


def predict(model_path, test_data, device):
    model = MixedDensityNetwork(device).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    # print(test_data[0])
    # x, y = test_data[1][0], test_data[1][1]
    # print(x, y)
    # print(len(test_data))
    # input()
    for i in range(len(test_data)):
        with torch.no_grad():
            # x = x.to(device)
            x, y = test_data[i][0], test_data[i][1]
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            print(x, pred, y)
        # print(f'Predicted: "{predicted}", Actual: "{actual}"')


def main():
    random.seed(314)
    np.random.seed(314)

    train_N = 500
    x, y, input_x, input_y = prepare_dataset.create_dataset(train_N)
    training_data = dataset.MixedDensityNetworkDataset(
        np.array(input_x), np.array(input_y)
    )
    test_N = 50
    x, y, input_x, input_y = prepare_dataset.create_dataset(test_N)
    test_data = dataset.MixedDensityNetworkDataset(np.array(input_x), np.array(input_y))
    batch_size = 1
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, t in test_dataloader:
        print(f"Shape of X [N]: {X.shape}")
        print(f"Shape of y: {t.shape} {t.dtype}")
        break

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    mixed_density_model = MixedDensityNetwork(device).to(device)
    print(mixed_density_model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(mixed_density_model.parameters(), lr=1e-3)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, mixed_density_model, loss_fn, optimizer, device)
        test(test_dataloader, mixed_density_model, loss_fn, device)
    print("Done!")

    model_path = "mixed_density_model.pth"
    torch.save(mixed_density_model.state_dict(), model_path)
    print("Saved PyTorch Model State to {0}".format(model_path))

    predict(model_path, test_data, device)


if __name__ == "__main__":
    main()
