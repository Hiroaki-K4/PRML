import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import dataset
import prepare_dataset


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class MixedDensityNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # TODO: Fix shape error
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 9),
        )
        self.t_candidates = torch.arange(0, 1.01, 0.01)

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        mu_0, sigma_0, mix_coef_0 = logits[0], logits[1], logits[2]
        mu_1, sigma_1, mix_coef_1 = logits[3], logits[4], logits[5]
        mu_2, sigma_2, mix_coef_2 = logits[6], logits[7], logits[8]
        normal_dist_0 = Normal(mu_0, sigma_0)
        normal_dist_1 = Normal(mu_1, sigma_1)
        normal_dist_2 = Normal(mu_2, sigma_2)
        values = (
            mix_coef_0 * normal_dist_0.log_prob(self.t_candidates).exp()
            + mix_coef_1 * normal_dist_1.log_prob(self.t_candidates).exp()
            + mix_coef_2 * normal_dist_2.log_prob(self.t_candidates).exp()
        )
        pred_t = torch.max(values, dim=0)
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
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Result: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def predict(model_path, test_data, device):
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path))
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def main():
    random.seed(314)
    np.random.seed(314)

    # mu = 0.0
    # sigma = 1.0
    # normal_dist = Normal(mu, sigma)
    # print(normal_dist)
    # x = torch.arange(0, 1.01, 0.01)
    # pdf_value = 3 * normal_dist.log_prob(x).exp()
    # print(pdf_value)
    # print(torch.max(pdf_value, dim=0))
    # input()

    train_N = 100
    x, y, input_x, input_y = prepare_dataset.create_dataset(train_N)
    training_data = dataset.MixedDensityNetworkDataset(
        np.array(input_x), np.array(input_y)
    )
    test_N = 10
    x, y, input_x, input_y = prepare_dataset.create_dataset(test_N)
    test_data = dataset.MixedDensityNetworkDataset(np.array(input_x), np.array(input_y))
    batch_size = 16
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, t in test_dataloader:
        print(f"Shape of X [N]: {X.shape}")
        print(f"Shape of y: {t.shape} {t.dtype}")
        break

    # # Download training data from open datasets.
    # training_data = datasets.FashionMNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=ToTensor(),
    # )
    # # Download test data from open datasets.
    # test_data = datasets.FashionMNIST(
    #     root="data",
    #     train=False,
    #     download=True,
    #     transform=ToTensor(),
    # )

    # # Create data loaders.
    # train_dataloader = DataLoader(training_data, batch_size=batch_size)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # model = NeuralNetwork().to(device)
    mixed_density_model = MixedDensityNetwork().to(device)
    print(mixed_density_model)
    input()

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(mixed_density_model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train(train_dataloader, model, loss_fn, optimizer, device)
        # test(test_dataloader, model, loss_fn, device)
        train(train_dataloader, mixed_density_model, loss_fn, optimizer, device)
        test(test_dataloader, mixed_density_model, loss_fn, device)
    print("Done!")

    model_path = "mixed_density_model.pth"
    # torch.save(model.state_dict(), model_path)
    torch.save(mixed_density_model.state_dict(), model_path)
    print("Saved PyTorch Model State to model.pth")

    # predict(model_path, test_data, device)


if __name__ == "__main__":
    main()
