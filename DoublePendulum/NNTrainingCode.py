import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ExactODE as Exact
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from random import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

prev_trajectory = []
future_trajectory = []
mu_x1 = None
mu_y1 = None
sigma_x1 = None
sigma_y1 = None
mu_x2 = None
mu_y2 = None
sigma_x2 = None
sigma_y2 = None
first = True
lens = 10
for types in [[random() * 5, random() * 2, random() * 5, random() * 2, [random() * 2 - 1, random() * 3 - 1, random() * 2 - 1, random() * 3 - 1]] for i in range(100)]:
    #print("Generating Data")
    m1 = types[0]; l1 = types[1]; m2 = types[2]; l2 = types[3]; g = 9.8
    u = 0.0
    t = np.linspace(0, 100, 10_000)
    x_init = np.array(types[4])  # I.C.
    x = Exact.exact_integration(x_init, t, u, m1, l1, m2, l2, g)
    px1, py1 = l1 * np.sin(x[:, 0]), -l1 * np.cos(x[:, 0])
    px2, py2 = px1 + l2 * np.sin(x[:, 2]), py1 - l2 * np.cos(x[:, 2])

    #get normalization
    normalization = lambda x, mu, sigma: (x - mu) / (sigma + 1e-6)
    #print("Normalizing Dataset")
    if first:
        mu_x1 = np.mean(px1)
        mu_y1 = np.mean(py1)
        sigma_x1 = np.std(px1)
        sigma_y1 = np.std(py1)
        mu_x2 = np.mean(px2)
        mu_y2 = np.mean(py2)
        sigma_x2 = np.std(px2)
        sigma_y2 = np.std(py2)
        first = False

    px1 = normalization(px1, mu_x1, sigma_x1)
    py1 = normalization(py1, mu_y1, sigma_y1)
    px2 = normalization(px2, mu_x2, sigma_x2)
    py2 = normalization(py2, mu_y2, sigma_y2)

    #print("Constructing Data")
    trajectory_idx = [i for i in range(lens, len(px1)-lens)]

    prev_trajectory.append([[[px1[j], py1[j], px2[j], py2[j]] for j in range(i-lens, i)] for i in trajectory_idx])
    future_trajectory.append([[[px1[j], py1[j], px2[j], py2[j]] for j in range(i, i+lens)] for i in trajectory_idx])
prev_trajectory = torch.tensor(prev_trajectory, dtype=torch.float32)
future_trajectory = torch.tensor(future_trajectory, dtype=torch.float32)
prev_trajectory = prev_trajectory.reshape(-1, lens * 4)
future_trajectory = future_trajectory.reshape(-1, lens * 4)
print("prev shape:", prev_trajectory.shape)
print("future shape:", future_trajectory.shape)

print("Batching data")
train_percentage = 0.8
train_prev_traj = prev_trajectory[:int (0.8*len(prev_trajectory))]
train_future_traj = future_trajectory[:int (0.8*len(prev_trajectory))]

test_prev_traj = prev_trajectory[int (0.8*len(prev_trajectory)):]
test_future_traj = future_trajectory[int (0.8*len(prev_trajectory)):]

batch_size = 16
train_dataset = TensorDataset(train_prev_traj, train_future_traj)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_prev_traj, test_future_traj)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, z_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.encoder_net = self.encoder(input_size, hidden_size, z_size)
        self.decoder_net = self.decoder(output_size, hidden_size, z_size)

    def forward(self, x):
        z = self.encoder_net(x)
        x_hat = self.decoder_net(z)
        return x_hat

    def encoder(self, input_dim, hidden_size, z_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, z_dim),
        )

    def decoder(self, output_dim, hidden_size, z_dim):
        return nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

def train(model, optimizer, train_loader, test_loader, num_epochs = 20):
    train_loss_vals = []
    test_loss_vals = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            predicted_future = model(batch_input)
            loss_fn = nn.MSELoss()
            loss = loss_fn(predicted_future, batch_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        #testing
        model.eval()
        test_loss = 0.0
        for batch_input, batch_target in test_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            predicted_future = model(batch_input)
            loss_fn = nn.MSELoss()
            loss = loss_fn(predicted_future, batch_target)
            test_loss += loss.item()

        test_loss /= len(test_loader)
        train_loss_vals.append(train_loss)
        test_loss_vals.append(test_loss)

        if epoch % 5 == 0:
            print("="*30)
            print("Epoch #", epoch)
            print("\tTrain loss: ", train_loss)
            print("\tTest loss: ", test_loss)
            print("=" * 30)
            print()

    #plot loss curves
    print("Plotting loss curve")
    plt.figure()
    plt.plot(train_loss_vals)
    plt.plot(test_loss_vals)
    plt.legend(['Train loss', 'Test loss'], loc='lower right')
    plt.savefig("loss_curve.png")

print("Creating model")
lr = 1e-4
opt = optim.Adam
num_epochs = 20
input_size = lens * 4
hidden_size = input_size // 2
output_size = input_size
z_dim = 10
model = NeuralNetwork(input_size, hidden_size, z_dim, output_size)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Training Model...")
train(model, optimizer, train_loader, test_loader, num_epochs)
print("Model Trained")

#save model and normalization
print("Saving model...")
torch.save(model.state_dict(), "pendulum_model.pth")
np.savez("normalization.npz", mu_x1=mu_x1, sigma_x1=sigma_x1, mu_y1=mu_y1, sigma_y1=sigma_y1, mu_x2 = mu_x2, sigma_x2=sigma_x2, mu_y2=mu_y2, sigma_y2=sigma_y2)
print("Model Saved")
