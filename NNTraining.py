import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ExactODE as Exact
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

print("Generating Data")
m = 1.0; l = 1.0; g = 9.8
u = 0.0
t = np.linspace(0, 1000, 100_000)
x_init = np.array([0, 2.0])  # I.C.
x = Exact.exact_integration(x_init, t, u, m, l, g)
px, py = l*np.sin(x[:, 0]), -l*np.cos(x[:, 0])

#get normalization
normalization = lambda x, mu, sigma: (x - mu) / (sigma + 1e-6)
print("Normalizing Dataset")
mu_x = np.mean(px)
mu_y = np.mean(py)
sigma_x = np.std(px)
sigma_y = np.std(py)
px = normalization(px, mu_x, sigma_x)
py = normalization(py, mu_y, sigma_y)

print("Constructing Data")
lens = 10
trajectory_idx = [i for i in range(lens, len(px)-lens)]

prev_trajectory = torch.tensor([[[px[j], py[j]] for j in range(i-lens, i)] for i in trajectory_idx], dtype=torch.float32)
future_trajectory = torch.tensor([[[px[j], py[j]] for j in range(i, i+lens)] for i in trajectory_idx], dtype=torch.float32)
prev_trajectory = prev_trajectory.view(prev_trajectory.shape[0], -1)
future_trajectory = future_trajectory.view(future_trajectory.shape[0], -1)
print("Length of previous trajectory: ", len(prev_trajectory[0]))
print("Length of future trajectory: ", len(future_trajectory[0]))

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
        self.encoder_net = self.encoder(input_size, hidden_size, output_size)
        self.decoder_net = self.decoder(input_size, hidden_size, output_size)

    def forward(self, x):
        z = self.encoder_net(x)
        x_hat = self.decoder_net(z)
        return x_hat

    def encoder(self, input_dim, hidden_size, z_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, z_dim),
        )

    def decoder(self, output_dim, hidden_size, z_dim):
        return nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.Linear(hidden_size, hidden_size),
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
input_size = lens * 2
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
np.savez("normalization.npz", mu_x=mu_x, sigma_x=sigma_x, mu_y=mu_y, sigma_y=sigma_y)
print("Model Saved")
