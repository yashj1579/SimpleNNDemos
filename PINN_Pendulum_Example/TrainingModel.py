import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ExactODE as Exact
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from random import random

import os

frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

print("Generating Data")
time = []
ic = []
position = []
lens = 10
for types in [[random() * 2 - 1, random() * 3 - 1] for i in range(1000)]:
    m = 1.0; l = 1.0; g = 9.8; b = 0.2
    u = 0.0
    t = np.linspace(0, 5, 150)
    x_init = np.array(types)  # I.C.
    x = Exact.exact_integration(x_init, t, u, m, l, g)

    lens = 10
    trajectory_idx = range(len(t))
    time.append([[t[i]] for i in trajectory_idx])
    ic.append([[x_init[0], x_init[1]] for i in trajectory_idx])
    position.append([[x[i, 0]] for i in trajectory_idx])
time = torch.tensor(time, dtype=torch.float32, requires_grad=True)
position = torch.tensor(position, dtype=torch.float32)
ic = torch.tensor(ic, dtype=torch.float32)
time = time.reshape(-1, 1)
ic = ic.reshape(-1, 2)
position = position.reshape(-1, 1)
print("Length of time: ", time.shape)
print("Length of ic: ", ic.shape)
print("Length of position: ", position.shape)

print("Batching data")
train_percentage = 0.8
train_time = time[:int (train_percentage*len(time))]
train_ic = ic[:int (train_percentage*len(time))]
train_position = position[:int (train_percentage*len(position))]

test_time = time[int (train_percentage*len(time)):]
test_ic = ic[int (train_percentage*len(time)):]
test_position = position[int (train_percentage*len(position)):]

batch_size = 16
train_dataset = TensorDataset(train_time, train_ic, train_position)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_time, test_ic, test_position)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.encoder_net = self.NN_Arch(input_size, hidden_size, output_size)

    def forward(self, t, ic):
        x = torch.cat((t, ic), dim=1)
        return self.encoder_net(x)

    def NN_Arch(self, input_dim, hidden_size, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_dim),
        )

def pinn_residual_from_batch(model, t, ic, b, g=9.8, l=1.0):
    x = torch.cat([t, ic], dim=1).requires_grad_(True)
    u = model.encoder_net(x)  # [B,1]

    grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = grads[:, 0:1]
    u_tt = torch.autograd.grad(u_t.sum(), x, create_graph=True)[0][:, 0:1]

    res = u_tt + b * u_t + (g / l) * torch.sin(u)
    return u, res

def train(models, lambdas, optimizers, train_loader, test_loader, bs, names, num_epochs=200,
          t_collocation_max=20.0, collocation_per_batch=128):

    train_loss_vals = [[] for _ in models]
    test_loss_vals = [[] for _ in models]

    t_plot = np.linspace(0, 20, 600)
    x_init_plot = np.array([0.0, 3.0])

    for epoch in range(num_epochs):
        print("Epoch", epoch)

        for model_idx, model in enumerate(models):
            model.train()
            total_train_loss = 0.0

            for t_data, ic_data, u_data in train_loader:
                t_data = t_data.to(device)
                ic_data = ic_data.to(device)
                u_data = u_data.to(device)

                optimizers[model_idx].zero_grad()

                u_pred = model(t_data, ic_data)
                data_loss = ((u_pred - u_data) ** 2).mean()

                lam = lambdas[model_idx]
                if lam > 0.0:
                    t_col = torch.rand(collocation_per_batch, 1, device=device) * t_collocation_max

                    idx = torch.randint(0, ic_data.shape[0], (collocation_per_batch,), device=device)
                    ic_col = ic_data[idx]

                    _, res_col = pinn_residual_from_batch(model, t_col, ic_col, bs[model_idx])
                    phys_loss = (res_col ** 2).mean()
                    loss = data_loss + lam * phys_loss
                else:
                    loss = data_loss

                loss.backward()
                optimizers[model_idx].step()
                total_train_loss += loss.item()

            total_train_loss /= len(train_loader)
            train_loss_vals[model_idx].append(total_train_loss)

            # test loss
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for t_data, ic_data, u_data in test_loader:
                    t_data = t_data.to(device)
                    ic_data = ic_data.to(device)
                    u_data = u_data.to(device)
                    u_pred = model(t_data, ic_data)
                    total_test_loss += ((u_pred - u_data) ** 2).mean().item()
            total_test_loss /= len(test_loader)
            test_loss_vals[model_idx].append(total_test_loss)

        if epoch % 10 == 0:
            m, l, g, b_true, u0 = 1.0, 1.0, 9.8, 0.2, 0.0
            x_exact = Exact.exact_integration(x_init_plot, t_plot, u0, m, l, g, b_true)[:, 0]

            fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
            if len(models) == 1:
                axes = [axes]

            for ax, model_idx in zip(axes, range(len(models))):
                t_tensor = torch.tensor(t_plot[:, None], dtype=torch.float32, device=device)
                ic_tensor = torch.tensor(x_init_plot, dtype=torch.float32, device=device).repeat(len(t_plot), 1)

                with torch.no_grad():
                    pred = models[model_idx](t_tensor, ic_tensor).cpu().numpy().flatten()

                ax.plot(t_plot, x_exact, label="Exact")
                ax.plot(t_plot, pred, label="Predicted")
                ax.set_title(f"{names[model_idx]} epoch {epoch} b={bs[model_idx].item():.3f}")
                ax.legend()

            frame_path = os.path.join(frames_dir, f"frame_{epoch:04d}.png")
            plt.tight_layout()
            plt.savefig(frame_path, dpi=120)
            plt.close(fig)

    plt.figure()
    for j in range(len(models)):
        plt.plot(train_loss_vals[j], label=f"{names[j]} train")
        plt.plot(test_loss_vals[j], label=f"{names[j]} test")
    plt.legend()
    plt.savefig("loss_curve.png")


print("Creating model")
lr = 1e-4
opt = optim.Adam
num_epochs = 200
input_size = 3
hidden_size = 50
output_size = 1
z_dim = 10
model1 = NeuralNetwork(input_size, hidden_size, output_size)
model1.to(device)
b1 = torch.nn.Parameter(torch.tensor(1.0, device=device))
optimizer1 = torch.optim.Adam(list(model1.parameters()) + [b1], lr=1e-3)

model2 = NeuralNetwork(input_size, hidden_size, output_size)
model2.to(device)
b2 = torch.nn.Parameter(torch.tensor(1.0, device=device))
optimizer2 = torch.optim.Adam(list(model2.parameters()) + [b2], lr=1e-3)

models = [model1, model2]
bs = [b1, b2]
optimizers = [optimizer1, optimizer2]
ls = [10.0, 0.0]
name = ["PINN", "Basic NN"]

print("Training Model...")
train(models, ls, optimizers, train_loader, test_loader, bs, name, num_epochs)
print("Model Trained")



#save model and normalization
print("Saving model...")
torch.save(models[0].state_dict(), "pinn_pendulum_model.pth")
