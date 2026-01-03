import numpy as np
import ExactODE as Exact
import matplotlib

normalization = lambda x, mu, sigma: (x - mu) / (sigma + 1e-6)

matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

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


print("Collecting Data")
m = 1.0; l = 1.0; g = 9.8
t = np.linspace(0, 10, 1000)
x_init = np.array([0, 1.0])  # I.C.
x = Exact.exact_integration(x_init, t, m, l, g)
px, py = l * np.sin(x[:, 0]), -l * np.cos(x[:, 0])

#loading model
print("Loading Model...")
lens = 10
input_size = lens * 2
hidden_size = input_size // 2
output_size = input_size
z_dim = 10
model = NeuralNetwork(input_size, hidden_size, z_dim, output_size)
model.load_state_dict(torch.load("pendulum_model.pth"))
model.eval()
model = model.to("cpu")
print("Model loaded")

print("Loading Normalization...")
norm_data = np.load("normalization.npz")
mu_x = norm_data["mu_x"]
mu_y = norm_data["mu_y"]
sigma_x = norm_data["sigma_x"]
sigma_y = norm_data["sigma_y"]
print("Normalization loaded")

# Animate
print("Animating")
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-1.2 * l, 1.2 * l)
ax.set_ylim(-1.2 * l, 1.2 * l)
ax.set_title("Pendulum Animation")
rod, = ax.plot([], [], lw=2, color="black")
bob, = ax.plot([], [], "o", markersize=12, color="red")
prev_points = [ax.plot([], [], "o", markersize=3-2*i/lens, color="orange", alpha=0.8)[0] for i in range(lens)]
pred_points = [ax.plot([], [], "o", markersize=1+4*i/lens, color="purple")[0] for i in range(lens)]
actual_points = [ax.plot([], [], "o", markersize=1+4*i/lens, color="green", alpha=0.1)[0] for i in range(lens)]
pivot, = ax.plot(0, 0, "ko")

def init():
    rod.set_data([], [])
    bob.set_data([], [])
    for i in range(lens):
        prev_points[i].set_data([], [])
        actual_points[i].set_data([], [])
        pred_points[i].set_data([], [])
    return [rod, bob] + prev_points + actual_points + pred_points

def update(frame):
    xdata = [0, px[frame]]
    ydata = [0, py[frame]]
    rod.set_data(xdata, ydata)
    bob.set_data([px[frame]], [py[frame]])
    if frame >= lens and frame < len(px) - lens:
        prev_trajectory = torch.tensor([[normalization(px[j], mu_x, sigma_x), normalization(py[j], mu_y, sigma_y)] for j in range(frame - lens, frame)], dtype=torch.float32)
        future_trajectory = torch.tensor([[normalization(px[j], mu_x, sigma_x), normalization(py[j], mu_y, sigma_y)] for j in range(frame, frame+lens)], dtype=torch.float32)
        prev_trajectory = prev_trajectory.view(1, -1)
        prev_trajectory = prev_trajectory.to(device)
        pred_traj = model(prev_trajectory)
        pred_traj = pred_traj.detach().numpy().flatten()
        pred_traj = np.array([[pred_traj[2*i] * (sigma_x + 1e-6) + mu_x, pred_traj[2*i+1] * (sigma_y + 1e-6) + mu_y] for i in range(lens)])
        #print(nn.MSELoss(pred_traj, future_trajectory))

        for i in range(lens):
            prev_points[i].set_data([px[frame - lens + i]], [py[frame - lens + i]])
            actual_points[i].set_data([px[frame + i]], [py[frame + i]])
            pred_points[i].set_data([pred_traj[i][0]], [pred_traj[i][1]])

    return [rod, bob] + prev_points + actual_points + pred_points


ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=100, blit=False)
ani.save("NN_pendulum.mp4", fps=60, dpi=150)
