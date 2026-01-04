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


print("Collecting Data")
m1 = 1.0; l1 = 1.0; m2 = 1.0; l2 = 1.0; g = 9.8
u = 0.0
t = np.linspace(0, 10, 1000)
x_init = np.array([0, 1.0, np.pi / 4, 2.0])  # I.C.
x = Exact.exact_integration(x_init, t, u, m1, l1, m2, l2, g)
px1, py1 = l1 * np.sin(x[:, 0]), -l1 * np.cos(x[:, 0])
px2, py2 = px1 + l2 * np.sin(x[:, 2]), py1 - l2 * np.cos(x[:, 2])

#loading model
print("Loading Model...")
lens = 10
input_size = lens * 4
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
mu_x1 = norm_data["mu_x1"]
mu_y1 = norm_data["mu_y1"]
sigma_x1 = norm_data["sigma_x1"]
sigma_y1 = norm_data["sigma_y1"]
mu_x2 = norm_data["mu_x2"]
mu_y2 = norm_data["mu_y2"]
sigma_x2 = norm_data["sigma_x2"]
sigma_y2 = norm_data["sigma_y2"]
print("Normalization loaded")

# Animate
print("Animating")
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-1.2 * (l1 + l2), 1.2 * (l1+l2))
ax.set_ylim(-1.2 * (l1 + l2), 1.2 * (l1 + l2))
ax.set_title("Pendulum Animation")
rod1, = ax.plot([], [], lw=2, color="black")
rod2, = ax.plot([], [], lw=2, color="black")
bob1, = ax.plot([], [], "o", markersize=12, color="red")
bob2, = ax.plot([], [], "o", markersize=12, color="red")
prev_points1 = [ax.plot([], [], "o", markersize=3-2*i/lens, color="orange", alpha=0.8)[0] for i in range(lens*2)]
pred_points1 = [ax.plot([], [], "o", markersize=1+4*i/lens, color="purple")[0] for i in range(lens*2)]
actual_points1 = [ax.plot([], [], "o", markersize=1+4*i/lens, color="green", alpha=0.1)[0] for i in range(lens*2)]
prev_points2 = [ax.plot([], [], "o", markersize=3-2*i/lens, color="orange", alpha=0.8)[0] for i in range(lens*2)]
pred_points2 = [ax.plot([], [], "o", markersize=1+4*i/lens, color="purple")[0] for i in range(lens*2)]
actual_points2 = [ax.plot([], [], "o", markersize=1+4*i/lens, color="green", alpha=0.1)[0] for i in range(lens*2)]

pivot, = ax.plot(0, 0, "ko")

def init():
    rod1.set_data([], [])
    rod2.set_data([], [])
    bob1.set_data([], [])
    bob2.set_data([], [])
    for i in range(lens):
        prev_points1[i].set_data([], [])
        actual_points1[i].set_data([], [])
        pred_points1[i].set_data([], [])
        prev_points2[i].set_data([], [])
        actual_points2[i].set_data([], [])
        pred_points2[i].set_data([], [])
    return [rod1, rod2, bob1, bob2] + prev_points1 + actual_points1 + pred_points1 + prev_points2 + actual_points2 + pred_points2

def update(frame):
    xdata1 = [0, px1[frame]]
    ydata1 = [0, py1[frame]]
    xdata2 = [px1[frame], px2[frame]]
    ydata2 = [py1[frame], py2[frame]]

    rod1.set_data(xdata1, ydata1)
    rod2.set_data(xdata2, ydata2)
    bob1.set_data([px1[frame]], [py1[frame]])
    bob2.set_data([px2[frame]], [py2[frame]])
    if frame >= lens and frame < len(px1) - lens:
        prev_trajectory = torch.tensor([[normalization(px1[j], mu_x1, sigma_x1), normalization(py1[j], mu_y1, sigma_y1), normalization(px2[j], mu_x2, sigma_x2), normalization(py2[j], mu_y2, sigma_y2)] for j in range(frame - lens, frame)], dtype=torch.float32)
        prev_trajectory = prev_trajectory.view(1, -1)
        prev_trajectory = prev_trajectory.to(device)
        pred_traj = model(prev_trajectory)
        pred_traj = pred_traj.detach().numpy().flatten()
        pred_traj = np.array([[pred_traj[4*i] * (sigma_x1 + 1e-6) + mu_x1, pred_traj[4*i+1] * (sigma_y1 + 1e-6) + mu_y1, pred_traj[4*i+2] * (sigma_x2 + 1e-6) + mu_x2, pred_traj[4*i+3] * (sigma_y2 + 1e-6) + mu_y2] for i in range(lens)])
        #print(nn.MSELoss(pred_traj, future_trajectory))

        for i in range(lens):
            prev_points1[i].set_data([px1[frame - lens + i]], [py1[frame - lens + i]])
            prev_points2[i].set_data([px2[frame - lens + i]], [py2[frame - lens + i]])
            actual_points1[i].set_data([px1[frame + i]], [py1[frame + i]])
            actual_points2[i].set_data([px2[frame + i]], [py2[frame + i]])
            pred_points1[i].set_data([pred_traj[i][0]], [pred_traj[i][1]])
            pred_points2[i].set_data([pred_traj[i][2]], [pred_traj[i][3]])

    return [rod1, rod2, bob1, bob2] + prev_points1 + actual_points1 + pred_points1 + prev_points2 + actual_points2 + pred_points2


ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=100, blit=False)
ani.save("NN_double_pendulum.mp4", fps=60, dpi=150)
