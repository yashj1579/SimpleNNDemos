import scipy
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def pendulum_ode(x, t, u=0.0, m=1.0, l=1.0, g=9.8, b = 0.0):
    """
    Given state vector x, determine dynamics equation of a simple pendulum machine
    :param x: State vector [theta, theta_dot]
    :param u: input to system
    :return: Dynamics equation vector of x_dot [theta_dot, theta_dot_dot]
    """
    x_dot = np.array([x[1], -g / l * np.sin(x[0]) - b / m / l ** 2 * x[1]])
    return x_dot


def total_energy(x, m=1.0, l=1.0, g=9.8):
    E = 1 / 2 * m * l ** 2 * x[:, 1] ** 2 + m * g * l * (1 - np.cos(x[:, 0]))
    return E


def exact_integration(x_init, t, u=0.0, m=1.0, l=1.0, g=9.8, b=0.0):
    x = scipy.integrate.odeint(pendulum_ode, x_init, t, args=(u,m,l,g,b))
    return x


def sample_case():
    print("Collecting Data")
    m = 1.0;
    l = 1.0;
    g = 9.8
    b = 0.2;
    u = 0.0;
    t = np.linspace(0, 10, 1000)
    x_init = np.array([0, 2.0])  # I.C.
    x = exact_integration(x_init, t, u=u, m=m, l=l, g=g, b=b)
    px, py = l * np.sin(x[:, 0]), -l * np.cos(x[:, 0])
    print("Plotting Figures")
    # print(x)
    """
    plt.figure()
    plt.plot(t, x)
    plt.legend(['Theta', 'Theta Dot'], loc='best')
    plt.title("Dynamics equation")
    plt.show()

    plt.figure()
    E = total_energy(x, m=1.0, l=1.0, g=9.8)
    #print(E)
    plt.plot(t, E)
    plt.yticks(np.arange(0, 0.2, 0.01))
    plt.title("Total Energy")
    plt.show()

    """

    # Animate
    print("Animating")
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(-1.2 * l, 1.2 * l)
    ax.set_ylim(-1.2 * l, 1.2 * l)
    ax.set_title("Pendulum Animation")
    rod, = ax.plot([], [], lw=2, color="black")
    bob, = ax.plot([], [], "o", markersize=12, color="red")
    pivot, = ax.plot(0, 0, "ko")

    def init():
        rod.set_data([], [])
        bob.set_data([], [])
        return rod, bob

    def update(frame):
        xdata = [0, px[frame]]
        ydata = [0, py[frame]]
        rod.set_data(xdata, ydata)
        bob.set_data([px[frame]], [py[frame]])
        return rod, bob

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=100, blit=False)
    ani.save("pendulum.mp4", fps=60, dpi=150)
    print("Animation Complete")

if __name__ == "__main__":
    sample_case()
