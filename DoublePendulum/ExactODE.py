import scipy
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5 installed
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def pendulum_ode(x, t, u=0.0, m1=1.0, l1=1.0, m2=1.0, l2=1.0, g=9.8):
    """
    Given state vector x, determine dynamics equation of a simple pendulum machine
    :param x: State vector [theta, theta_dot]
    :param u: input to system
    :return: Dynamics equation vector of x_dot [theta_dot, theta_dot_dot]
    """
    delta = x[2] - x[0]
    x_dot = np.array([x[1], (m2 * l1 * x[1] ** 2 * np.sin(delta) * np.cos(delta) + m2 * g * np.sin(x[2]) * np.cos(delta) + m2 * l2 * x[3] ** 2 * np.sin(delta) - (m1 + m2) * g * np.sin(x[0])) / (l1 * (m1 + m2 * np.sin(delta) ** 2)), x[3], (-m2 * l2 * x[3] ** 2 * np.sin(delta) * np.cos(delta) + (m1+m2)*g*np.sin(x[0])*np.cos(delta) - (m1 + m2) * l1 * x[1] ** 2 * np.sin(delta) - (m1 + m2) * g * np.sin(x[2])) / (l2 * (m1 + m2 * np.sin(delta) ** 2))])
    return x_dot


def total_energy(x, m1=1.0, l1=1.0, m2=1.0, l2=1.0, g=9.8):
    E = 1 / 2 * (m1+m2) * l1 ** 2 * x[:, 1] ** 2 - (m1 + m2) * g * l1 * np.cos(x[:, 0]) + 1 / 2 * m2 * l2 ** 2 * x[:, 3] ** 2 - m2 * g * l2 * np.cos(x[:, 2]) + m2 * l1 * l2 * x[:, 1] * x[:, 3] * np.cos(x[:, 0] - x[:, 2])
    return E


def exact_integration(x_init, t, u=0.0, m1=1.0, l1=1.0, m2=1.0, l2=1.0, g=9.8):
    x = scipy.integrate.odeint(pendulum_ode, x_init, t, args=(u, m1, l1, m2, l2, g))
    return x


def sample_case():
    print("Collecting Data")
    m1 = 1.0; l1 = 1.0; m2 = 1.0; l2 = 1.0; u = 0.0
    g = 9.8
    t = np.linspace(0, 10, 1000)
    x_init = np.array([0, 1.0, np.pi / 4, 2.0])  # I.C.
    x = exact_integration(x_init, t, u, m1, l1, m2, l2, g)
    px1, py1 = l1 * np.sin(x[:, 0]), -l1 * np.cos(x[:, 0])
    px2, py2 = px1 + l2 * np.sin(x[:, 2]), py1 - l2 * np.cos(x[:, 2])

    """
    print("Plotting Figures")
    # print(x)

    plt.figure()
    plt.plot(t, x)
    plt.legend(['Theta 1', 'Theta Dot 1', 'Theta 2', 'Theta Dot 2'], loc='best')
    plt.title("Dynamics equation")
    plt.show()

    plt.figure()
    E = total_energy(x, m1, l1, m2, l2, g)
    #print(E)
    plt.plot(t, E)
    plt.yticks(np.linspace(E[0] - 0.5, E[0] + 0.5, 5))
    plt.title("Total Energy")
    plt.show()
    """


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
    pivot, = ax.plot(0, 0, "ko")

    def init():
        rod1.set_data([], [])
        rod2.set_data([], [])
        bob1.set_data([], [])
        bob2.set_data([], [])
        return rod1, rod2, bob1, bob2

    def update(frame):
        xdata1 = [0, px1[frame]]
        ydata1 = [0, py1[frame]]
        xdata2 = [px1[frame], px2[frame]]
        ydata2 = [py1[frame], py2[frame]]
        rod1.set_data(xdata1, ydata1)
        rod2.set_data(xdata2, ydata2)
        bob1.set_data([px1[frame]], [py1[frame]])
        bob2.set_data([px2[frame]], [py2[frame]])
        return rod1, rod2, bob1, bob2


    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=100, blit=False)
    ani.save("double_pendulum.mp4", fps=60, dpi=150)


