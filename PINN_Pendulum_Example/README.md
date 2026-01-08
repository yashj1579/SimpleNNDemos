# PiNN Pendulum: Learning Pendulum Dynamics with Physics-Informed Neural Networks

## Project Overview

This project demonstrates how **Physics-Informed Neural Networks (PINNs)** can learn the dynamics of a simple pendulum **while explicitly respecting the underlying physics**. Unlike purely data-driven models, a PINN embeds the governing equations of motion directly into the training process, allowing the model to:

* Learn from limited data
* Generalize better to unseen trajectories
* Identify unknown physical parameters (such as damping)

In this project, the PINN is trained to model pendulum motion and **infer the damping coefficient `b`** directly from data.

---

## What Is a Physics-Informed Neural Network (PINN)?

A **PINN** is a neural network trained not only to fit observed data, but also to satisfy known physical laws, typically expressed as **differential equations**.

### Traditional Neural Network

* Learns patterns purely from data
* May violate conservation laws or physical constraints
* Requires large datasets to generalize well

### Physics-Informed Neural Network

* Learns a function ( x(t) ) that approximates the system state
* Uses **automatic differentiation** to compute time derivatives
* Penalizes violations of the governing ODEs in the loss function

This makes the network behave like a **soft numerical solver** that respects physics.

---

## Pendulum Dynamics

The motion of a damped pendulum is governed by the second-order ODE:

[
\ddot{\theta}(t) + b\dot{\theta}(t) + \frac{g}{L} \sin(\theta(t)) = 0
]

Where:

* ( \theta(t) ): pendulum angle
* ( \dot{\theta}(t) ): angular velocity
* ( b ): **damping coefficient** (unknown)
* ( g ): gravity
* ( L ): pendulum length

The goal of the PINN is to:

1. Learn ( \theta(t) )
2. Infer the unknown parameter **( b )** from data

---

## How the PINN Works in This Project

### Neural Network Model

The neural network takes time ( t ) as input and outputs:

[
\theta(t), \quad \dot{\theta}(t)
]

Using PyTorch’s automatic differentiation, the model computes:

* ( \dot{\theta}(t) )
* ( \ddot{\theta}(t) )

These derivatives are used to evaluate the pendulum ODE **inside the loss function**.

---

### Loss Function Structure

The total loss consists of two main parts:

#### 1. Data Loss

Ensures the network matches observed trajectories:

[
\mathcal{L}*{data} = | \theta*{pred} - \theta_{true} |^2
]

#### 2. Physics Loss

Penalizes violations of the pendulum equation:

[
\mathcal{L}_{physics} = \left| \ddot{\theta} + b\dot{\theta} + \frac{g}{L} \sin(\theta) \right|^2
]

#### Total Loss

[
\mathcal{L} = \mathcal{L}*{data} + \lambda \mathcal{L}*{physics}
]

This forces the network to learn **physically consistent dynamics**.

---

## Learning the Damping Parameter `b`

A key feature of this project is that **`b` is not fixed**.

Instead:

* `b` is defined as a **trainable parameter** in PyTorch
* Gradients flow through the physics loss
* The optimizer updates `b` alongside the network weights

```python
self.b = torch.nn.Parameter(torch.tensor(0.1))
```

During training, the PINN automatically adjusts `b` to minimize ODE residuals.

### Why This Works

* Incorrect `b` causes systematic physics violations
* The physics loss pushes `b` toward its true physical value
* No explicit supervision on `b` is required

This is **parameter identification**, not just prediction.

---

## Files Overview

| File               | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| `ExactODE.py`      | Generates ground-truth pendulum trajectories via numerical integration     |
| `TrainingModel.py` | Defines and trains the PINN, including physics loss and parameter learning |

---

## Educational Value

This project demonstrates:

* How differential equations can be embedded into neural networks
* How PINNs differ from black-box ML models
* How unknown physical parameters can be learned from data
* The power of automatic differentiation in scientific ML

---

## Potential Extensions

* Learn multiple parameters (e.g., gravity, length)
* Extend to double pendulum dynamics
* Add energy-conservation regularization
* Compare PINNs vs purely data-driven models
* Use neural ODEs or Hamiltonian neural networks

---

## Summary

The PiNN Pendulum project shows how neural networks can go beyond curve fitting to **reason about physics**. By embedding the pendulum’s governing equations into the training process, the model learns accurate trajectories and uncovers hidden physical parameters like damping — bridging the gap between machine learning and classical mechanics.
