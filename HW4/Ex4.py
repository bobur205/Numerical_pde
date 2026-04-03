import numpy as np
import matplotlib.pyplot as plt

def solve_sir_2eq(beta, gamma, x0, y0, k, T_max):
    N = int(T_max / k)
    t = np.linspace(0, T_max, N + 1)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    x[0] = x0
    y[0] = y0

    # Very essential part, calculating x and y
    for n in range(N):
        x[n + 1] = x[n] - k * beta * x[n] * y[n]
        y[n + 1] = y[n] + k * (beta * x[n] * y[n] - gamma * y[n])
    return t, x, y

def plot_single_case(beta, gamma, x0, y0, k, T_max, R0):
    t, x, y = solve_sir_2eq(beta, gamma, x0, y0, k, T_max)

    plt.figure(figsize=(8, 5))
    plt.plot(t, x, 'b-', linewidth=2, label='Susceptible (x)')
    plt.plot(t, y, 'r-', linewidth=2, label='Infected (y)')

    plt.title(fr'SIR Dynamics: $\beta = {beta}, \gamma = {gamma}$ ($R_0 = {R0}$)', fontsize=14)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Proportion of Population', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# MAin parametrs
x0 = 0.99
y0 = 0.01
gamma = 0.1
k = 0.1
T_max = 250

# visualizing every single case respect to beta,
plot_single_case(0.3, gamma, x0, y0, k, T_max, 3.0)  # Case 1
plot_single_case(0.15, gamma, x0, y0, k, 350, 1.5)  # Case 2
plot_single_case(0.05, gamma, x0, y0, k, 250, 0.5)  # Case 3