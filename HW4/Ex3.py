import numpy as np
import matplotlib.pyplot as plt


lam = -0.6
eta = 0.5
T_max = 10.0
def leapfrog_solve(lam, eta, k, T_max):
    N = int(T_max / k)
    t = np.linspace(0, T_max, N + 1)
    U = np.zeros(N + 1)

    # IV
    U[0] = eta
    U[1] = eta * (1 + k * lam)  # Forward Euler

    # Leapfrog
    for n in range(1, N):
        U[n + 1] = U[n - 1] + 2 * k * lam * U[n]

    return t, U


t1, U1 = leapfrog_solve(lam, eta, 0.5, T_max)
t2, U2 = leapfrog_solve(lam, eta, 0.25, T_max)

# exact solution
t_exact = np.linspace(0, T_max, 500)
U_exact = eta * np.exp(lam * t_exact)

# Visual
plt.figure(figsize=(10, 6))
plt.plot(t_exact, U_exact, 'k-', linewidth=2, label='True Solution (Exact)')
plt.plot(t1, U1, 'r--', label='Leapfrog (k = 0.5)')
plt.plot(t2, U2, 'b-.', label='Leapfrog (k = 0.25)')

plt.title('Convergence vs. Absolute Instability in Leapfrog Method')
plt.xlabel('Time (t)')
plt.ylabel('U(t)')
plt.ylim(-2, 2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()