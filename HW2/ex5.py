import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Helper functions (from previous exercises)
# ---------------------------------------------------------
def coarsen(f):
    return f[1::2]


def interpolate(f, alpha, beta):
    m_coarse = len(f)
    m_fine = 2 * m_coarse + 1
    f_interp = np.zeros(m_fine)
    f_interp[1::2] = f
    f_interp[2:-1:2] = 0.5 * (f[:-1] + f[1:])
    f_interp[0] = 0.5 * (f_interp[1] + alpha)
    f_interp[-1] = 0.5 * (f_interp[-2] + beta)
    return f_interp


def Jacobi(U, f, alpha, beta, m, nu, omega=2. / 3):
    h = 1. / (m + 1)
    F = 0.5 * h ** 2 * f.copy()
    F[0] -= alpha / 2.
    F[-1] -= beta / 2.
    e = np.ones(m - 1)
    G = 0.5 * (np.diag(e, -1) + np.diag(e, 1))
    for i in range(nu):
        U = (1. - omega) * U + omega * (np.dot(G, U) - F)
    A = 2. / h ** 2 * (G - np.eye(m))
    FF = f.copy()
    FF[0] -= alpha / h ** 2
    FF[-1] -= beta / h ** 2
    rr = FF - np.dot(A, U)
    return U, rr


# ---------------------------------------------------------
# Standalone V-cycle Function
# ---------------------------------------------------------
def V_cycle(U, F, alpha, beta, k, nu, omega=2. / 3):
    m = 2 ** k - 1
    rdep = k - 1

    # Downward phase
    U, rr = Jacobi(U, F, alpha, beta, m, nu, omega)
    r = [None] * (rdep + 1)
    error = [None] * (rdep + 1)
    r[0] = rr

    m_temp = m
    for i in range(1, rdep + 1):
        m_temp = (m_temp - 1) // 2
        r[i] = coarsen(rr)
        # Residual equations have homogeneous (0) boundary conditions
        error[i], rr = Jacobi(np.zeros(m_temp), -r[i], 0., 0., m_temp, nu, omega)

    # Upward phase
    for i in range(1, rdep):
        m_temp = 2 * m_temp + 1
        err = error[rdep - i] - interpolate(error[rdep + 1 - i], 0, 0)
        error[-i - 1], rr = Jacobi(err, -r[rdep - i], 0., 0., m_temp, nu, omega)

    # Final correction
    U = U - interpolate(error[1], 0, 0)
    U, rr = Jacobi(U, F, alpha, beta, m, nu, omega)

    return U


# ---------------------------------------------------------
# Exact analytical function for RHS formulation
# ---------------------------------------------------------
def rhs_func(x):
    phi = lambda x: 20. * np.pi * x ** 3
    return -20 + 0.5 * 120 * np.pi * x * np.cos(phi(x)) - 0.5 * (60 * np.pi * x ** 2) ** 2 * np.sin(phi(x))


# ---------------------------------------------------------
# Full Multigrid (FMG) Algorithm
# ---------------------------------------------------------
def Full_Multigrid(k_fine, nu, omega=2. / 3):
    alpha = 1.0
    beta = 3.0

    # Step 1: Start at the coarsest grid (k=1)
    k = 1
    m = 2 ** k - 1
    x = np.linspace(0, 1, m + 2)[1:-1]
    F = rhs_func(x)

    # Initial naive guess for the coarsest grid
    U = np.linspace(alpha, beta, m)
    # Solve exactly (or use enough Jacobi iterations) on the coarsest grid
    U, _ = Jacobi(U, F, alpha, beta, m, nu=50, omega=omega)

    # Step 2: Iterate upwards to the finest grid
    for current_k in range(2, k_fine + 1):
        m = 2 ** current_k - 1
        x = np.linspace(0, 1, m + 2)[1:-1]
        F = rhs_func(x)

        # Interpolate the solution from the coarser grid to use as a prime initial guess
        U = interpolate(U, alpha, beta)

        # Run a single V-cycle on the current grid level
        U = V_cycle(U, F, alpha, beta, current_k, nu, omega)

    return U


# ---------------------------------------------------------
# Execute and Plot
# ---------------------------------------------------------
k_fine = 8
nu = 2
U_fmg = Full_Multigrid(k_fine, nu)

m_fine = 2 ** k_fine - 1
x_fine = np.linspace(0, 1, m_fine + 2)[1:-1]
phi = lambda x: 20. * np.pi * x ** 3
u_exact = 1. + 12. * x_fine - 10. * x_fine ** 2 + 0.5 * np.sin(phi(x_fine))

error = np.max(np.abs(U_fmg - u_exact))
print(f"Max absolute error with FMG (k={k_fine}, nu={nu}): {error:.4e}")

plt.figure(figsize=(10, 6))
plt.plot(x_fine, u_exact, 'k-', linewidth=2.5, label='Exact Solution')
plt.plot(x_fine, U_fmg, 'r--', linewidth=2, label=f'FMG Solution ($\\nu$={nu})')
plt.title(f'Full Multigrid (FMG) Solution vs Exact (k={k_fine})')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True, linestyle=':')
plt.show()