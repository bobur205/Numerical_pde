import numpy as np
import matplotlib.pyplot as plt


# Helper functions (coarsen, interpolate, Jacobi) - unchanged
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


def Vcycle_error(k, nu, omega=2. / 3, smooth_after_interp=True):
    m = 2 ** k - 1
    rdep = k - 1
    alpha = 1.
    beta = 3.
    U = np.linspace(alpha, beta, m)
    x = np.linspace(0, 1, m + 2)[1:-1]
    phi = lambda x: 20. * np.pi * x ** 3
    u_exact = 1. + 12. * x - 10. * x ** 2 + 0.5 * np.sin(phi(x))
    F = -20 + 0.5 * 120 * np.pi * x * np.cos(phi(x)) - 0.5 * (60 * np.pi * x ** 2) ** 2 * np.sin(phi(x))

    # Downward
    U, rr = Jacobi(U, F, alpha, beta, m, nu, omega)
    r = [None] * (rdep + 1)
    error = [None] * (rdep + 1)
    r[0] = rr
    for i in range(1, rdep + 1):
        m = (m - 1) // 2
        r[i] = coarsen(rr)
        error[i], rr = Jacobi(np.zeros(m), -r[i], 0., 0., m, nu, omega)

    # Upward
    for i in range(1, rdep):
        m = 2 * m + 1
        err = error[rdep - i] - interpolate(error[rdep + 1 - i], 0, 0)
        if smooth_after_interp:
            error[-i - 1], rr = Jacobi(err, -r[rdep - i], 0., 0., m, nu, omega)
        else:
            error[-i - 1] = err

    # Final correction
    m = 2 * m + 1
    U = U - interpolate(error[1], 0, 0)
    U, rr = Jacobi(U, F, alpha, beta, m, nu, omega)

    return np.max(np.abs(U - u_exact))


# -------------------- Experiment --------------------
k_values = [6, 8]
nu_range = range(1, 11)
omega_vals = [2. / 3, 1.0]
results = {}

for k in k_values:
    for omega in omega_vals:
        errors = []
        for nu in nu_range:
            e = Vcycle_error(k, nu, omega=omega)  # smooth_after_interp=True by default
            errors.append(e)
        results[(k, omega)] = errors

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, k in enumerate(k_values):
    ax = axes[idx]
    ax.semilogy(nu_range, results[(k, 2. / 3)], 'o-', label=r'$\omega=2/3$')
    ax.semilogy(nu_range, results[(k, 1.0)], 's-', label=r'$\omega=1$')
    ax.set_xlabel('Number of Jacobi iterations $\\nu$')
    ax.set_ylabel('Max absolute error')
    ax.set_title(f'k = {k} ({2 ** k - 1} points)')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()