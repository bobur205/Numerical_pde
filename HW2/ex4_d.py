import numpy as np
import matplotlib.pyplot as plt


# Helper functions coarsen, interpolate, Jacobi as defined previously

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


def Vcycle_error(k, nu, depth=None, omega=2. / 3):
    """Run one V-cycle with given recursion depth.
       If depth is None, use full depth (k-1)."""
    if depth is None:
        depth = k - 1
    else:
        depth = min(depth, k - 1)  # cannot exceed full depth

    m = 2 ** k - 1
    rdep = depth  # number of coarsening steps
    alpha = 1.;
    beta = 3.
    U = np.linspace(alpha, beta, m)
    x = np.linspace(0, 1, m + 2)[1:-1]
    phi = lambda x: 20. * np.pi * x ** 3
    u_exact = 1. + 12. * x - 10. * x ** 2 + 0.5 * np.sin(phi(x))
    F = -20 + 0.5 * 120 * np.pi * x * np.cos(phi(x)) - 0.5 * (60 * np.pi * x ** 2) ** 2 * np.sin(phi(x))

    # Downward leg
    U, rr = Jacobi(U, F, alpha, beta, m, nu, omega)
    r = [None] * (rdep + 1);
    error = [None] * (rdep + 1)
    r[0] = rr
    for i in range(1, rdep + 1):
        m = (m - 1) // 2
        r[i] = coarsen(rr)
        error[i], rr = Jacobi(np.zeros(m), -r[i], 0., 0., m, nu, omega)

    # Upward leg
    for i in range(1, rdep):
        m = 2 * m + 1
        err = error[rdep - i] - interpolate(error[rdep + 1 - i], 0, 0)
        # Always smooth after interpolation (as in original)
        error[-i - 1], rr = Jacobi(err, -r[rdep - i], 0., 0., m, nu, omega)

    # Final correction on finest grid
    m = 2 * m + 1
    U = U - interpolate(error[1], 0, 0)
    U, rr = Jacobi(U, F, alpha, beta, m, nu, omega)

    return np.max(np.abs(U - u_exact))


# Experiment
k = 8
nu = 4
depths = range(1, k)  # 1 to 7
errors = [Vcycle_error(k, nu, depth=d) for d in depths]

plt.figure(figsize=(8, 5))
plt.plot(depths, errors, 'o-', linewidth=2, markersize=8)
plt.xlabel('Recursion depth (number of coarsening steps)')
plt.ylabel('Maximum absolute error')
plt.title(f'Effect of recursion depth (k={k}, $\\nu$={nu})')
plt.grid(True)
plt.show()