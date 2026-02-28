import numpy as np
import matplotlib.pyplot as plt

# ---------- Helper functions (unchanged) ----------
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

def Jacobi(U, f, alpha, beta, m, nu, omega=2./3):
    h = 1./(m+1)
    F = 0.5 * h**2 * f.copy()
    F[0] -= alpha/2.
    F[-1] -= beta/2.
    e = np.ones(m-1)
    G = 0.5 * (np.diag(e, -1) + np.diag(e, 1))
    for i in range(nu):
        U = (1.-omega)*U + omega*(np.dot(G, U) - F)
    A = 2./h**2 * (G - np.eye(m))
    FF = f.copy()
    FF[0] -= alpha/h**2
    FF[-1] -= beta/h**2
    rr = FF - np.dot(A, U)
    return U, rr

# ---------- Modified V‑cycle that returns error ----------
def Vcycle_error(k, nu, omega=2./3, smooth_after_interp=True):
    """Run one V‑cycle and return max absolute error."""
    m = 2**k - 1
    rdep = k - 1               # full V‑cycle depth
    alpha = 1.
    beta = 3.

    # Initial guess (linear between boundaries)
    U = np.linspace(alpha, beta, m)

    # Grid and exact solution
    x = np.linspace(0, 1, m+2)[1:-1]
    phi = lambda x: 20. * np.pi * x**3
    u_exact = 1. + 12.*x - 10.*x**2 + 0.5*np.sin(phi(x))
    F = -20 + 0.5*120*np.pi*x*np.cos(phi(x)) - 0.5*(60*np.pi*x**2)**2 * np.sin(phi(x))

    # ----- Downward leg -----
    U, rr = Jacobi(U, F, alpha, beta, m, nu, omega)
    r = [None] * (rdep + 1)
    error = [None] * (rdep + 1)
    r[0] = rr   # store residual on finest grid (not used further)

    for i in range(1, rdep + 1):
        m = (m - 1) // 2                     # coarser grid size
        r[i] = coarsen(rr)                   # restrict residual
        # Solve error equation on coarse grid (starting from zero)
        error[i], rr = Jacobi(np.zeros(m), -r[i], 0., 0., m, nu, omega)

    # ----- Upward leg -----
    for i in range(1, rdep):
        m = 2 * m + 1                         # finer grid size
        err = error[rdep - i] - interpolate(error[rdep + 1 - i], 0, 0)
        if smooth_after_interp:
            error[-i-1], rr = Jacobi(err, -r[rdep - i], 0., 0., m, nu, omega)
        else:
            error[-i-1] = err

    # ----- Final correction on finest grid -----
    m = 2 * m + 1
    U = U - interpolate(error[1], 0, 0)
    U, rr = Jacobi(U, F, alpha, beta, m, nu, omega)

    # Compute error
    err_max = np.max(np.abs(U - u_exact))
    return err_max

# ---------- Experiment: vary nu ----------
k = 8                      # grid level (2^8 - 1 = 255 points)
omega = 2./3
smooth = True
nu_values = range(1, 11)   # 1 to 10 iterations
errors = []

for nu in nu_values:
    e = Vcycle_error(k, nu, omega, smooth)
    errors.append(e)
    print(f"nu = {nu:2d}, max error = {e:.3e}")

# Plot results
plt.figure(figsize=(8,5))
plt.plot(nu_values, errors, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Jacobi iterations $\\nu$')
plt.ylabel('Max absolute error')
plt.title(f'Accuracy vs. number of smoothing steps (k={k})')
plt.grid(True)
plt.show()