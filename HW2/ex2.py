import numpy as np
import matplotlib.pyplot as plt


# Berilgan funksiyalar (G va J)
def G(theta, alpha, beta, h):
    Gval = np.zeros(len(theta))
    Gval[0] = theta[1] - 2 * theta[0] + alpha
    Gval[1:-1] = theta[:-2] - 2 * theta[1:-1] + theta[2:]
    Gval[-1] = theta[-2] - 2 * theta[-1] + beta
    Gval /= h ** 2
    Gval += np.sin(theta)
    return Gval


def J(theta, m, T):
    h = T / (m + 1)
    e = np.ones(m)
    return 1. / h ** 2 * (np.diag(-2 * e, 0) + np.diag(e[:-1], -1) + np.diag(e[:-1], 1)) + np.diag(np.cos(theta))


# Newton usulini hisoblovchi asosiy funksiya (Yaqinlashishni hisoblash qo'shildi)
def solve_pendulum_newton(T, alpha, beta, m, theta_guess, tol=1e-15, max_iter=10):
    h = T / (m + 1)
    theta = np.copy(theta_guess)
    history = [theta.copy()]
    norms = []

    for k in range(max_iter):
        G_k = G(theta, alpha, beta, h)
        J_k = J(theta, m, T)
        delta = np.linalg.solve(J_k, -G_k)
        # (convergence)
        norm_delta = np.linalg.norm(delta, np.inf)
        norms.append(norm_delta)

        theta = theta + delta
        history.append(theta.copy())
        if norm_delta < tol:
            break
    return history, norms


# --- Parametrlar va Dasturni ishga tushirish ---
T = 6 * np.pi
alpha = 0.7
beta = 0.7
m = 100
h = T / (m + 1)

t_full = np.linspace(0, T, m + 2)
t_inner = np.linspace(h, T - h, m)

# (a) holat
theta0_1 = 0.7 + 4.00 * np.sin(np.pi * t_inner / T)
history_a, norms_a = solve_pendulum_newton(T, alpha, beta, m, theta0_1)

# (b) holat
theta0_2 = 0.7 + 4.09 * np.sin(np.pi * t_inner / T)
history_b, norms_b = solve_pendulum_newton(T, alpha, beta, m, theta0_2)

# --- Yaqinlashish (Convergence) natijalarini konsolga chop etish ---
print("Change ||delta^[k]||_infty in solution in each iteration:")
print("-" * 45)
print(f"{'k':<5} | {'Figure 2.4(a)':<17} | {'Figure 2.4(b)':<17}")
print("-" * 45)
max_k = max(len(norms_a), len(norms_b))
for k in range(max_k):
    norm_a_str = f"{norms_a[k]:.4e}" if k < len(norms_a) else ""
    norm_b_str = f"{norms_b[k]:.4e}" if k < len(norms_b) else ""
    print(f"{k:<5} | {norm_a_str:<17} | {norm_b_str:<17}")
print("-" * 45)

# --- Natijalarni chizish (Plotting) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


def plot_iterations(ax, history, title, line_color):
    for k, theta in enumerate(history):
        if k > 4:
            continue
        theta_full = np.concatenate(([alpha], theta, [beta]))

        # Oxirgi chiziqni qalinroq qilish
        linewidth = 2.5 if (k == len(history) - 1 or k == 4) else 1

        # Ranglarni qo'llash
        ax.plot(t_full, theta_full, color=line_color, linewidth=linewidth, alpha=0.9)

        mid_idx = m // 2
        offset = 0.05 if k % 2 == 0 else -0.05
        ax.text(t_full[mid_idx], theta_full[mid_idx] + offset, str(k), fontsize=9)

    ax.set_title(title)
    ax.set_ylim([-12, 18])
    ax.set_xlim([0, 20])


# a va b grafiklar uchun to'q ko'k va to'q qizil ranglarni berish
plot_iterations(ax1, history_a, "(a) Starting from $\\theta_i^{[0]} = 0.7 + 4.00\\sin(\\pi t_i / T)$", 'darkblue')
plot_iterations(ax2, history_b, "(b) Starting from $\\theta_i^{[0]} = 0.7 + 4.09\\sin(\\pi t_i / T)$", 'darkred')

plt.tight_layout()
plt.show()