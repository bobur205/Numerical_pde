import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, linalg
import tracemalloc
import time


# ---------------------------------------------------------
# 1. Asosiy funksiyalar (Matritsa yaratish)
# ---------------------------------------------------------
def five_pt_laplacian_sparse(m):
    e = np.ones(m ** 2)
    e2 = ([1] * (m - 1) + [0]) * m
    e3 = ([0] + [1] * (m - 1)) * m
    h = 1. / (m + 1)
    A = spdiags([-4 * e, e2, e3, e, e], [0, -1, 1, -m, m], m ** 2, m ** 2)
    A /= h ** 2
    return A


# (c) Task uchun o'zgartirilgan asosiy yechuvchi funksiya
def solve_poisson_task_c(m, track_memory=False):
    h = 1.0 / (m + 1)

    # x va y faqat ichki (interior) tugunlar uchun
    x = np.linspace(0, 1, m + 2)[1:-1]
    y = np.linspace(0, 1, m + 2)[1:-1]
    X, Y = np.meshgrid(x, y)

    # 1-qadam: Yangi RHS funksiyasi f(x,y) = (2 - pi^2 * x^2) * cos(pi * y)
    F_grid = (2 - np.pi ** 2 * X ** 2) * np.cos(np.pi * Y)

    # 2-qadam: Chegara shartlarini (Boundary conditions) F vektoriga qo'shish
    # Formula bo'yicha chekkadagi qiymatlar o'ng tomonga minus bo'lib o'tadi va h^2 ga bo'linadi.

    # Chap chegara (x=0): u(0, y) = 0 (O'zgarish bo'lmaydi)

    # O'ng chegara (x=1): u(1, y) = cos(pi * y)
    F_grid[:, -1] -= np.cos(np.pi * y) / h ** 2

    # Pastki chegara (y=0): u(x, 0) = x^2
    F_grid[0, :] -= (x ** 2) / h ** 2

    # Yuqori chegara (y=1): u(x, 1) = -x^2
    F_grid[-1, :] -= -(x ** 2) / h ** 2

    # Matritsa formatiga o'tkazish
    F = F_grid.reshape(m ** 2)
    A = five_pt_laplacian_sparse(m).tocsr()

    peak_mem_mb = 0
    solve_time = 0

    if track_memory:
        tracemalloc.start()
        start_t = time.time()

    # Chiziqli sistemani yechish
    U_vec = linalg.spsolve(A, F)

    if track_memory:
        solve_time = time.time() - start_t
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    U_num = U_vec.reshape((m, m))

    # 3-qadam: Yangi aniq yechim (Exact solution): u(x,y) = x^2 * cos(pi * y)
    U_exact = (X ** 2) * np.cos(np.pi * Y)

    # Maksimal xatolik (Infinity norm)
    max_error = np.max(np.abs(U_num - U_exact))

    return X, Y, U_num, U_exact, max_error, solve_time, peak_mem_mb


# ---------------------------------------------------------
# 2. 3D Grafiklarni chizish (m=50)
# ---------------------------------------------------------
m_plot = 50
X, Y, U_num, U_exact, _, _, _ = solve_poisson_task_c(m_plot)

fig1 = plt.figure(figsize=(14, 6))
ax1 = fig1.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, U_num, cmap='viridis', edgecolor='none')
ax1.set_title(f'Numerical Solution (Task C, m={m_plot})')
ax1.set_xlabel('X');
ax1.set_ylabel('Y');
ax1.set_zlabel('U')
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

ax2 = fig1.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_exact, cmap='plasma', edgecolor='none')
ax2.set_title('Exact Solution (Task C)')
ax2.set_xlabel('X');
ax2.set_ylabel('Y');
ax2.set_zlabel('U')
fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
plt.show()

# ---------------------------------------------------------
# 3. Yaqinlashish tezligi (Convergence Rate) tahlili
# ---------------------------------------------------------
m_values_conv = [10, 20, 40, 80, 160]
errors = []
h_values = []

print("Convergence Analysis for Task (c):")
print(f"{'m':<5} | {'h':<10} | {'Max Error':<15} | {'Rate'}")
print("-" * 45)

for i, m in enumerate(m_values_conv):
    _, _, _, _, err, _, _ = solve_poisson_task_c(m)
    h = 1.0 / (m + 1)
    h_values.append(h)
    errors.append(err)

    if i == 0:
        print(f"{m:<5} | {h:.6f}   | {err:.6e}    | -")
    else:
        rate = np.log(errors[i - 1] / errors[i]) / np.log(h_values[i - 1] / h_values[i])
        print(f"{m:<5} | {h:.6f}   | {err:.6e}    | {rate:.4f}")

fig2, ax = plt.subplots(figsize=(7, 5))
ax.loglog(h_values, errors, 'o-', markerfacecolor='red', markersize=8, label='Numerical Error')
ax.loglog(h_values, [errors[0] * (h / h_values[0]) ** 2 for h in h_values], 'k--', label='$O(h^2)$ reference')
ax.set_xlabel('Grid spacing (h)')
ax.set_ylabel('Max Error')
ax.set_title('Convergence Rate (Task C)')
ax.legend()
ax.grid(True, which="both", ls="--")
plt.show()