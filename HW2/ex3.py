import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, linalg
import tracemalloc
import time

# 1. Main functions (Based our lecture codes)

def five_pt_laplacian(m):
    e=np.ones(m**2)
    e2=([0]+[1]*(m-1))*m
    h=1./(m+1)
    A=np.diag(-4*e,0)+np.diag(e2[1:],-1)+np.diag(e2[1:],1)+np.diag(e[m:],m)+np.diag(e[m:],-m)
    A/=h**2
    return A

def five_pt_laplacian_sparse(m):
    e=np.ones(m**2)
    e2=([1]*(m-1)+[0])*m
    e3=([0]+[1]*(m-1))*m
    h=1./(m+1)
    A=spdiags([-4*e,e2,e3,e,e],[0,-1,1,-m,m],m**2,m**2)
    A/=h**2
    return A

def five_pt_laplacian_dense(m):
    e = np.ones(m**2)
    e2 = ([0] + [1]*(m-1))*m
    h = 1.0 / (m+1)
    A = np.diag(-4*e, 0) + np.diag(e2[1:], -1) + np.diag(e2[1:], 1) + np.diag(e[m:], m) + np.diag(e[m:], -m)
    A /= h**2
    return A

def solve_poisson(m, track_memory=False):
    h = 1.0 / (m + 1)
    x = np.linspace(0, 1, m + 2)[1:-1]
    y = np.linspace(0, 1, m + 2)[1:-1]
    X, Y = np.meshgrid(x, y)

    # RHS function
    F_grid = -(20 * Y ** 3 + 9 * np.pi ** 2 * (Y - Y ** 5)) * np.sin(3 * np.pi * X)
    F = F_grid.reshape(m ** 2)

    A = five_pt_laplacian_sparse(m).tocsr()

    peak_mem_mb = 0
    solve_time = 0

    if track_memory:
        tracemalloc.start()  # RAM kuzatishni boshlash
        start_t = time.time()

    # solving linear system
    U_vec = linalg.spsolve(A, F)

    if track_memory:
        solve_time = time.time() - start_t
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mb = peak_mem_bytes / (1024 * 1024)  # Byte --->Mb

    U_num = U_vec.reshape((m, m))
    U_exact = (Y - Y ** 5) * np.sin(3 * np.pi * X)

    # Max eror (Infinity norm)
    max_error = np.max(np.abs(U_num - U_exact))

    return X, Y, U_num, U_exact, max_error, solve_time, peak_mem_mb

# 2. 3D visualization (for m=50)

m_plot = 50
X, Y, U_num, U_exact, _, _, _ = solve_poisson(m_plot)

fig1 = plt.figure(figsize=(14, 6))
ax1 = fig1.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, U_num, cmap='viridis', edgecolor='none')
ax1.set_title(f'Numerical Solution (m={m_plot})')
ax1.set_xlabel('X');
ax1.set_ylabel('Y');
ax1.set_zlabel('U')
fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

ax2 = fig1.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_exact, cmap='plasma', edgecolor='none')
ax2.set_title('Exact Solution')
ax2.set_xlabel('X');
ax2.set_ylabel('Y');
ax2.set_zlabel('U')
fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
plt.show()

# ---------------------------------------------------------
# 3. Convergence Rate (Yaqinlashish tezligi) tahlili
# ---------------------------------------------------------
m_values_conv = [10, 15, 20, 30, 40, 50, 80, 100,120, 140, 160]
errors = []
h_values = []

print("Convergence Analysis:")
print(f"{'m':<5} | {'h':<10} | {'Max Error':<15} | {'Rate'}")
print("-" * 45)

for i, m in enumerate(m_values_conv):
    _, _, _, _, err, _, _ = solve_poisson(m)
    h = 1.0 / (m + 1)
    h_values.append(h)
    errors.append(err)

    if i == 0:
        print(f"{m:<5} | {h:.6f}   | {err:.6e}    | -")
    else:
        # Rate = log(E1/E2) / log(h1/h2)
        rate = np.log(errors[i - 1] / errors[i]) / np.log(h_values[i - 1] / h_values[i])
        print(f"{m:<5} | {h:.6f}   | {err:.6e}    | {rate:.4f}")

fig2, ax = plt.subplots(figsize=(7, 5))
ax.loglog(h_values, errors, 'o-', markerfacecolor='red', markersize=8, label='Numerical Error')
# Reference chiziq O(h^2) uchun
ax.loglog(h_values, [errors[0] * (h / h_values[0]) ** 2 for h in h_values], 'k--', label='$O(h^2)$ reference')
ax.set_xlabel('Grid spacing (h)')
ax.set_ylabel('Max Error')
ax.set_title('Convergence Rate (Log-Log Plot)')
ax.legend()
ax.grid(True, which="both", ls="--")
plt.show()


# 4. RAM memory and Time spent analysis

m_values_perf = [10, 15, 20, 30, 40, 50, 80, 100,120, 140, 160]
times = []
memories = []

print("\nPerformance & Memory Tracking:")
print(f"{'m':<5} | {'Grid Size':<12} | {'Time (s)':<10} | {'Peak RAM (MB)'}")
print("-" * 50)

for m in m_values_perf:
    _, _, _, _, _, t, mem = solve_poisson(m, track_memory=True)
    times.append(t)
    memories.append(mem)
    print(f"{m:<5} | {m ** 2:<12} | {t:.4f}     | {mem:.2f} MB")

fig3, ax_time = plt.subplots(figsize=(8, 5))
ax_mem = ax_time.twinx()

ax_time.plot(m_values_perf, times, 'b-o', label='Solve Time (s)')
ax_mem.plot(m_values_perf, memories, 'r-s', label='Peak RAM (MB)')

ax_time.set_xlabel('m (Number of grid points in each direction)')
ax_time.set_ylabel('Time (seconds)', color='b')
ax_mem.set_ylabel('Memory (MB)', color='r')
plt.title('Performance vs Memory Scaling (Part a)')
fig3.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
ax_time.grid(True, linestyle=':')
plt.show()