"""
Fuel Pin 2D FDM Thermal Diffusion Solver  – FULL PIN (full pin)
================================================================
Geometry : Full fuel pin cross-section in Cartesian (x, y) domain
           Domain : [-pitch/2, pitch/2] × [-pitch/2, pitch/2]

Regions   :
  0 – Fuel   (r ≤ r_fuel)
  1 – Gap    (r_fuel < r ≤ r_ci)   – He gas
  2 – Clad   (r_ci   < r ≤ r_co)
  3 – Water  (r_co   < r ≤ pitch/2 square boundary)

BC        :
  Dirichlet T = T_cool  on pitch boundary (x = ±pitch/2, y = ±pitch/2)
  (Water Region)

Physics   :
  -∇·(k ∇T) = q'''   in fuel
   q''' = q_lin / (π r_fuel²)   [W/cm³]
  k_fuel(T) = 1 / (0.0452 + 2.46e-4·T + 5.47e9·exp(-16350/T)/T²)  [W/cm/K]
  k_gap   = 0.003   W/cm/K  (He gas, constant)
  k_clad  = 0.173   W/cm/K  (Zircaloy, constant)
  k_water = 0.006   W/cm/K  (PWR status, ~300°C)

Numerics  :
  Uniform Cartesian grid over [-pitch/2, pitch/2]²
  5-point FDM with harmonic-mean conductivity at cell faces
  Dirichlet BC on pitch boundary
  Newton iteration for temperature-dependent fuel conductivity
  Sparse direct solver (scipy.sparse.linalg.spsolve)

Outputs   :
  fuel_pin_2d_full_temperature.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GEOMETRY & OPERATING CONDITIONS
# ─────────────────────────────────────────────
r_fuel  = 0.39    # cm  fuel pellet outer radius
r_ci    = 0.40    # cm  clad inner radius
r_co    = 0.46    # cm  clad outer radius
pitch   = 1.26    # cm  pin pitch (full cell: pitch × pitch)

q_lin   = 125.0   # W/cm  linear heat rate
T_cool  = 560.0   # K     bulk coolant / pitch-boundary temperature

# ─────────────────────────────────────────────
# MATERIAL PROPERTIES
# ─────────────────────────────────────────────
k_gap_val   = 0.003   # W/cm/K  He gas
k_clad_val  = 0.173   # W/cm/K  Zircaloy
k_water_val = 0.006   # W/cm/K  PWR water (~300°C)

def k_fuel_func(T):
    """UO₂ thermal conductivity, OECD/NEA  [W/cm/K]"""
    T = np.clip(T, 300., 3500.)
    return 1.0 / (0.0452 + 2.46e-4 * T + 5.47e9 * np.exp(-16350.0 / T) / T**2)

# ─────────────────────────────────────────────
# GRID  –  full domain [-P/2, P/2] × [-P/2, P/2]
# ─────────────────────────────────────────────
N  = 201            # odd → node at x=0, y=0 (pin center)
P2 = pitch / 2.     # half-pitch
x  = np.linspace(-P2, P2, N)
y  = np.linspace(-P2, P2, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)   # shape (N, N), row = y-index
R    = np.sqrt(X**2 + Y**2)

# ─────────────────────────────────────────────
# REGION MAP
#  0 = fuel | 1 = gap | 2 = clad | 3 = water
# ─────────────────────────────────────────────
region = np.full((N, N), 3, dtype=int)   # everything starts as water
region[R <= r_co]   = 2   # clad
region[R <= r_ci]   = 1   # gap
region[R <= r_fuel] = 0   # fuel

# Pitch boundary mask (outermost ring of nodes → Dirichlet)
pitch_bc = np.zeros((N, N), dtype=bool)
pitch_bc[0, :]  = True   # y = -P/2
pitch_bc[-1, :] = True   # y = +P/2
pitch_bc[:, 0]  = True   # x = -P/2
pitch_bc[:, -1] = True   # x = +P/2

# ─────────────────────────────────────────────
# VOLUMETRIC HEAT SOURCE
# ─────────────────────────────────────────────
q_vol = q_lin / (np.pi * r_fuel**2)   # W/cm³
Q     = np.where(region == 0, q_vol, 0.)

# ─────────────────────────────────────────────
# DOF NUMBERING  (all interior nodes are free)
# ─────────────────────────────────────────────
interior = ~pitch_bc                    # inner nodes → unknowns
dofs     = -np.ones((N, N), dtype=int)
n_free   = int(np.sum(interior))
dofs[interior] = np.arange(n_free)

# ─────────────────────────────────────────────
# CONDUCTIVITY FIELD
# ─────────────────────────────────────────────
def build_k_field(T_field):
    K = np.full((N, N), k_water_val)
    K[region == 0] = k_fuel_func(T_field[region == 0])
    K[region == 1] = k_gap_val
    K[region == 2] = k_clad_val
    # water keeps k_water_val
    return K

def harmonic(a, b):
    return np.where((a + b) > 0., 2. * a * b / (a + b + 1e-30), 0.)

# ─────────────────────────────────────────────
# FDM MATRIX ASSEMBLY
# ─────────────────────────────────────────────
def assemble(T_field):
    """
    -∇·(k ∇T) = Q  with Dirichlet T = T_cool on pitch boundary.
    """
    K = build_k_field(T_field)

    A   = lil_matrix((n_free, n_free))
    rhs = np.zeros(n_free)

    for i in range(N):
        for j in range(N):
            d = dofs[i, j]
            if d < 0:
                continue   # pitch BC node, skip

            # face conductivities
            # right  (j+1)
            if j < N-1:
                kr = harmonic(K[i, j], K[i, j+1])
            else:
                kr = 0.   # shouldn't happen (boundary already excluded)
            # left   (j-1)
            if j > 0:
                kl = harmonic(K[i, j], K[i, j-1])
            else:
                kl = 0.
            # top    (i+1)
            if i < N-1:
                kt = harmonic(K[i, j], K[i+1, j])
            else:
                kt = 0.
            # bottom (i-1)
            if i > 0:
                kb = harmonic(K[i, j], K[i-1, j])
            else:
                kb = 0.

            diag = (kr + kl) / dx**2 + (kt + kb) / dy**2
            A[d, d] = diag

            # off-diagonal / Dirichlet contribution
            for (di, dj, k_face, h2) in [
                (0,  1, kr, dx**2),
                (0, -1, kl, dx**2),
                (1,  0, kt, dy**2),
                (-1, 0, kb, dy**2),
            ]:
                ni, nj = i + di, j + dj
                nb = dofs[ni, nj]
                if nb >= 0:
                    A[d, nb] -= k_face / h2
                else:
                    # Dirichlet node (pitch boundary) → move to RHS
                    rhs[d] += k_face / h2 * T_cool

            rhs[d] += Q[i, j]

    return A.tocsr(), rhs

# ─────────────────────────────────────────────
# NEWTON ITERATION
# ─────────────────────────────────────────────
print("=" * 60)
print("  Fuel Pin 2D FDM Thermal Solver  – FULL PIN")
print("=" * 60)
print(f"  Grid          : {N}×{N}  |  dx = {dx*1e4:.2f} μm")
print(f"  Domain        : [{-P2:.3f}, {P2:.3f}] cm  (pitch = {pitch} cm)")
print(f"  r_fuel / r_ci / r_co : {r_fuel} / {r_ci} / {r_co} cm")
print(f"  q_lin         : {q_lin} W/cm")
print(f"  T_cool (pitch BC) : {T_cool} K")
print()

# Initial temperature field
T_field = np.full((N, N), T_cool)
T_field[pitch_bc] = T_cool   # Dirichlet on boundary always

max_newton = 25
tol_newton = 1e-4   # K

print("  Newton iter   |  max ΔT (K)")
print("  " + "-" * 30)

for nit in range(max_newton):
    A, rhs = assemble(T_field)
    T_vec_new = spsolve(A, rhs)

    # Reconstruct 2D field
    T_new = np.full((N, N), T_cool)
    T_new[interior] = T_vec_new

    dT_max = np.max(np.abs(T_new[interior] - T_field[interior]))
    print(f"  {nit+1:4d}          |  {dT_max:.4e}")

    T_field = T_new.copy()

    if dT_max < tol_newton and nit >= 1:
        print(f"\n  Converged in {nit+1} iterations (ΔT < {tol_newton} K)")
        break

# ─────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────
T_plot = T_field.copy()

# Key radial temperatures (along x-axis for simplicity)
# Use 1D slice along y=0 (middle row)
mid = N // 2
x_pos  = x[mid:]                  # x >= 0
T_xrow = T_field[mid, mid:]       # T along y=0, x>=0

r_vals = np.sqrt(X.ravel()**2 + Y.ravel()**2)

def T_at_r(r_target):
    """Interpolate T at a given radius from the 2D field."""
    mask = interior.ravel()
    rs   = r_vals[mask]
    Ts   = T_field.ravel()[mask]
    return float(np.interp(r_target, rs[np.argsort(rs)], Ts[np.argsort(rs)]))

T_center = T_field[mid, mid]
T_fuel_s = T_at_r(r_fuel)
T_clad_i = T_at_r(r_ci)
T_clad_o = T_at_r(r_co)
T_water_m = T_at_r((r_co + P2 * np.sqrt(2)) / 2.)   # midpoint in water

print()
print("  Temperature budget")
print("  " + "-" * 45)
print(f"  T_center (fuel)      = {T_center:.1f} K")
print(f"  T_fuel surface       = {T_fuel_s:.1f} K   ΔT_fuel = {T_center - T_fuel_s:.1f} K")
print(f"  T_clad inner         = {T_clad_i:.1f} K   ΔT_gap  = {T_fuel_s - T_clad_i:.1f} K")
print(f"  T_clad outer         = {T_clad_o:.1f} K   ΔT_clad = {T_clad_i - T_clad_o:.1f} K")
print(f"  T_cool (pitch BC)    = {T_cool:.1f} K   ΔT_water= {T_clad_o - T_cool:.1f} K")
print()
print(f"  k_fuel at center     = {k_fuel_func(T_center):.4f} W/cm/K")
print()

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left: 2D temperature map ──────────────────
ax = axes[0]
pcm = ax.pcolormesh(X, Y, T_plot, cmap='hot', shading='auto',
                    vmin=T_cool, vmax=T_center)
cb  = fig.colorbar(pcm, ax=ax, label='T (K)', fraction=0.046, pad=0.04)

# Region boundary circles
theta_full = np.linspace(0., 2. * np.pi, 400)
for r_line, color, lbl in [
    (r_fuel, '#00e5ff', f'fuel/gap  r={r_fuel} cm'),
    (r_ci,   '#00ff88', f'gap/clad  r={r_ci} cm'),
    (r_co,   '#ffffff', f'clad OD   r={r_co} cm'),
]:
    ax.plot(r_line * np.cos(theta_full), r_line * np.sin(theta_full),
            '--', color=color, linewidth=1.2, label=lbl)

# Pitch boundary square
sq = plt.Polygon(
    [[-P2,-P2],[P2,-P2],[P2,P2],[-P2,P2],[-P2,-P2]],
    closed=True, fill=False, edgecolor='cyan', linewidth=1.5,
    linestyle='-', label=f'pitch {pitch} cm')
ax.add_patch(sq)

ax.set_xlim(-P2 * 1.05, P2 * 1.05)
ax.set_ylim(-P2 * 1.05, P2 * 1.05)
ax.set_aspect('equal')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_title('2D Temperature Field – Full Pin', fontsize=12)
ax.legend(fontsize=8, loc='upper right', framealpha=0.55, facecolor='#111')

# ── Right: 1D radial profile (along x-axis, x≥0) ──
ax2 = axes[1]

ax2.axvspan(0.,     r_fuel, alpha=0.15, color='#ff6b35', label='fuel')
ax2.axvspan(r_fuel, r_ci,   alpha=0.15, color='#aaaaaa', label='gap')
ax2.axvspan(r_ci,   r_co,   alpha=0.15, color='#4488ff', label='clad')
ax2.axvspan(r_co,   P2,     alpha=0.12, color='#00ccff', label='water')

# 1D profile (y=0 slice)
ax2.plot(x_pos, T_xrow, 'k-', linewidth=2.0, label='2D FDM (y=0 slice)')

# Pitch corner distance
r_corner = P2 * np.sqrt(2)
ax2.axvline(P2, color='cyan', linewidth=1.0, linestyle=':', label=f'pitch/2 = {P2:.3f} cm')

# Analytical fuel reference
kf_avg = k_fuel_func(0.5 * (T_center + T_fuel_s))
r_ana  = np.linspace(1e-6, r_fuel, 200)
T_ana  = T_fuel_s + q_vol / (4. * kf_avg) * (r_fuel**2 - r_ana**2)
ax2.plot(r_ana, T_ana, 'r--', linewidth=1.5, label='analytic (const k_f)')

# Key temperature markers
for r_pt, T_pt, lbl in [
    (0.,     T_center, f'center {T_center:.0f} K'),
    (r_fuel, T_fuel_s, f'fuel OD {T_fuel_s:.0f} K'),
    (r_ci,   T_clad_i, f'clad ID {T_clad_i:.0f} K'),
    (r_co,   T_clad_o, f'clad OD {T_clad_o:.0f} K'),
    (P2,     T_cool,   f'pitch BC {T_cool:.0f} K'),
]:
    ax2.axhline(T_pt, color='gray', linewidth=0.5, linestyle=':')
    ax2.scatter([r_pt], [T_pt], zorder=5, s=40, color='black')
    ax2.annotate(lbl, (r_pt, T_pt), textcoords='offset points',
                 xytext=(5, 4), fontsize=8)

ax2.set_xlabel('r (cm)')
ax2.set_ylabel('T (K)')
ax2.set_title('Radial Temperature Profile (y=0 slice)', fontsize=12)
ax2.legend(fontsize=8)
ax2.set_xlim(0., P2 * 1.05)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f'Fuel Pin 2D FDM – Full Pin  |  q\' = {q_lin} W/cm  |  T_cool = {T_cool} K  |  '
    f'pitch = {pitch} cm  |  Grid {N}×{N}',
    fontsize=10, y=1.01)

plt.tight_layout()
outfile = 'fuel_pin_2d_full_temperature.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Plot saved → {outfile}")
print("=" * 60)
