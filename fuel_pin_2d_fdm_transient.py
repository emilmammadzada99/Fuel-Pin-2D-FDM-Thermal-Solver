"""
Fuel Pin 2D FDM  –  TAM PIN  –  TRANSIENT SOLVER
==================================================
Geometry : Full fuel pin, [-pitch/2, pitch/2]²
Regions  : fuel | gap | clad | water

Time Integration : Backward Euler (fully implicit, unconditionally stable)
  ρ cp (T^{n+1} - T^n) / dt = ∇·(k ∇T^{n+1}) + q'''

  → (M/dt + A) T^{n+1} = M/dt T^n + f
  where M = lumped mass matrix (ρ cp / dt diagonal)
        A = stiffness matrix (FDM diffusion)
        f = source + Dirichlet contribution

  k_fuel(T) nonlinearity handled by lagged-coefficient Newton iterations
  at each time step.

Scenario : Power ramp (step change)
  t < 0  : steady-state at q_lin_initial
  t ≥ 0  : q_lin jumps to q_lin_final  (default: 200 → 300 W/cm)
  Simulation runs until new steady-state is approached.

Material thermal properties:
  ρ cp (fuel)  = 3.45  J/cm³/K   (UO₂, ~const)
  ρ cp (gap)   = 0.001 J/cm³/K   (He, negligible)
  ρ cp (clad)  = 1.478 J/cm³/K   (Zircaloy: ρ=6.55 g/cm³, cp=0.2255 J/g/K)
  ρ cp (water) = 4.0   J/cm³/K   (PWR water ~300°C)

Outputs :
  fuel_pin_transient_snapshots.png  – 2D maps at key times
  fuel_pin_transient_timeseries.png – T_center, T_clad_o vs time
  fuel_pin_transient.gif            – animation (optional, needs Pillow)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve
import warnings, os, sys
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# GEOMETRY & STEADY-STATE OPERATING CONDITIONS
# ══════════════════════════════════════════════
r_fuel  = 0.39    # cm
r_ci    = 0.40    # cm
r_co    = 0.46    # cm
pitch   = 1.26    # cm
P2      = pitch / 2.

T_cool  = 560.0   # K  (Dirichlet at pitch boundary, constant in time)

# ── Power ramp scenario ─────────────────────
q_lin_init  = 200.0   # W/cm  (initial steady-state)
q_lin_final = 300.0   # W/cm  (after step at t=0)

# ── Transient time control ───────────────────
t_end   = 10.0    # s   simulation duration after ramp
dt      = 0.05    # s   time step (Backward Euler is stable for any dt)
n_steps = int(np.ceil(t_end / dt))

# Snapshots to plot (seconds)
snap_times = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, t_end]

# ══════════════════════════════════════════════
# MATERIAL PROPERTIES
# ══════════════════════════════════════════════
k_gap_val   = 0.003
k_clad_val  = 0.173
k_water_val = 0.006

rhocp_fuel  = 3.45    # J/cm³/K  UO₂
rhocp_gap   = 0.001   # J/cm³/K  He (tiny)
rhocp_clad  = 1.478   # J/cm³/K  Zr
rhocp_water = 4.0     # J/cm³/K

def k_fuel_func(T):
    T = np.clip(T, 300., 3500.)
    return 1.0 / (0.0452 + 2.46e-4*T + 5.47e9*np.exp(-16350./T)/T**2)

# ══════════════════════════════════════════════
# GRID
# ══════════════════════════════════════════════
N  = 101            # coarser for transient speed; increase for accuracy
x  = np.linspace(-P2, P2, N)
y  = np.linspace(-P2, P2, N)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
R    = np.sqrt(X**2 + Y**2)
mid  = N // 2

# ── Region map ───────────────────────────────
region = np.full((N, N), 3, dtype=int)
region[R <= r_co]   = 2
region[R <= r_ci]   = 1
region[R <= r_fuel] = 0

# ── Pitch BC ─────────────────────────────────
pitch_bc = np.zeros((N, N), dtype=bool)
pitch_bc[0,:]  = True
pitch_bc[-1,:] = True
pitch_bc[:,0]  = True
pitch_bc[:,-1] = True

interior = ~pitch_bc
dofs     = -np.ones((N, N), dtype=int)
n_free   = int(np.sum(interior))
dofs[interior] = np.arange(n_free)

# ── ρ cp field ───────────────────────────────
rhocp = np.zeros((N, N))
rhocp[region == 0] = rhocp_fuel
rhocp[region == 1] = rhocp_gap
rhocp[region == 2] = rhocp_clad
rhocp[region == 3] = rhocp_water
rhocp_vec = rhocp[interior]   # shape (n_free,)

# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def build_k_field(T_field):
    K = np.full((N, N), k_water_val)
    K[region == 0] = k_fuel_func(T_field[region == 0])
    K[region == 1] = k_gap_val
    K[region == 2] = k_clad_val
    return K

def harmonic(a, b):
    return np.where((a + b) > 0., 2.*a*b/(a+b+1e-30), 0.)

def q_vol_from_lin(q_lin):
    return q_lin / (np.pi * r_fuel**2)

def make_Q(q_lin):
    return np.where(region == 0, q_vol_from_lin(q_lin), 0.)

# ══════════════════════════════════════════════
# FDM STIFFNESS ASSEMBLY  (returns A_stiff, f_source)
# ══════════════════════════════════════════════
def assemble_stiffness(T_field, q_lin):
    K = build_k_field(T_field)
    Q = make_Q(q_lin)

    A   = lil_matrix((n_free, n_free))
    rhs = np.zeros(n_free)

    for i in range(N):
        for j in range(N):
            d = dofs[i, j]
            if d < 0:
                continue

            kr = harmonic(K[i,j], K[i,j+1]) if j < N-1 else 0.
            kl = harmonic(K[i,j], K[i,j-1]) if j > 0   else 0.
            kt = harmonic(K[i,j], K[i+1,j]) if i < N-1 else 0.
            kb = harmonic(K[i,j], K[i-1,j]) if i > 0   else 0.

            diag = (kr + kl)/dx**2 + (kt + kb)/dy**2
            A[d, d] = diag

            for (di, dj, kf, h2) in [(0,1,kr,dx**2),(0,-1,kl,dx**2),
                                      (1,0,kt,dy**2),(-1,0,kb,dy**2)]:
                ni, nj = i+di, j+dj
                nb = dofs[ni, nj]
                if nb >= 0:
                    A[d, nb] -= kf / h2
                else:
                    rhs[d] += kf / h2 * T_cool

            rhs[d] += Q[i, j]

    return A.tocsr(), rhs

# ══════════════════════════════════════════════
# STEADY-STATE SOLVE  (Newton)
# ══════════════════════════════════════════════
def solve_steady(q_lin, T_init=None, verbose=True, label=""):
    T = np.full((N, N), T_cool) if T_init is None else T_init.copy()
    T[pitch_bc] = T_cool

    if verbose:
        print(f"\n  Steady-state solve  {label}")
        print(f"  q_lin = {q_lin} W/cm")
        print("  iter  |  max ΔT (K)")
        print("  " + "-"*25)

    for nit in range(30):
        A, f = assemble_stiffness(T, q_lin)
        T_vec = spsolve(A, f)
        T_new = np.full((N, N), T_cool)
        T_new[interior] = T_vec
        dT_max = np.max(np.abs(T_new[interior] - T[interior]))
        if verbose:
            print(f"  {nit+1:3d}   |  {dT_max:.4e}")
        T = T_new.copy()
        if dT_max < 1e-4 and nit >= 1:
            if verbose:
                print(f"  → Converged in {nit+1} iterations")
            break
    return T

# ══════════════════════════════════════════════
# TRANSIENT TIME STEP  (Backward Euler + Newton)
# ══════════════════════════════════════════════
def timestep(T_old, q_lin, dt):
    """
    Solve  (M/dt + A) T^{n+1} = M/dt T_old + f
    with Newton iterations for k_fuel(T).
    M = diag(rhocp_vec / dt)
    """
    M_dt = rhocp_vec / dt   # diagonal mass/dt vector

    T = T_old.copy()

    for nit in range(15):
        A_stiff, f_src = assemble_stiffness(T, q_lin)

        # System: (M/dt + A) T = M/dt T_old + f
        # Add mass matrix to diagonal
        T_old_vec = T_old[interior]
        rhs = M_dt * T_old_vec + f_src

        # Modify diagonal of A_stiff
        A_t = A_stiff.copy().tolil()
        for d in range(n_free):
            A_t[d, d] += M_dt[d]
        A_t = A_t.tocsr()

        T_vec = spsolve(A_t, rhs)
        T_new = np.full((N, N), T_cool)
        T_new[interior] = T_vec

        dT_max = np.max(np.abs(T_new[interior] - T[interior]))
        T = T_new.copy()
        if dT_max < 1e-3:
            break

    return T

# ══════════════════════════════════════════════
# KEY TEMPERATURE EXTRACTION
# ══════════════════════════════════════════════
def key_temps(T_field):
    T_center = T_field[mid, mid]
    T_xrow   = T_field[mid, mid:]
    x_pos    = x[mid:]

    def T_at_r(r_t):
        return float(np.interp(r_t, x_pos, T_xrow))

    return {
        'center'  : T_center,
        'fuel_od' : T_at_r(r_fuel),
        'clad_id' : T_at_r(r_ci),
        'clad_od' : T_at_r(r_co),
    }

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
print("=" * 65)
print("  Fuel Pin 2D FDM  –  TAM PIN  –  TRANSIENT")
print("=" * 65)
print(f"  Grid   : {N}×{N}  dx = {dx*1e4:.1f} μm")
print(f"  dt     : {dt} s   |  t_end : {t_end} s   |  steps : {n_steps}")
print(f"  Ramp   : {q_lin_init} → {q_lin_final} W/cm at t=0")
print()

# ── Step 0: initial steady-state ──
T_ss0 = solve_steady(q_lin_init, label=f"(initial, q={q_lin_init} W/cm)")
kts0  = key_temps(T_ss0)
print(f"  T_center={kts0['center']:.1f} K  T_clad_od={kts0['clad_od']:.1f} K")

# ── Step 1: final steady-state (reference) ──
T_ss1 = solve_steady(q_lin_final, T_init=T_ss0, label=f"(final, q={q_lin_final} W/cm)")
kts1  = key_temps(T_ss1)
print(f"  T_center={kts1['center']:.1f} K  T_clad_od={kts1['clad_od']:.1f} K")

# ── Step 2: transient ──────────────────────
print(f"\n  Running transient ({n_steps} steps)...")

time_hist  = [0.0]
Tc_hist    = [kts0['center']]
Tco_hist   = [kts0['clad_od']]
Tci_hist   = [kts0['clad_id']]
Tfs_hist   = [kts0['fuel_od']]

snap_fields = {}   # time → T_field

# Collect t=0 snapshot (before ramp)
snap_fields[0.0] = T_ss0.copy()

T_cur = T_ss0.copy()

for step in range(1, n_steps + 1):
    t = step * dt
    T_cur = timestep(T_cur, q_lin_final, dt)

    kt = key_temps(T_cur)
    time_hist.append(t)
    Tc_hist.append(kt['center'])
    Tco_hist.append(kt['clad_od'])
    Tci_hist.append(kt['clad_id'])
    Tfs_hist.append(kt['fuel_od'])

    # Store snapshot if near a requested time
    for ts in snap_times:
        if ts > 0 and abs(t - ts) < dt * 0.51 and ts not in snap_fields:
            snap_fields[ts] = T_cur.copy()

    if step % max(1, n_steps // 20) == 0:
        pct = 100 * step / n_steps
        print(f"  t={t:.2f}s ({pct:.0f}%)  T_c={kt['center']:.1f} K  T_co={kt['clad_od']:.1f} K")

time_hist = np.array(time_hist)
Tc_hist   = np.array(Tc_hist)
Tco_hist  = np.array(Tco_hist)
Tci_hist  = np.array(Tci_hist)
Tfs_hist  = np.array(Tfs_hist)

print("\n  Transient complete.")

# ══════════════════════════════════════════════
# PLOT 1: Snapshots  (2D maps)
# ══════════════════════════════════════════════
snap_keys = sorted(snap_fields.keys())
n_snaps   = len(snap_keys)

Tmin_all = T_cool
Tmax_all = max(T.max() for T in snap_fields.values())

theta_full = np.linspace(0., 2.*np.pi, 400)

ncols = min(4, n_snaps)
nrows = (n_snaps + ncols - 1) // ncols
fig1, axes1 = plt.subplots(nrows, ncols,
                            figsize=(4.5*ncols, 4.2*nrows),
                            squeeze=False)

for idx, ts in enumerate(snap_keys):
    r = idx // ncols
    c = idx % ncols
    ax = axes1[r][c]
    Tp = snap_fields[ts].copy()
    pcm = ax.pcolormesh(X, Y, Tp, cmap='hot', shading='auto',
                        vmin=Tmin_all, vmax=Tmax_all)
    fig1.colorbar(pcm, ax=ax, label='T (K)', fraction=0.046, pad=0.04)

    for r_line, col in [(r_fuel,'#00e5ff'),(r_ci,'#00ff88'),(r_co,'#ffffff')]:
        ax.plot(r_line*np.cos(theta_full), r_line*np.sin(theta_full),
                '--', color=col, linewidth=0.9)
    sq = plt.Polygon([[-P2,-P2],[P2,-P2],[P2,P2],[-P2,P2],[-P2,-P2]],
                     closed=True, fill=False, edgecolor='cyan', linewidth=1.2)
    ax.add_patch(sq)
    ax.set_aspect('equal')
    ax.set_xlim(-P2*1.05, P2*1.05)
    ax.set_ylim(-P2*1.05, P2*1.05)
    ax.set_xlabel('x (cm)', fontsize=8)
    ax.set_ylabel('y (cm)', fontsize=8)
    kt = key_temps(snap_fields[ts])
    ax.set_title(f't = {ts:.2f} s\nT_c = {kt["center"]:.0f} K', fontsize=9)

# Hide unused axes
for idx in range(n_snaps, nrows*ncols):
    axes1[idx//ncols][idx%ncols].set_visible(False)

fig1.suptitle(
    f'Transient 2D Snapshots  |  q\': {q_lin_init}→{q_lin_final} W/cm  |  '
    f'T_cool={T_cool} K  |  Grid {N}×{N}',
    fontsize=11, y=1.01)
plt.tight_layout()
out1 = 'fuel_pin_transient_snapshots.png'
fig1.savefig(out1, dpi=130, bbox_inches='tight')
plt.close(fig1)
print(f"  Snapshots saved → {out1}")

# ══════════════════════════════════════════════
# PLOT 2: Time series
# ══════════════════════════════════════════════
fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# ── Power ramp indicator ──
ax_top.axvspan(0., t_end, alpha=0.05, color='red')
ax_top.axvline(0., color='red', linewidth=1.0, linestyle='--', label='ramp at t=0')

ax_top.plot(time_hist, Tc_hist,  'r-',  linewidth=2.0, label='T_center (fuel)')
ax_top.plot(time_hist, Tfs_hist, 'orange', linewidth=1.5, linestyle='--', label='T_fuel surface')
ax_top.plot(time_hist, Tci_hist, 'b--', linewidth=1.5, label='T_clad inner')
ax_top.plot(time_hist, Tco_hist, 'b-',  linewidth=2.0, label='T_clad outer')

# Steady-state reference lines
ax_top.axhline(kts1['center'],  color='r',      linestyle=':', linewidth=1.0,
               label=f'SS T_c = {kts1["center"]:.0f} K')
ax_top.axhline(kts1['clad_od'], color='blue',   linestyle=':', linewidth=1.0,
               label=f'SS T_co = {kts1["clad_od"]:.0f} K')

ax_top.set_ylabel('Temperature (K)', fontsize=11)
ax_top.set_title('Key Temperatures vs Time', fontsize=12)
ax_top.legend(fontsize=9, loc='center right')
ax_top.grid(True, alpha=0.3)

# ── Bottom: ΔT from initial steady-state ──
ax_bot.axvline(0., color='red', linewidth=1.0, linestyle='--')
ax_bot.plot(time_hist, Tc_hist  - kts0['center'],  'r-',  linewidth=2.0, label='ΔT_center')
ax_bot.plot(time_hist, Tfs_hist - kts0['fuel_od'], 'orange', linewidth=1.5, linestyle='--', label='ΔT_fuel surface')
ax_bot.plot(time_hist, Tci_hist - kts0['clad_id'], 'b--', linewidth=1.5, label='ΔT_clad inner')
ax_bot.plot(time_hist, Tco_hist - kts0['clad_od'], 'b-',  linewidth=2.0, label='ΔT_clad outer')

ax_bot.set_xlabel('Time (s)', fontsize=11)
ax_bot.set_ylabel('ΔT from initial SS (K)', fontsize=11)
ax_bot.set_title('Temperature Rise After Power Ramp', fontsize=12)
ax_bot.legend(fontsize=9, loc='center right')
ax_bot.grid(True, alpha=0.3)
ax_bot.axhline(0., color='gray', linewidth=0.8)

fig2.suptitle(
    f'Fuel Pin Transient  |  q\': {q_lin_init}→{q_lin_final} W/cm  |  '
    f'dt={dt}s  |  Backward Euler',
    fontsize=11)
plt.tight_layout()
out2 = 'fuel_pin_transient_timeseries.png'
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close(fig2)
print(f"  Time series saved → {out2}")

# ══════════════════════════════════════════════
# PLOT 3: Radial profiles at multiple times
# ══════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.axvspan(0.,     r_fuel, alpha=0.10, color='#ff6b35', label='fuel')
ax3.axvspan(r_fuel, r_ci,   alpha=0.10, color='#aaaaaa', label='gap')
ax3.axvspan(r_ci,   r_co,   alpha=0.10, color='#4488ff', label='clad')
ax3.axvspan(r_co,   P2,     alpha=0.08, color='#00ccff', label='water')

colors = cm.plasma(np.linspace(0.1, 0.9, len(snap_keys)))
x_pos  = x[mid:]

for idx, ts in enumerate(snap_keys):
    Tp   = snap_fields[ts]
    Trow = Tp[mid, mid:]
    kt   = key_temps(Tp)
    ax3.plot(x_pos, Trow, color=colors[idx], linewidth=1.8,
             label=f't={ts:.2f}s  T_c={kt["center"]:.0f}K')

ax3.axvline(P2, color='cyan', linewidth=1.0, linestyle=':', label=f'pitch/2={P2:.3f}cm')
ax3.set_xlabel('r (cm)', fontsize=11)
ax3.set_ylabel('T (K)', fontsize=11)
ax3.set_title(f'Radial Profiles at Snapshot Times  (q\': {q_lin_init}→{q_lin_final} W/cm)', fontsize=12)
ax3.legend(fontsize=8, loc='upper right')
ax3.set_xlim(0., P2*1.05)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
out3 = 'fuel_pin_transient_radial.png'
fig3.savefig(out3, dpi=130, bbox_inches='tight')
plt.close(fig3)
print(f"  Radial profiles saved → {out3}")

# ══════════════════════════════════════════════
# OPTIONAL: GIF animation
# ══════════════════════════════════════════════
try:
    from PIL import Image
    import io

    print("  Generating GIF animation...")
    frames = []
    anim_times = sorted(snap_fields.keys())

    fig_a, ax_a = plt.subplots(figsize=(5, 4.5))
    for ts in anim_times:
        ax_a.clear()
        Tp = snap_fields[ts]
        pcm = ax_a.pcolormesh(X, Y, Tp, cmap='hot', shading='auto',
                              vmin=Tmin_all, vmax=Tmax_all)
        for r_line, col in [(r_fuel,'#00e5ff'),(r_ci,'#00ff88'),(r_co,'#ffffff')]:
            ax_a.plot(r_line*np.cos(theta_full), r_line*np.sin(theta_full),
                      '--', color=col, linewidth=0.8)
        sq = plt.Polygon([[-P2,-P2],[P2,-P2],[P2,P2],[-P2,P2],[-P2,-P2]],
                         closed=True, fill=False, edgecolor='cyan', linewidth=1.0)
        ax_a.add_patch(sq)
        ax_a.set_aspect('equal')
        ax_a.set_xlim(-P2*1.05, P2*1.05)
        ax_a.set_ylim(-P2*1.05, P2*1.05)
        kt = key_temps(Tp)
        ax_a.set_title(f't = {ts:.2f} s  |  T_center = {kt["center"]:.0f} K', fontsize=9)
        ax_a.set_xlabel('x (cm)'); ax_a.set_ylabel('y (cm)')
        fig_a.tight_layout()

        buf = io.BytesIO()
        fig_a.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig_a)
    out_gif = 'fuel_pin_transient.gif'
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=600, loop=0)
    print(f"  GIF saved → {out_gif}")

except ImportError:
    print("  (Pillow not available – skipping GIF)")

print("\n" + "=" * 65)
print("  DONE")
print("=" * 65)
