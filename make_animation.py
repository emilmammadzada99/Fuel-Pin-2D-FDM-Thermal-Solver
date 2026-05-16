"""
Fuel Pin Transient – Full Animation GIF
Runs the complete transient and saves every Nth step as a GIF frame.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from PIL import Image
import io, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════
r_fuel  = 0.39; r_ci = 0.40; r_co = 0.46; pitch = 1.26; P2 = pitch/2.
T_cool  = 560.0
q_lin_init  = 125.0
q_lin_final = 150.0

t_end   = 50.0
dt      = 0.10          # slightly larger step → fewer frames, faster render
n_steps = int(np.ceil(t_end / dt))
frame_every = 1         # save every step as a frame → smooth animation

k_gap_val   = 0.003
k_clad_val  = 0.173
k_water_val = 0.006
rhocp_fuel  = 3.45
rhocp_gap   = 0.001
rhocp_clad  = 1.478
rhocp_water = 4.0

def k_fuel_func(T):
    T = np.clip(T, 300., 3500.)
    return 1.0 / (0.0452 + 2.46e-4*T + 5.47e9*np.exp(-16350./T)/T**2)

# ══════════════════════════════════════════════
# GRID
# ══════════════════════════════════════════════
N  = 101
x  = np.linspace(-P2, P2, N)
y  = np.linspace(-P2, P2, N)
dx = x[1]-x[0]; dy = y[1]-y[0]
X, Y = np.meshgrid(x, y)
R    = np.sqrt(X**2+Y**2)
mid  = N//2

region = np.full((N,N), 3, dtype=int)
region[R<=r_co]=2; region[R<=r_ci]=1; region[R<=r_fuel]=0

pitch_bc = np.zeros((N,N), dtype=bool)
pitch_bc[0,:]=True; pitch_bc[-1,:]=True
pitch_bc[:,0]=True; pitch_bc[:,-1]=True

interior = ~pitch_bc
dofs = -np.ones((N,N), dtype=int)
n_free = int(np.sum(interior))
dofs[interior] = np.arange(n_free)

rhocp = np.zeros((N,N))
rhocp[region==0]=rhocp_fuel; rhocp[region==1]=rhocp_gap
rhocp[region==2]=rhocp_clad; rhocp[region==3]=rhocp_water
rhocp_vec = rhocp[interior]

theta_full = np.linspace(0., 2.*np.pi, 400)

# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def build_k(T):
    K = np.full((N,N), k_water_val)
    K[region==0]=k_fuel_func(T[region==0])
    K[region==1]=k_gap_val
    K[region==2]=k_clad_val
    return K

def harm(a,b): return np.where((a+b)>0., 2.*a*b/(a+b+1e-30), 0.)

def assemble(T, q_lin):
    K = build_k(T)
    q_vol = q_lin / (np.pi*r_fuel**2)
    Q = np.where(region==0, q_vol, 0.)
    A = lil_matrix((n_free,n_free)); rhs=np.zeros(n_free)
    for i in range(N):
        for j in range(N):
            d=dofs[i,j]
            if d<0: continue
            kr = harm(K[i,j],K[i,j+1]) if j<N-1 else 0.
            kl = harm(K[i,j],K[i,j-1]) if j>0   else 0.
            kt = harm(K[i,j],K[i+1,j]) if i<N-1 else 0.
            kb = harm(K[i,j],K[i-1,j]) if i>0   else 0.
            A[d,d]=(kr+kl)/dx**2+(kt+kb)/dy**2
            for di,dj,kf,h2 in [(0,1,kr,dx**2),(0,-1,kl,dx**2),(1,0,kt,dy**2),(-1,0,kb,dy**2)]:
                nb=dofs[i+di,j+dj]
                if nb>=0: A[d,nb]-=kf/h2
                else:     rhs[d]+=kf/h2*T_cool
            rhs[d]+=Q[i,j]
    return A.tocsr(), rhs

def steady(q_lin, T0=None):
    T = np.full((N,N),T_cool) if T0 is None else T0.copy()
    for _ in range(30):
        A,f=assemble(T,q_lin)
        Tv=spsolve(A,f)
        Tn=np.full((N,N),T_cool); Tn[interior]=Tv
        if np.max(np.abs(Tn[interior]-T[interior]))<1e-4: T=Tn; break
        T=Tn
    return T

def timestep(T_old, q_lin, dt):
    M_dt=rhocp_vec/dt
    T=T_old.copy()
    for _ in range(15):
        A_s,f_s=assemble(T,q_lin)
        rhs=M_dt*T_old[interior]+f_s
        Al=A_s.tolil()
        for d in range(n_free): Al[d,d]+=M_dt[d]
        Tv=spsolve(Al.tocsr(),rhs)
        Tn=np.full((N,N),T_cool); Tn[interior]=Tv
        if np.max(np.abs(Tn[interior]-T[interior]))<1e-3: T=Tn; break
        T=Tn
    return T

# ══════════════════════════════════════════════
# SOLVE INITIAL & FINAL SS
# ══════════════════════════════════════════════
print("Solving initial steady-state...")
T_ss0 = steady(q_lin_init)
T_c0  = T_ss0[mid,mid]
print(f"  T_center = {T_c0:.1f} K")

print("Solving final steady-state (for colorbar range)...")
T_ss1 = steady(q_lin_final, T0=T_ss0)
T_c1  = T_ss1[mid,mid]
print(f"  T_center = {T_c1:.1f} K")

Tmin = T_cool
Tmax = T_c1 * 1.02

# ══════════════════════════════════════════════
# TRANSIENT – COLLECT ALL FRAMES
# ══════════════════════════════════════════════
print(f"\nRunning transient ({n_steps} steps, dt={dt}s)...")

# Time series storage
time_hist = [0.0]
Tc_hist   = [T_c0]
Tco_hist  = [T_ss0[mid, -2]]   # near pitch edge along x

T_cur = T_ss0.copy()
all_frames_T = [T_ss0.copy()]   # t=0 frame
all_frames_t = [0.0]

for step in range(1, n_steps+1):
    t = step*dt
    T_cur = timestep(T_cur, q_lin_final, dt)
    time_hist.append(t)
    Tc_hist.append(T_cur[mid,mid])
    Tco_hist.append(T_cur[mid,-2])
    if step % frame_every == 0:
        all_frames_T.append(T_cur.copy())
        all_frames_t.append(t)
    if step % 20 == 0:
        print(f"  t={t:.1f}s  T_c={T_cur[mid,mid]:.1f} K")

n_frames = len(all_frames_T)
print(f"\n  Total frames to render: {n_frames}")

time_hist = np.array(time_hist)
Tc_hist   = np.array(Tc_hist)
Tco_hist  = np.array(Tco_hist)

# ══════════════════════════════════════════════
# RENDER FRAMES
# ══════════════════════════════════════════════
print("Rendering frames...")

fig = plt.figure(figsize=(12, 5.2), facecolor='white')
gs  = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0],
                        left=0.05, right=0.97, top=0.88, bottom=0.12,
                        wspace=0.35)
ax_map = fig.add_subplot(gs[0])
ax_rad = fig.add_subplot(gs[1])
ax_ts  = fig.add_subplot(gs[2])

for ax in [ax_map, ax_rad, ax_ts]:
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='black', labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor('#555')

# Pre-compute colormap norm
norm = matplotlib.colors.Normalize(vmin=Tmin, vmax=Tmax)
cmap = cm.jet

# Radial x-axis data (y=0 slice, x>=0)
x_pos = x[mid:]

# Steady-state reference profile (final SS)
T_ss1_row = T_ss1[mid, mid:]

pil_frames = []

for fi, (T_fr, t_fr) in enumerate(zip(all_frames_T, all_frames_t)):

    # ── 2D map ──
    ax_map.clear()
    ax_map.set_facecolor('white')
    pcm = ax_map.pcolormesh(X, Y, T_fr, cmap=cmap, norm=norm, shading='auto')
    for r_line, col, ls in [(r_fuel,'#00e5ff','--'),(r_ci,'#00ff88','--'),(r_co,'#ffffff','-')]:
        ax_map.plot(r_line*np.cos(theta_full), r_line*np.sin(theta_full),
                    ls, color=col, linewidth=0.9, alpha=0.8)
    sq = plt.Polygon([[-P2,-P2],[P2,-P2],[P2,P2],[-P2,P2],[-P2,-P2]],
                     closed=True, fill=False, edgecolor='cyan', linewidth=1.2, linestyle='-')
    ax_map.add_patch(sq)
    ax_map.set_aspect('equal')
    ax_map.set_xlim(-P2*1.05, P2*1.05); ax_map.set_ylim(-P2*1.05, P2*1.05)
    ax_map.set_xlabel('x (cm)', color='black', fontsize=8)
    ax_map.set_ylabel('y (cm)', color='black', fontsize=8)
    ax_map.set_title('2D Temperature Field', color='white', fontsize=9, pad=4)
    ax_map.tick_params(colors='black', labelsize=7)
    for sp in ax_map.spines.values(): sp.set_edgecolor('#555')

    # Colorbar (only first frame; reuse handle)
    if fi == 0:
        cb = fig.colorbar(pcm, ax=ax_map, fraction=0.046, pad=0.04)
        cb.set_label('T (K)', color='black', fontsize=8)
        cb.ax.yaxis.set_tick_params(color='black', labelcolor='black', labelsize=7)

    # T_center annotation
    Tc_now = T_fr[mid,mid]
    ax_map.text(0.03, 0.97, f'T_c = {Tc_now:.0f} K',
                transform=ax_map.transAxes, color='white',
                fontsize=8, va='top', fontweight='bold',
                bbox=dict(facecolor='#222', edgecolor='none', alpha=0.7, pad=2))

    # ── Radial profile ──
    ax_rad.clear()
    ax_rad.set_facecolor('white')
    T_row = T_fr[mid, mid:]
    ax_rad.axvspan(0.,     r_fuel, alpha=0.12, color='#ff6b35')
    ax_rad.axvspan(r_fuel, r_ci,   alpha=0.12, color='#888888')
    ax_rad.axvspan(r_ci,   r_co,   alpha=0.12, color='#4488ff')
    ax_rad.axvspan(r_co,   P2,     alpha=0.08, color='#00ccff')
    ax_rad.plot(x_pos, T_ss1_row, color='#888', linewidth=1.0,
                linestyle=':', label=f'final SS')
    ax_rad.plot(x_pos, T_row, color='#ff9900', linewidth=2.0, label='current')
    ax_rad.set_xlim(0., P2*1.05); ax_rad.set_ylim(Tmin-50, Tmax+100)
    ax_rad.set_xlabel('r (cm)', color='black', fontsize=8)
    ax_rad.set_ylabel('T (K)', color='black', fontsize=8)
    ax_rad.set_title('Radial Profile (y=0)', color='black', fontsize=9, pad=4)
    ax_rad.tick_params(colors='black', labelsize=7)
    for sp in ax_rad.spines.values(): sp.set_edgecolor('#555')
    ax_rad.legend(fontsize=7, facecolor='#222', labelcolor='white', loc='upper right')
    ax_rad.grid(True, alpha=0.2, color='#444')

    # ── Time series ──
    ax_ts.clear()
    ax_ts.set_facecolor('white')
    idx_now = np.searchsorted(time_hist, t_fr)
    ax_ts.plot(time_hist[:idx_now+1], Tc_hist[:idx_now+1],
               color='#ff4444', linewidth=1.8, label='T_center')
    ax_ts.plot(time_hist[:idx_now+1], Tco_hist[:idx_now+1],
               color='#4499ff', linewidth=1.8, label='T_clad OD')
    ax_ts.axhline(T_c1,              color='#ff4444', linewidth=0.8, linestyle=':')
    ax_ts.axhline(T_ss1[mid,-2],     color='#4499ff', linewidth=0.8, linestyle=':')
    ax_ts.axvline(t_fr, color='white', linewidth=0.8, linestyle='--', alpha=0.6)
    ax_ts.set_xlim(0., t_end); ax_ts.set_ylim(Tmin-50, Tmax+100)
    ax_ts.set_xlabel('Time (s)', color='black', fontsize=8)
    ax_ts.set_ylabel('T (K)', color='black', fontsize=8)
    ax_ts.set_title('Temperature History', color='black', fontsize=9, pad=4)
    ax_ts.tick_params(colors='black', labelsize=7)
    for sp in ax_ts.spines.values(): sp.set_edgecolor('#555')
    ax_ts.legend(fontsize=7, facecolor='#222', labelcolor='white', loc='lower right')
    ax_ts.grid(True, alpha=0.2, color='#444')

    # ── Title ──
    fig.suptitle(
        f"Fuel Pin Transient  |  q': {q_lin_init:.0f} → {q_lin_final:.0f} W/cm  "
        f"|  t = {t_fr:.2f} s  |  Backward Euler  dt={dt}s",
        color='black', fontsize=9.5, y=0.97)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, facecolor=fig.get_facecolor())
    buf.seek(0)
    pil_frames.append(Image.open(buf).copy())
    buf.close()

    if (fi+1) % 10 == 0 or fi == n_frames-1:
        print(f"  rendered {fi+1}/{n_frames} frames")

plt.close(fig)

# ══════════════════════════════════════════════
# SAVE GIF
# ══════════════════════════════════════════════
out_gif = 'fuel_pin_transient_anim.gif'
print(f"\nSaving GIF ({n_frames} frames)...")

# Quantize each frame for smaller file, smoother animation
durations = [300] + [80]*(n_frames-2) + [1500]   # hold start/end longer

pil_frames[0].save(
    out_gif,
    save_all=True,
    append_images=pil_frames[1:],
    duration=durations,
    loop=0,
    optimize=False,
)
import os
size_mb = os.path.getsize(out_gif) / 1e6
print(f"GIF saved → {out_gif}  ({size_mb:.1f} MB,  {n_frames} frames)")
