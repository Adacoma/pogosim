#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toner–Tu simulator (2D) with Taichi (+GPU), stabilized + metrics compatible with tt_classify.py

Features
--------
- Periodic BCs, explicit Euler with adaptive sub-stepping (CFL advection + diffusion)
- NaN/Inf guards, |v| clip, rho clamping
- PNG frames, colormap configurable via YAML or CLI
- tqdm progress bar
- metrics.csv with columns:
    frame_id,t,mean_speed,polarization,dP_dt,
    rho_mean,rho_std,rho_contrast,
    vort_rms,vort_kurt,ow_neg_frac,
    anisotropy_A,band_theta_deg,
    streamness_S,
    dom_kx,dom_ky,dom_k_mag,dom_wavelength,
    render_field

CLI
---
python toner_tu_sim.py -c config.yaml -o out_dir [--cmap plasma] [--field speed]

Minimal YAML
------------
grid: {nx: 256, ny: 256, Lx: 20.0, Ly: 20.0}
time: {dt: 0.02, n_steps: 5000, save_every: 25, seed: 3}
model:
  alpha: 0.6; beta: 0.6; lambda1: 0.35; lambda2: -0.6; lambda3: 0.10
  cs2: 1.2; Dv: 0.16; Db: 0.08; Drho: 0.03; noise_sigma: 0.001
render: {field: speed, cmap: viridis, vmin: auto, vmax: auto}
metrics: {spectrum_field: rho, vortex_kappa: 1.0}
stability: {cfl: 0.25, cfl_diff: 0.10, max_substeps: 20, v_clip: 5.0, rho_clip: [0.1, 10.0]}
"""
import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timezone

from tqdm.auto import tqdm
import taichi as ti

# ----------------------------- CLI -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Toner–Tu Taichi GPU simulator")
    ap.add_argument("-c", "--config", required=True, help="YAML configuration file")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory for PNG/CSV")
    ap.add_argument("--cmap", default=None, help="Override colormap (e.g., viridis, plasma)")
    ap.add_argument("--field", default=None, choices=["rho", "speed", "vorticity"],
                    help="Override rendered field")
    return ap.parse_args()

# --------------------------- Utilities --------------------------
def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_metrics_header(csv_path):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "frame_id","t","mean_speed","polarization","dP_dt",
                "rho_mean","rho_std","rho_contrast",
                "vort_rms","vort_kurt","ow_neg_frac",
                "anisotropy_A","band_theta_deg",
                "streamness_S",
                "dom_kx","dom_ky","dom_k_mag","dom_wavelength",
                "render_field"
            ])

# ------------------------ Simulation Core -----------------------
@ti.func
def pbc(i, n):
    # Taichi 1.7.x: no early returns in @ti.func
    res = i
    if res < 0:
        res += n
    if res >= n:
        res -= n
    return res

@ti.func
def isfinite_scalar(x: ti.f32) -> ti.i32:
    # Finite check for Taichi 1.7.x (no ti.math.isfinite)
    return ti.cast((x == x) and (ti.abs(x) < 1e19), ti.i32)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # --- Grid & time ---
    nx = int(cfg["grid"]["nx"])
    ny = int(cfg["grid"]["ny"])
    Lx = float(cfg["grid"].get("Lx", nx))
    Ly = float(cfg["grid"].get("Ly", ny))
    dx = Lx / nx
    dy = Ly / ny

    dt_user = float(cfg["time"]["dt"])
    n_steps = int(cfg["time"]["n_steps"])
    save_every = int(cfg["time"]["save_every"])
    seed = int(cfg["time"].get("seed", 0))

    # --- Model params ---
    m = cfg["model"]
    alpha = float(m["alpha"])
    beta  = float(m["beta"])
    lam1  = float(m["lambda1"])
    lam2  = float(m["lambda2"])
    lam3  = float(m["lambda3"])
    cs2   = float(m["cs2"])
    Dv    = float(m["Dv"])
    Db    = float(m["Db"])
    Drho  = float(m["Drho"])
    noise_sigma = float(m.get("noise_sigma", 0.0))

    # --- Render & metrics config ---
    render_cfg = cfg.get("render", {})
    cmaps_cfg = (render_cfg.get("cmaps") or {})  # dict: {rho: ..., speed: ..., vorticity: ...}

    # CLI --cmap acts as fallback for any field whose cmap isn't specified
    def cmap_for(field_name: str) -> str:
        if field_name in cmaps_cfg and cmaps_cfg[field_name]:
            return str(cmaps_cfg[field_name])
        if args.cmap:  # fallback from CLI
            return str(args.cmap)
        return "viridis"

    # We no longer select a single field; keep legacy CLI --field but ignore it.
    # vmin/vmax autoscale for now (simplest + safest); easy to add per-field later if you want.
    vmin_cfg = "auto"
    vmax_cfg = "auto"

    metrics_cfg = cfg.get("metrics", {})
    spectrum_field = metrics_cfg.get("spectrum_field", "rho")  # rho or speed

    # --- Stability config ---
    stab = cfg.get("stability", {})
    cfl = float(stab.get("cfl", 0.25))
    cfl_diff = float(stab.get("cfl_diff", 0.10))
    max_substeps = int(stab.get("max_substeps", 20))
    v_clip = float(stab.get("v_clip", 5.0))
    rho_min, rho_max = stab.get("rho_clip", [0.1, 10.0])
    rho_min = float(rho_min); rho_max = float(rho_max)

    outdir = args.outdir
    ensure_outdir(outdir)
    metrics_csv = os.path.join(outdir, "metrics.csv")
    write_metrics_header(metrics_csv)

    # --- Taichi init ---
    ti.init(arch=ti.gpu, random_seed=seed)  # falls back to CPU if no GPU
    np.random.seed(seed)

    # Fields
    rho  = ti.field(dtype=ti.f32, shape=(nx, ny))
    v    = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
    vnew = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
    rhonew = ti.field(dtype=ti.f32, shape=(nx, ny))

    # Scratch fields
    grad_rho = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
    div_v    = ti.field(dtype=ti.f32, shape=(nx, ny))
    curl_v   = ti.field(dtype=ti.f32, shape=(nx, ny))
    speed    = ti.field(dtype=ti.f32, shape=(nx, ny))
    lap_v    = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
    lap_rho  = ti.field(dtype=ti.f32, shape=(nx, ny))
    smax     = ti.field(dtype=ti.f32, shape=())

    inv2dx = 1.0 / (2.0 * dx)
    inv2dy = 1.0 / (2.0 * dy)
    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)

    # -------------------- Initialization --------------------
    @ti.kernel
    def init_fields():
        for i, j in rho:
            rho[i, j] = 1.0 + 0.01 * ti.randn()
            v[i, j] = ti.Vector([0.1 * ti.randn(), 0.1 * ti.randn()])

    # ----------------- Differential Operators ----------------
    @ti.kernel
    def compute_derivatives():
        for i, j in rho:
            im = pbc(i - 1, nx)
            ip = pbc(i + 1, nx)
            jm = pbc(j - 1, ny)
            jp = pbc(j + 1, ny)

            grad_rho[i, j] = ti.Vector([
                (rho[ip, j] - rho[im, j]) * inv2dx,
                (rho[i, jp] - rho[i, jm]) * inv2dy
            ])

            vx_ip = v[ip, j].x
            vx_im = v[im, j].x
            vy_jp = v[i, jp].y
            vy_jm = v[i, jm].y
            div_v[i, j] = (vx_ip - vx_im) * inv2dx + (vy_jp - vy_jm) * inv2dy

            vy_ip = v[ip, j].y
            vy_im = v[im, j].y
            vx_jp = v[i, jp].x
            vx_jm = v[i, jm].x
            curl_v[i, j] = (vy_ip - vy_im) * inv2dx - (vx_jp - vx_jm) * inv2dy

            speed[i, j] = v[i, j].norm()

            lap_vx = (v[ip, j].x - 2.0 * v[i, j].x + v[im, j].x) * invdx2 + \
                     (v[i, jp].x - 2.0 * v[i, j].x + v[i, jm].x) * invdy2
            lap_vy = (v[ip, j].y - 2.0 * v[i, j].y + v[im, j].y) * invdx2 + \
                     (v[i, jp].y - 2.0 * v[i, j].y + v[i, jm].y) * invdy2
            lap_v[i, j] = ti.Vector([lap_vx, lap_vy])

            lap_rho[i, j] = (rho[ip, j] - 2.0 * rho[i, j] + rho[im, j]) * invdx2 + \
                            (rho[i, jp] - 2.0 * rho[i, j] + rho[i, jm]) * invdy2

    @ti.kernel
    def update_velocity(dt: ti.f32, alpha: ti.f32, beta: ti.f32,
                        lam1: ti.f32, lam2: ti.f32, lam3: ti.f32,
                        cs2: ti.f32, Dv: ti.f32, Db: ti.f32, noise_sigma: ti.f32):
        for i, j in v:
            vi = v[i, j]
            s2 = vi.dot(vi)
            force_sp = (alpha - beta * s2) * vi
            force_press = -cs2 * grad_rho[i, j]

            im = pbc(i - 1, nx); ip = pbc(i + 1, nx)
            jm = pbc(j - 1, ny); jp = pbc(j + 1, ny)
            dv_dx = ti.Vector([
                (v[ip, j].x - v[im, j].x) * inv2dx,
                (v[ip, j].y - v[im, j].y) * inv2dx
            ])
            dv_dy = ti.Vector([
                (v[i, jp].x - v[i, jm].x) * inv2dy,
                (v[i, jp].y - v[i, jm].y) * inv2dy
            ])
            adv1 = ti.Vector([
                vi.x * dv_dx.x + vi.y * dv_dy.x,
                vi.x * dv_dx.y + vi.y * dv_dy.y
            ])  # (v·∇)v

            adv2 = div_v[i, j] * vi

            s2_x = ( (v[ip, j].dot(v[ip, j])) - (v[im, j].dot(v[im, j])) ) * inv2dx
            s2_y = ( (v[i, jp].dot(v[i, jp])) - (v[i, jm].dot(v[i, jm])) ) * inv2dy
            grad_s2 = ti.Vector([s2_x, s2_y])

            visc = Dv * lap_v[i, j] + Db * grad_s2
            eta = noise_sigma * ti.Vector([ti.randn(), ti.randn()])

            vnew[i, j] = vi + dt * (force_sp + force_press + lam1 * adv1 + lam2 * adv2 + lam3 * grad_s2 + visc) + (dt ** 0.5) * eta

    @ti.kernel
    def update_density(dt: ti.f32, Drho: ti.f32):
        for i, j in rho:
            im = pbc(i - 1, nx); ip = pbc(i + 1, nx)
            jm = pbc(j - 1, ny); jp = pbc(j + 1, ny)
            rhov_x_ip = rho[ip, j] * v[ip, j].x
            rhov_x_im = rho[im, j] * v[im, j].x
            rhov_y_jp = rho[i, jp] * v[i, jp].y
            rhov_y_jm = rho[i, jm] * v[i, jm].y
            div_rhov = (rhov_x_ip - rhov_x_im) * inv2dx + (rhov_y_jp - rhov_y_jm) * inv2dy

            rhonew[i, j] = rho[i, j] + dt * (-div_rhov + Drho * lap_rho[i, j])

    @ti.kernel
    def swap_fields():
        for i, j in v:
            v[i, j] = vnew[i, j]
            rho[i, j] = rhonew[i, j]

    @ti.kernel
    def compute_max_speed():
        s = 0.0
        for i, j in v:
            sp = v[i, j].norm()
            if sp > s:
                s = sp
        smax[None] = s

    @ti.kernel
    def sanitize_fields(vmax: ti.f32, rmin: ti.f32, rmax: ti.f32):
        for i, j in v:
            vx = v[i, j].x
            vy = v[i, j].y
            if isfinite_scalar(vx) == 0:
                vx = 0.0
            if isfinite_scalar(vy) == 0:
                vy = 0.0
            vv = ti.Vector([vx, vy])
            sp = vv.norm()
            if sp > vmax:
                vv = vv * (vmax / (sp + 1e-12))
            v[i, j] = vv

            r = rho[i, j]
            if isfinite_scalar(r) == 0:
                r = 1.0
            if r < rmin:
                r = rmin
            if r > rmax:
                r = rmax
            rho[i, j] = r

    init_fields()

    # ---------------------- Rendering ------------------------

    def get_field_np_by_name(name: str):
        if name == "rho":
            return rho.to_numpy()
        elif name == "speed":
            return speed.to_numpy()
        elif name == "vorticity":
            return curl_v.to_numpy()
        raise ValueError(f"Unknown field '{name}'")

    def get_render_field_np():
        if field_to_render == "rho":
            return rho.to_numpy()
        elif field_to_render == "speed":
            return speed.to_numpy()
        elif field_to_render == "vorticity":
            return curl_v.to_numpy()
        else:
            return speed.to_numpy()

    def _save_one_png(field_name: str, frame_id: int, t: float):
        arr = get_field_np_by_name(field_name)
        # guard renderer from NaNs/Infs
        finite = np.isfinite(arr)
        if not finite.any():
            arr = np.zeros_like(arr)
        else:
            arr = np.where(finite, arr, np.nanmedian(arr))

        vmin_auto = (vmin_cfg == "auto")
        vmax_auto = (vmax_cfg == "auto")
        vmin = float(np.min(arr)) if vmin_auto else float(vmin_cfg)
        vmax = float(np.max(arr)) if vmax_auto else float(vmax_cfg)
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-3

        plt.figure(figsize=(6, 6), dpi=150)
        plt.imshow(
            arr.T, origin="lower",
            cmap=cmap_for(field_name),
            vmin=vmin, vmax=vmax,
            extent=[0, Lx, 0, Ly], aspect='equal'
        )
        plt.title(f"{field_name}  t={t:.3f}")
        plt.xlabel("x"); plt.ylabel("y")
        plt.tight_layout()
        out_png = os.path.join(outdir, f"{field_name}_{frame_id:06d}.png")
        plt.savefig(out_png)
        plt.close()

    def save_all_pngs(frame_id: int, t: float):
        # Always save all three
        _save_one_png("rho", frame_id, t)
        _save_one_png("speed", frame_id, t)
        _save_one_png("vorticity", frame_id, t)


    # ---------------------- Metrics --------------------------
    prev_P = None
    time_at_prev_save = None
    saved_frame_idx = -1  # will increment on saves

    def compute_metrics_and_log(frame_id, t_now):
        nonlocal prev_P, time_at_prev_save, saved_frame_idx

        v_np = v.to_numpy()           # (nx, ny, 2)
        rho_np = rho.to_numpy()       # (nx, ny)
        curl_np = curl_v.to_numpy()   # (nx, ny)
        div_np  = div_v.to_numpy()    # (nx, ny)
        speed_np = np.linalg.norm(v_np, axis=2)

        mean_speed = float(np.mean(speed_np))
        mean_v = np.mean(v_np.reshape(-1, 2), axis=0)
        P = float(np.linalg.norm(mean_v) / (mean_speed + 1e-12))

        # dP_dt over saved frames (use time between saves)
        if prev_P is None or time_at_prev_save is None:
            dP_dt = 0.0
        else:
            dt_save = t_now - time_at_prev_save
            dP_dt = float((P - prev_P) / (dt_save + 1e-12))

        rho_mean = float(np.mean(rho_np))
        rho_std  = float(np.std(rho_np))
        rho_contrast = float(rho_std / (rho_mean + 1e-12))

        vort_rms = float(np.sqrt(np.mean(curl_np**2)))
        # robust kurtosis (excess + 3 approx via moment ratio)
        mu = float(np.mean(curl_np))
        var = float(np.mean((curl_np - mu)**2)) + 1e-20
        m4 = float(np.mean((curl_np - mu)**4))
        vort_kurt = float(m4 / (var**2))

        # Okubo–Weiss negative fraction (vortex indicator)
        # compute finite diffs on the fly (central)
        # ∂x vx, ∂y vx, ∂x vy, ∂y vy
        vx = v_np[..., 0]; vy = v_np[..., 1]
        # central with periodic wrap using roll
        dvx_dx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2*dx)
        dvx_dy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2*dy)
        dvy_dx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2*dx)
        dvy_dy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2*dy)
        # Strain components and vorticity
        s_n = dvx_dx - dvy_dy
        s_s = dvy_dx + dvx_dy
        omega = curl_np  # already ∂x vy - ∂y vx
        OW = (s_n**2 + s_s**2) - (omega**2)
        ow_neg_frac = float(np.mean(OW < 0.0))

        # FFT-based spectral measures
        if spectrum_field == "rho":
            field = rho_np
        else:
            field = speed_np
        F = np.fft.fft2(field - np.mean(field))
        Pk = np.abs(F)**2
        kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(ny, d=dy) * 2.0 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2)

        # exclude k=0
        mask = K > 0
        if not np.any(mask):
            dom_idx = (0, 0)
            dom_kx = dom_ky = dom_k_mag = 0.0
            dom_lambda = np.inf
            anisotropy_A = 0.0
            band_theta_deg = 0.0
        else:
            # Dominant k (peak power)
            peak_flat = np.argmax(Pk[mask])
            mask_idx = np.argwhere(mask)
            pi, pj = mask_idx[peak_flat]
            dom_kx = float(KX[pi, pj])
            dom_ky = float(KY[pi, pj])
            dom_k_mag = float(np.sqrt(dom_kx**2 + dom_ky**2))
            dom_lambda = float((2.0 * np.pi) / (dom_k_mag + 1e-12))
            # Orientation angle in [0, 180)
            band_theta_deg = float((np.degrees(np.arctan2(dom_ky, dom_kx)) + 180.0) % 180.0)

            # Anisotropy: power near kx-axis vs ky-axis
            ang = np.arctan2(KY, KX)  # [-pi, pi]
            ang = np.where(ang < 0, ang + np.pi, ang)  # [0, pi]
            tol = np.deg2rad(15.0)
            near_kx = ((ang < tol) | (np.abs(ang - np.pi) < tol)) & mask
            near_ky = (np.abs(ang - np.pi/2) < tol) & mask
            P_kx = float(np.sum(Pk[near_kx]))
            P_ky = float(np.sum(Pk[near_ky]))
            anisotropy_A = float((P_kx - P_ky) / (P_kx + P_ky + 1e-12))

        # Streamness: ω_rms / (ω_rms + div_rms)
        div_rms = float(np.sqrt(np.mean(div_np**2)))
        streamness_S = float(vort_rms / (vort_rms + div_rms + 1e-12))

        # Append CSV
        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                frame_id,
                f"{t_now:.9g}",
                f"{mean_speed:.9g}",
                f"{P:.9g}",
                f"{dP_dt:.9g}",
                f"{rho_mean:.9g}",
                f"{rho_std:.9g}",
                f"{rho_contrast:.9g}",
                f"{vort_rms:.9g}",
                f"{vort_kurt:.9g}",
                f"{ow_neg_frac:.9g}",
                f"{anisotropy_A:.9g}",
                f"{band_theta_deg:.9g}",
                f"{streamness_S:.9g}",
                f"{dom_kx:.9g}",
                f"{dom_ky:.9g}",
                f"{dom_k_mag:.9g}",
                f"{dom_lambda:.9g}",
                "all"
            ])

        # update dP_dt state
        prev_P = P
        time_at_prev_save = t_now

    # ---------------------- Main Loop ------------------------
    t = 0.0
    for step in tqdm(range(n_steps + 1), desc="Simulating", leave=True, dynamic_ncols=True):
        compute_derivatives()

        if step % save_every == 0:
            saved_frame_idx += 1
            save_all_pngs(saved_frame_idx, t)
            compute_metrics_and_log(saved_frame_idx, t)

        if step < n_steps:
            # Adaptive sub-stepping
            compute_max_speed()
            vmax_now = float(smax[None]) + 1e-8
            dx_min = min(dx, dy)
            dt_adv = cfl * dx_min / vmax_now
            D_eff = max(Dv, Drho)
            dt_diff = cfl_diff * (min(dx*dx, dy*dy) / (4.0 * max(D_eff, 1e-12)))
            dt_safe = max(1e-6, min(dt_adv, dt_diff, dt_user))

            n_sub = int(np.ceil(dt_user / dt_safe))
            n_sub = max(1, min(n_sub, max_substeps))
            dt_sub = dt_user / n_sub

            for _ in range(n_sub):
                update_velocity(dt_sub, alpha, beta, lam1, lam2, lam3, cs2, Dv, Db, noise_sigma)
                update_density(dt_sub, Drho)
                swap_fields()
                sanitize_fields(v_clip, rho_min, rho_max)
                compute_derivatives()
                # on-the-fly cap if needed
                compute_max_speed()
                vmax_now = float(smax[None]) + 1e-8
                dt_adv = cfl * dx_min / vmax_now
                dt_diff = cfl_diff * (min(dx*dx, dy*dy) / (4.0 * max(D_eff, 1e-12)))
                dt_cap = max(1e-6, min(dt_adv, dt_diff))
                if dt_sub > dt_cap:
                    dt_sub = dt_cap

            t += dt_user

    print(f"[OK] Frames + metrics saved to: {outdir}")

if __name__ == "__main__":
    main()

