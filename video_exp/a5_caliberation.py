"""
Batch microscope calibration (ellipse-aware).

Scans IMAGE_DIR for files named  m_<magnification>_d_<diameter>mm.<ext>
(e.g. m_2.0_d_1.5mm.jpg), fits an ellipse to each disc edge, takes the MAJOR axis
as the true diameter (tilt only foreshortens the minor axis), converts to um/px,
and writes ONE calibration json with a single um/px value per magnification.

Pipeline per image
  1. Rough segmentation (row-flattening + threshold + most circular blob).
  2. Sub-pixel edge points along 360 rays from the centre (parabolic refinement).
  3. Ellipse fit (cv2.fitEllipse) with MAD-based outlier rejection and refit.
  4. Error: bootstrap over edge points on the major axis length.
  5. um/px = D_true_um / major_axis_px.
Per magnification: inverse-variance weighted mean over all diameters.
"""
import os
import re
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # no GUI needed for batch runs
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter1d

# ------------------------------------------------------------------ paths
BASE_DIR = "F:/Mohit's BTP/Python Code/video_exp/caliberation_images"
IMAGE_DIR = f"{BASE_DIR}/actual_images"
FIG_DIR = f"{BASE_DIR}/fitted_images"
OUT_JSON = "F:/Mohit's BTP/Python Code/magnification.json"

DEFAULT_MAG = "2"          # used only if the json has no "default" yet
INCLUDE_DETAILS = True     # also write err_um_per_px and n_images per magnification
N_BOOT = 500

FNAME_RE = re.compile(r"^m_([\d.]+?)_d_([\d.]+?)mm\.(jpg|jpeg|png|tif|tiff|bmp)$", re.IGNORECASE)


# ------------------------------------------------------------------ step 1
def rough_circle(gray, thresh=0.75):
    g = gray.astype(np.float32)
    # remove row banding; 90th percentile (not median) so a big disc covering
    # >50% of a row doesn't get mistaken for the background level
    flat = g / (np.percentile(g, 90, axis=1, keepdims=True) + 1e-6)
    flat = cv2.GaussianBlur(flat, (0, 0), 5)
    mask = (flat < thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No dark blob found - adjust thresholding.")
    best, best_score = None, -1
    for c in cnts:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if a < 500 or p == 0:
            continue
        score = (4 * np.pi * a / p ** 2) * np.sqrt(a)
        if score > best_score:
            best, best_score = c, score
    if best is None:
        raise RuntimeError("No circular blob found.")
    M = cv2.moments(best)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    r = np.sqrt(cv2.contourArea(best) / np.pi)
    return cx, cy, r


# ------------------------------------------------------------------ step 2
def edge_points(gray, cx, cy, r0, n_rays=360, span=0.4, smooth=2.0, min_reach=1.15):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    rs = np.arange(r0 * (1 - span), r0 * (1 + span), 0.25)
    pts, angs = [], []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        xs_all = cx + rs * np.cos(th)
        ys_all = cy + rs * np.sin(th)
        inside = ((xs_all >= 1) & (ys_all >= 1) &
                  (xs_all <= g.shape[1] - 2) & (ys_all <= g.shape[0] - 2))
        n_ok = len(inside) if inside.all() else int(np.argmin(inside))
        rs_ray = rs[:n_ok]
        # ray must reach clearly past the expected edge, otherwise the disc edge
        # on this side is outside the frame and there is nothing to measure
        if n_ok < 40 or rs_ray[-1] < r0 * min_reach:
            continue
        prof = map_coordinates(g, [ys_all[:n_ok], xs_all[:n_ok]], order=1)
        d = np.gradient(gaussian_filter1d(prof, smooth / 0.25))
        i = np.argmax(d)  # dark inside -> bright outside
        if i < 3 or i > len(d) - 4:
            continue  # peak at the end of the window = no real edge found
        y0, y1, y2 = d[i - 1], d[i], d[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0
        r_edge = rs[0] + (i + off) * 0.25
        pts.append((cx + r_edge * np.cos(th), cy + r_edge * np.sin(th)))
        angs.append(th)
    return np.array(pts), np.array(angs)


# ------------------------------------------------------------------ step 3
def ellipse_radial_resid(pts, ell):
    """Signed radial distance (px) of each point from the ellipse, measured along
    the line through the ellipse centre (positive = outside)."""
    (ex, ey), (w, h), ang = ell
    a, b = w / 2.0, h / 2.0
    t = np.radians(ang)
    dx, dy = pts[:, 0] - ex, pts[:, 1] - ey
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)
    R = np.hypot(dx, dy)
    rho = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    return R * (1 - 1 / np.maximum(rho, 1e-9))


def fit_ellipse(pts):
    P = pts.astype(np.float32)
    ell = cv2.fitEllipse(P)
    r = ellipse_radial_resid(pts, ell)
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r) < max(3 * sigma, 0.5)
    if keep.sum() >= 5:
        ell = cv2.fitEllipse(P[keep])
    else:
        keep = np.ones(len(pts), bool)
    return ell, keep


def major_diameter(ell):
    return max(ell[1])  # cv2 returns full axis lengths


def bootstrap_major(pts, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    P = pts.astype(np.float32)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(P), len(P))
        try:
            out.append(major_diameter(cv2.fitEllipse(P[idx])))
        except cv2.error:
            continue
    return np.array(out)


# ------------------------------------------------------------------ per image
def process_image(path, diameter_mm, fig_path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cx0, cy0, r0 = rough_circle(gray)
    pts, angs = edge_points(gray, cx0, cy0, r0)
    coverage = len(pts) / 360.0
    if len(pts) < 20:
        raise RuntimeError("Too few edge points - disc mostly outside the frame "
                           "or segmentation failed (check threshold).")

    ell, keep = fit_ellipse(pts)
    (ex, ey), (w, h), ang = ell
    D_major = major_diameter(ell)
    D_minor = min(w, h)

    boot = bootstrap_major(pts[keep])
    D_err = boot.std()

    um_per_px = diameter_mm * 1000.0 / D_major
    um_err = um_per_px * D_err / D_major

    res = ellipse_radial_resid(pts, ell)
    rms = np.sqrt(np.mean(res[keep] ** 2))

    # ---- plot
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [2, 1.3, 1]})
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t = np.linspace(0, 2 * np.pi, 400)
    a, b, phi = w / 2, h / 2, np.radians(ang)
    ax[0].plot(pts[~keep, 0], pts[~keep, 1], "x", c="orange", ms=5, label="rejected")
    ax[0].plot(pts[keep, 0], pts[keep, 1], ".", c="cyan", ms=3, label="edge points")
    ax[0].plot(ex + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi),
               ey + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi),
               "r-", lw=1.2, label="fitted ellipse")
    ax[0].plot(ex, ey, "r+", ms=12)
    ax[0].set_title(f"major = {D_major:.2f} ± {D_err:.2f} px, minor = {D_minor:.2f} px\n"
                    f"{um_per_px:.4f} ± {um_err:.4f} µm/px")
    ax[0].legend(loc="upper right")
    ax[1].plot(np.degrees(angs[keep]), res[keep], ".", c="C0")
    ax[1].plot(np.degrees(angs[~keep]), res[~keep], "x", c="orange")
    ax[1].axhline(0, c="k", lw=0.7)
    ax[1].set_xlabel("angle (deg)")
    ax[1].set_ylabel("radial residual (px)")
    ax[1].set_title("Residuals vs angle")
    ax[2].hist(boot, bins=30, color="C2")
    ax[2].axvline(D_major, c="r")
    ax[2].set_xlabel("major axis (px)")
    ax[2].set_title("Bootstrap major axis")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130)
    plt.close(fig)

    return dict(um_per_px=um_per_px, um_err=um_err, major_px=D_major, minor_px=D_minor,
                axis_ratio=D_major / D_minor, rms=rms, n_used=int(keep.sum()), n_total=len(pts), coverage=coverage)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # collect images
    jobs = []
    for fn in sorted(os.listdir(IMAGE_DIR)):
        m = FNAME_RE.match(fn)
        if m:
            jobs.append((float(m.group(1)), float(m.group(2)), fn))
    if not jobs:
        raise SystemExit(f"No images matching m_<mag>_d_<diam>mm.<ext> in {IMAGE_DIR}")
    jobs.sort()
    print(f"Found {len(jobs)} images.\n")

    per_mag = {}  # mag_key -> list of (um_per_px, um_err)
    for mag, diam, fn in jobs:
        key = f"{mag:g}"
        fig_path = f"{FIG_DIR}/{os.path.splitext(fn)[0]}.png"
        try:
            r = process_image(os.path.join(IMAGE_DIR, fn), diam, fig_path)
        except Exception as e:
            print(f"[SKIP] {fn}: {e}")
            continue
        per_mag.setdefault(key, []).append((r["um_per_px"], r["um_err"]))
        print(f"{fn:<28} major {r['major_px']:8.2f} px  ratio {r['axis_ratio']:.3f}  "
              f"rms {r['rms']:.2f} px  pts {r['n_used']}/{r['n_total']}  "
              f"arc {r['coverage']*100:.0f}%{'  <-- PARTIAL ARC' if r['coverage'] < 0.8 else ''}  ->  "
              f"{r['um_per_px']:.4f} ± {r['um_err']:.4f} um/px")

    # combine diameters -> one value per magnification
    magnifications = {}
    print("\n--- Per-magnification result (inverse-variance weighted mean) ---")
    for key in sorted(per_mag, key=float):
        v = np.array([x[0] for x in per_mag[key]])
        e = np.array([x[1] for x in per_mag[key]])
        e = np.maximum(e, 1e-9)
        w = 1 / e ** 2
        mean = np.sum(w * v) / np.sum(w)
        err = max(1 / np.sqrt(np.sum(w)), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        spread = (v.max() - v.min()) / mean * 100 if len(v) > 1 else 0.0
        flag = "  <-- check: images disagree" if spread > 1.0 else ""
        print(f"mag {key:>5}: {mean:.4f} ± {err:.4f} um/px  "
              f"({len(v)} images, spread {spread:.2f}%){flag}")
        entry = {"um_per_px": round(float(mean), 4)}
        if INCLUDE_DETAILS:
            entry["err_um_per_px"] = round(float(err), 4)
            entry["n_images"] = len(v)
        magnifications[key] = entry

    # keep capture_resolution / default from an existing file if present
    capture_resolution, default = None, None
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON) as f:
                old = json.load(f)
            capture_resolution = old.get("capture_resolution")
            default = old.get("default")
        except Exception:
            pass
    if default not in magnifications:
        default = DEFAULT_MAG if DEFAULT_MAG in magnifications else next(iter(magnifications))

    out = {"capture_resolution": capture_resolution,
           "default": default,
           "magnifications": magnifications}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved calibration -> {OUT_JSON}")


if __name__ == "__main__":
    main()