"""
Microscope calibration: fit a circle of known diameter and get micrometres per pixel.

Usage:
    python circle_calibration.py image.jpg --diameter-mm 1.5 [--diameter-err-mm 0.0]

Pipeline
  1. Rough segmentation (blur + Otsu + largest blob) -> initial centre / radius.
  2. Sub-pixel edge points: along 360 rays from the centre, find the position of the
     steepest dark->bright intensity step (parabolic sub-pixel refinement).
  3. Robust geometric circle fit (least squares, soft-L1 loss) to the edge points.
  4. Error estimate: (a) covariance from the fit Jacobian, (b) bootstrap over edge points.
  5. Calibration: um/px = D_true / (2 r_px), with error propagation.
  6. Plot: fit overlay, residuals vs angle, residual histogram.
"""
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.optimize import least_squares
import json
import os

json_path = "F:/Mohit's BTP/Python Code/caliberate.json"

# ----------------------------------------------------------------- step 1
def rough_circle(gray, thresh=0.75):
    # Remove horizontal banding: divide every row by its median (the disc covers
    # well under half of any row, so the median is the background level).
    g = gray.astype(np.float32)
    flat = g / (np.median(g, axis=1, keepdims=True) + 1e-6)
    flat = cv2.GaussianBlur(flat, (0, 0), 5)
    # disc is ~0.5-0.7 of background (=1.0); 0.75 separates it cleanly.
    # Raise/lower `thresh` if your disc is lighter/darker.
    mask = (flat < thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No dark blob found - adjust thresholding.")
    # choose the most circular large blob
    best, best_score = None, -1
    for c in cnts:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if a < 500 or p == 0:
            continue
        circ = 4 * np.pi * a / p ** 2
        score = circ * np.sqrt(a)
        if score > best_score:
            best, best_score = c, score
    if best is None:
        raise RuntimeError("No circular blob found.")
    (cx, cy), r = cv2.minEnclosingCircle(best)
    area = cv2.contourArea(best)
    return cx, cy, np.sqrt(area / np.pi)

# ----------------------------------------------------------------- step 2
def edge_points(gray, cx, cy, r0, n_rays=360, span=0.35, smooth=2.0):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    rs = np.arange(r0 * (1 - span), r0 * (1 + span), 0.25)
    pts, angs = [], []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        xs = cx + rs * np.cos(th)
        ys = cy + rs * np.sin(th)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > g.shape[1] - 2 or ys.max() > g.shape[0] - 2:
            continue  # ray leaves the image
        prof = map_coordinates(g, [ys, xs], order=1)
        d = np.gradient(gaussian_filter1d(prof, smooth / 0.25))
        i = np.argmax(d)  # dark inside -> bright outside = positive gradient
        if 0 < i < len(d) - 1:  # parabolic sub-sample refinement
            y0, y1, y2 = d[i - 1], d[i], d[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den != 0 else 0
        else:
            off = 0
        r_edge = rs[0] + (i + off) * 0.25
        pts.append((cx + r_edge * np.cos(th), cy + r_edge * np.sin(th)))
        angs.append(th)
    return np.array(pts), np.array(angs)

# ----------------------------------------------------------------- step 3
def fit_circle(pts, init):
    def resid(p, P):
        return np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) - p[2]

    sol = least_squares(resid, init, args=(pts,), loss="soft_l1", f_scale=1.0)
    # drop gross outliers (MAD based) and refit with plain least squares
    r = resid(sol.x, pts)
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r) < max(3 * sigma, 0.5)
    sol = least_squares(resid, sol.x, args=(pts[keep],))
    return sol, keep, resid


def param_cov(sol, n):
    J = sol.jac
    dof = max(n - 3, 1)
    s2 = np.sum(sol.fun ** 2) / dof
    return np.linalg.inv(J.T @ J) * s2, np.sqrt(s2)


def bootstrap(pts, p0, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pts), len(pts))
        s = least_squares(lambda p, P: np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) - p[2],
                          p0, args=(pts[idx],))
        out.append(s.x)
    return np.array(out)


# ----------------------------------------------------------------- main
def main():
    while True:
        try:
            magnification = float(input("Magnification: "))
            diameter = float(input("Diameter: "))
        except ValueError:
            print("Please enter valid numbers.")
            continue
        image = f"F:/Mohit's BTP/Python Code/video_exp/caliberation_images/actual_images/m_{magnification}_d_{diameter}mm.jpg"

        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cx0, cy0, r0 = rough_circle(gray)
        pts, angs = edge_points(gray, cx0, cy0, r0)
        sol, keep, resid = fit_circle(pts, [cx0, cy0, r0])
        cx, cy, r = sol.x

        cov, rms = param_cov(sol, keep.sum())
        r_err_cov = np.sqrt(cov[2, 2])
        boot = bootstrap(pts[keep], sol.x)
        r_err_boot = boot[:, 2].std()
        r_err = max(r_err_cov, r_err_boot)  # conservative

        # also an ellipse fit, as a sanity check on pixel aspect ratio / tilt
        (ex, ey), (MA, ma), ang = cv2.fitEllipse(pts[keep].astype(np.float32))
        axis_ratio = max(MA, ma) / min(MA, ma)

        D_um = diameter * 1000.0
        um_per_px = D_um / (2 * r)
        rel = np.sqrt((r_err / r) ** 2)
        um_err = um_per_px * rel

        print(f"Centre (px)       : ({cx:.2f}, {cy:.2f})")
        print(f"Radius (px)       : {r:.2f} +/- {r_err:.2f}   "
            f"(covariance {r_err_cov:.2f}, bootstrap {r_err_boot:.2f})")
        print(f"Diameter (px)     : {2 * r:.2f} +/- {2 * r_err:.2f}")
        print(f"Fit RMS residual  : {rms:.2f} px  ({keep.sum()}/{len(pts)} edge points used)")
        print(f"Ellipse axis ratio: {axis_ratio:.3f}  (1.000 = perfect circle)")
        print(f"Calibration       : {um_per_px:.3f} +/- {um_err:.3f} um/px")
        print(f"                    ({1 / um_per_px:.4f} px/um)")

        output_data = json.load(open(json_path, "r"))

        mag_key = str(magnification)

        if mag_key in output_data.keys():
            output_data[mag_key].append({
            "actual_diameter_mm": diameter,
            "calibration_um_per_px": round(um_per_px, 4),
            "calibration_err_um_per_px": round(um_err, 4),
        })
        else:
            output_data[mag_key] = [{
                    "actual_diameter_mm": diameter,
                    "calibration_um_per_px": round(um_per_px, 4),
                    "calibration_err_um_per_px": round(um_err, 4),
                }]

        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=4)

        # ------------------------------------------------------------- plots
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [2, 1.3, 1]})
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = np.linspace(0, 2 * np.pi, 400)
        ax[0].plot(pts[~keep, 0], pts[~keep, 1], "x", c="orange", ms=5, label="rejected")
        ax[0].plot(pts[keep, 0], pts[keep, 1], ".", c="cyan", ms=3, label="edge points")
        ax[0].plot(cx + r * np.cos(t), cy + r * np.sin(t), "r-", lw=1.2, label="fitted circle")
        ax[0].plot(cx, cy, "r+", ms=12)
        ax[0].set_title(f"r = {r:.2f} ± {r_err:.2f} px  →  {um_per_px:.3f} ± {um_err:.3f} µm/px")
        ax[0].legend(loc="upper right")
        # zoomed view of the fit
        ax[0].set_xlim(0, img.shape[1]); ax[0].set_ylim(img.shape[0], 0)

        res = resid(sol.x, pts)
        ax[1].plot(np.degrees(angs[keep]), res[keep], ".", c="C0")
        ax[1].plot(np.degrees(angs[~keep]), res[~keep], "x", c="orange")
        ax[1].axhline(0, c="k", lw=0.7)
        ax[1].set_xlabel("angle (deg)"); ax[1].set_ylabel("radial residual (px)")
        ax[1].set_title("Residuals vs angle\n(systematic waves = not a circle / uneven lighting)")

        ax[2].hist(boot[:, 2], bins=30, color="C2")
        ax[2].axvline(r, c="r")
        ax[2].set_xlabel("radius (px)"); ax[2].set_title("Bootstrap radius")
        plt.tight_layout()
        plt.savefig(f"F:/Mohit's BTP/Python Code/video_exp/caliberation_images/fitted_images/m_{magnification}_d_{diameter}.png", dpi=130)
        print(f"Saved figure -> {f"F:/Mohit's BTP/Python Code/video_exp/caliberation_images/fitted_images/m_{magnification}_d_{diameter}.png"}")
        # plt.show()

        choice = input("Press Enter to calibrate another image, or type 'q' to quit: ")
        if choice.strip().lower() == 'q':
            print("Exiting calibration tool.")
            break


if __name__ == "__main__":
    main()