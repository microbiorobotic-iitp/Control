#!/usr/bin/env python3
"""
Step-by-step demo of every OpenCV function used in track_robot.py.

A synthetic test image is generated (OpenCV ships no sample images), containing:
  * a FULL dark round "robot"                (the target)
  * a PARTIAL robot cut by the right border  (tests the circle-fit logic)
  * an elongated dark ellipse                (distractor: low circularity)
  * a lighter grey disc                      (distractor: not dark enough)
  * tiny dark specks                         (noise, removed by morphological open)

Usage:
    python opencv_demo_steps.py              # press any key to advance, ESC to quit
    python opencv_demo_steps.py --no-display # just save every step to ./demo_out/*.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--no-display", action="store_true")
ap.add_argument("--out", default="demo_out")
ARGS = ap.parse_args()
os.makedirs(ARGS.out, exist_ok=True)
STEP = [0]


def show(title, img, note=""):
    """Print an explanation, save the image, and (optionally) display it."""
    STEP[0] += 1
    name = f"{STEP[0]:02d}_{title.replace(' ', '_')}"
    print(f"\n=== STEP {STEP[0]}: {title} ===")
    if note:
        print(note)
    cv2.imwrite(os.path.join(ARGS.out, name + ".png"), img)
    if not ARGS.no_display:
        view = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        view = view.copy()
        cv2.putText(view, f"STEP {STEP[0]}: {title}  (any key = next, ESC = quit)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("demo", view)
        if cv2.waitKey(0) & 0xFF == 27:
            cv2.destroyAllWindows()
            sys.exit(0)


# ------------------------------------------------------------------ 0. test image
def make_test_image():
    H, W = 360, 480
    rng = np.random.default_rng(1)
    # lighter background with a gentle left-to-right gradient + noise
    grad = np.tile(np.linspace(190, 225, W, dtype=np.float32), (H, 1))
    img = grad + rng.normal(0, 4, (H, W))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.circle(img, (150, 200), 32, (35, 35, 35), -1)                   # FULL robot
    cv2.circle(img, (470, 90), 32, (35, 35, 35), -1)                    # PARTIAL robot
    cv2.ellipse(img, (300, 280), (55, 12), 25, 0, 360, (40, 40, 40), -1)  # elongated distractor
    cv2.circle(img, (290, 110), 30, (150, 150, 150), -1)                # too-light distractor
    for _ in range(25):                                                 # specks
        cv2.circle(img, (int(rng.integers(5, W - 5)), int(rng.integers(5, H - 5))),
                   1, (30, 30, 30), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


frame = make_test_image()
H, W = frame.shape[:2]
show("original BGR frame", frame,
     "Synthetic frame. In the tracker this is what cap.read() returns (BGR, uint8).")

# ------------------------------------------------------------------ 1. prep()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
show("cvtColor BGR to GRAY", gray,
     "cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY): 3 channels -> 1 intensity channel.\n"
     "The tracker only cares about 'how dark', so colour is dropped.")

blur = cv2.GaussianBlur(gray, (5, 5), 0)
show("GaussianBlur 5x5", blur,
     "cv2.GaussianBlur(gray, (5,5), 0): weighted-average smoothing; sigma=0 means\n"
     "'derive sigma from kernel size'. Suppresses sensor noise so thresholds are stable.")

# ------------------------------------------------------------------ 2. threshold
_, fixed = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
show("threshold fixed (100) INV", fixed,
     "cv2.threshold(..., 100, 255, THRESH_BINARY_INV): pixels < 100 become 255 (white),\n"
     "others 0. INV is used because the robot is DARK and we want it white in the mask.\n"
     "In find() this is written as (win < T).astype(uint8)*255, which is the same idea.")

otsu_val, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
show("threshold Otsu", otsu,
     f"THRESH_OTSU picks the threshold automatically (here T = {otsu_val:.0f}) by\n"
     "maximising between-class variance. init_from_roi() uses this on your ROI crop.")

# ------------------------------------------------------------------ 3. morphology
OPEN_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
print("\nKernel from cv2.getStructuringElement(MORPH_ELLIPSE,(3,3)):\n", OPEN_K)

opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, OPEN_K)
show("morphologyEx OPEN", opened,
     "MORPH_OPEN = erode then dilate. Removes blobs smaller than the kernel (the specks)\n"
     "while leaving big blobs almost unchanged.")

big_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
show("erode 15x15", cv2.erode(opened, big_k),
     "cv2.erode shrinks white regions. core_median() uses it to get the INTERIOR of the\n"
     "robot, avoiding blurry edge pixels when measuring darkness.")
show("dilate 15x15", cv2.dilate(opened, big_k),
     "cv2.dilate grows white regions. ring_median() dilates twice and subtracts to get\n"
     "a donut of background around the robot.")

# ------------------------------------------------------------------ 4. contours
cs, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"\nfindContours found {len(cs)} outer contours "
      "(RETR_EXTERNAL = ignore holes, CHAIN_APPROX_NONE = keep every boundary pixel).")
vis = frame.copy()
cv2.drawContours(vis, cs, -1, (0, 255, 0), 1)
show("findContours + drawContours", vis,
     "cv2.drawContours(img, contours, -1, colour, thickness): -1 = draw all contours.\n"
     "thickness=-1 would FILL them instead (used to build masks).")

# ------------------------------------------------------------------ 5. shape stats
vis = frame.copy()
print("\nPer-contour stats (same math as contour_stats()):")
print(f"{'#':>2} {'area':>7} {'perim':>7} {'circ':>5} {'solid':>5}  centroid        touches border")
for i, c in enumerate(cs):
    a = cv2.contourArea(c)
    if a < 15:
        continue
    p = cv2.arcLength(c, True)
    circ = 4 * np.pi * a / (p * p)
    hull = cv2.convexHull(c)
    sol = a / cv2.contourArea(hull)
    m = cv2.moments(c)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    bx, by, bw, bh = cv2.boundingRect(c)
    touch = bx == 0 or by == 0 or bx + bw >= W or by + bh >= H
    print(f"{i:>2} {a:7.0f} {p:7.1f} {circ:5.2f} {sol:5.2f}  ({cx:5.1f},{cy:5.1f})  {touch}")
    cv2.polylines(vis, [hull], True, (255, 0, 0), 1)                 # hull = blue
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 165, 255), 1)  # bbox = orange
    cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)          # centroid = red
    cv2.putText(vis, f"c={circ:.2f} s={sol:.2f}", (bx, max(12, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
show("area circularity solidity moments bbox", vis,
     "contourArea = pixel area; arcLength = perimeter; circularity = 4*pi*A/P^2\n"
     "(1.0 = perfect circle, the elongated ellipse scores low); convexHull = tightest\n"
     "convex outline (blue), solidity = A/hull_area (dents lower it); moments -> centroid\n"
     "(m10/m00, m01/m00); boundingRect (orange) is used to detect border contact.")

# ------------------------------------------------------------------ 6. core / ring median
robot = min(cs, key=lambda c: np.linalg.norm(
    np.array([cv2.moments(c)["m10"] / cv2.moments(c)["m00"],
              cv2.moments(c)["m01"] / cv2.moments(c)["m00"]]) - (150, 200)))
R = float(np.sqrt(cv2.contourArea(robot) / np.pi))
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [robot], -1, 255, -1)
show("contour filled as mask", mask,
     "drawContours with thickness=-1 gives a filled binary mask of one contour.")

k = max(3, int(R * 0.5)) | 1
core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
show("core region (eroded mask)", core,
     f"Eroded by a {k}px ellipse. Pixel values of blur[core>0] are used for the median\n"
     f"-> robot darkness = {np.median(blur[core > 0]):.0f}.")

r1 = max(3, int(R * 0.4)) | 1
r2 = (max(5, int(R * 1.0)) | 1) * 2 + 1
d1 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r1, r1)))
d2 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r2, r2)))
ring_mask = ((d2 > 0) & (d1 == 0)).astype(np.uint8) * 255
show("ring region (background donut)", ring_mask,
     f"(dilate big) minus (dilate small) = donut around the robot.\n"
     f"Median there = local background = {np.median(blur[ring_mask > 0]):.0f}.")

# ------------------------------------------------------------------ 7. multi-threshold sweep
robot_int = float(np.median(blur[core > 0]))
bg = float(np.median(blur[ring_mask > 0]))
dark_lim = robot_int + 12
hi = max(dark_lim + 8, robot_int + 0.6 * (bg - robot_int))
Ts = np.linspace(dark_lim, hi, 10)
tiles = []
for T in Ts:
    m = (blur < T).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, OPEN_K)
    t = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    cv2.putText(t, f"T={T:.0f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    tiles.append(cv2.resize(t, (W // 2, H // 2)))
montage = np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])])
show("threshold sweep (find() loop)", montage,
     f"find() tries 10 thresholds between {dark_lim:.0f} and {hi:.0f} and keeps the best-\n"
     "scoring contour from any of them, which makes it robust to lighting drift.")

# ------------------------------------------------------------------ 8. partial robot + circle fit
def fit_circle_fixed_radius(pts, R, c0, iters=20):
    c = np.array(c0, float)
    for _ in range(iters):
        d = pts - c
        r = np.linalg.norm(d, axis=1) + 1e-9
        f = r - R
        J = -d / r[:, None]
        step, *_ = np.linalg.lstsq(J, -f, rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-3:
            break
    return c, float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - R) ** 2)))


part = None
for c in cs:
    bx, by, bw, bh = cv2.boundingRect(c)
    if bx + bw >= W and cv2.contourArea(c) > 100:
        part = c
if part is not None:
    pts = part.reshape(-1, 2).astype(float)
    free = pts[(pts[:, 0] >= 2) & (pts[:, 0] <= W - 3) & (pts[:, 1] >= 2) & (pts[:, 1] <= H - 3)]
    m = cv2.moments(part)
    naive = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    center, rms = fit_circle_fixed_radius(free, R, naive)
    vis = frame.copy()
    for p in free:
        cv2.circle(vis, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
    cv2.circle(vis, tuple(int(v) for v in naive), 4, (255, 0, 0), -1)        # blue: naive centroid
    cv2.circle(vis, tuple(int(v) for v in center), int(R), (0, 255, 255), 1)  # yellow: fitted circle
    cv2.circle(vis, tuple(int(v) for v in center), 3, (0, 0, 255), -1)       # red: fitted centre
    show("partial robot circle fit", vis,
         f"Green = visible edge points (border-hugging ones dropped). Blue = naive centroid\n"
         f"(biased inward, {naive.round(1)}). Red/yellow = Gauss-Newton fit of a circle of\n"
         f"KNOWN radius R={R:.1f} -> centre {center.round(1)} (truth is (470, 90)), rms={rms:.2f}px.")

# ------------------------------------------------------------------ 9. gating / candidate acceptance
vis = frame.copy()
A_ref = cv2.contourArea(robot)
circ_ref = 4 * np.pi * A_ref / cv2.arcLength(robot, True) ** 2
for c in cs:
    a = cv2.contourArea(c)
    if a < 15:
        continue
    p = cv2.arcLength(c, True)
    circ = 4 * np.pi * a / p ** 2
    mk = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mk, [c], -1, 255, -1)
    kk = max(3, int(R * 0.5)) | 1
    cr = cv2.erode(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    dark = float(np.median(blur[cr > 0] if cr.any() else blur[mk > 0]))
    ok = (abs(a / A_ref - 1) < 0.4 and circ > circ_ref - 0.15 and dark <= dark_lim)
    col = (0, 255, 0) if ok else (0, 0, 255)
    cv2.drawContours(vis, [c], -1, col, 2)
show("candidate gating", vis,
     "Green = passes area/circularity/darkness gates, red = rejected. (The real tracker also\n"
     "applies solidity and a max-distance gate from the last known position; the border-cut\n"
     "robot fails the *full-robot* area test here and is handled by the partial branch.)")

# ------------------------------------------------------------------ 10. video I/O
vid = os.path.join(ARGS.out, "synthetic.mp4")
vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for t in range(60):
    f = frame.copy()
    cv2.circle(f, (150 + 2 * t, 200), 6, (0, 0, 255), -1)
    vw.write(f)
vw.release()

cap = cv2.VideoCapture(vid)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
ok, f40 = cap.read()
cap.release()
if ok:
    show("VideoCapture seek to frame 40", f40,
         f"VideoCapture: CAP_PROP_FRAME_COUNT={n}, CAP_PROP_FPS={fps:.0f}. cap.set(\n"
         "CAP_PROP_POS_FRAMES, 40) jumps to frame 40; cap.read() returns (ok, frame).\n"
         "VideoWriter wrote this clip; the tracker only reads.")

# ------------------------------------------------------------------ 11. interactive UI (optional)
print("\nNot run here: cv2.selectROI (drag a rectangle, returns (x,y,w,h)), cv2.waitKey,\n"
      "cv2.destroyWindow. Try:  r = cv2.selectROI('t', frame); print(r)")
if not ARGS.no_display:
    cv2.destroyAllWindows()
print(f"\nAll step images saved to ./{ARGS.out}/")