#!/usr/bin/env python3
"""
Interactive micro-robot tracker (dark blob on lighter background), v2.

Workflow
--------
1. Step to a good frame (a / d), press 's', drag a rectangle around the robot.
2. Confirm the outline with 'y' ('n' to re-select).
3. First N frames: learn area, circularity, solidity, interior darkness.
4. Then everything is LOCKED. A candidate is accepted ONLY if it passes ALL gates:
     - interior darkness  <= learned robot darkness + margin   (robot never gets lighter)
     - area within learned tolerance
     - circularity & solidity close to learned values
     - position reachable from the last confirmed position at <= max speed
   If the blob touches the frame border it is treated as a PARTIAL robot:
   the centre is estimated by fitting a circle of the known radius to the
   visible edge points.
   If nothing passes, the frame is marked ABSENT (no guessing, no jumping).

Statuses: learn / track / PARTIAL / ABSENT

Usage:  python track_robot.py robot.mp4 [--start 0] [--learn-frames 30]
                              [--max-speed PX_PER_FRAME] [--dark-margin 12]
Outputs: <video>_track.csv  and  <video>_contours.npz
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

OPEN_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# ----------------------------------------------------------------- helpers
def prep(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(g, (5, 5), 0)


def contour_stats(c):
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    circ = 4 * np.pi * a / (p * p) if p > 0 else 0.0
    hull = cv2.contourArea(cv2.convexHull(c))
    sol = a / hull if hull > 0 else 0.0
    m = cv2.moments(c)
    cen = None
    if m["m00"] > 0:
        cen = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    return a, circ, sol, cen


def core_median(gray, contour, R):
    """Median gray level in the eroded interior of a contour."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    k = max(3, int(R * 0.5)) | 1
    core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    v = gray[core > 0] if core.any() else gray[mask > 0]
    return float(np.median(v)) if v.size else 255.0


def ring_median(gray, contour, R):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    r1 = max(3, int(R * 0.4)) | 1
    r2 = (max(5, int(R * 1.0)) | 1) * 2 + 1
    d1 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r1, r1)))
    d2 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r2, r2)))
    ring = gray[(d2 > 0) & (d1 == 0)]
    return float(np.median(ring)) if ring.size else 255.0


def fit_circle_fixed_radius(pts, R, c0, iters=20):
    """Gauss-Newton fit of a circle centre with known radius R."""
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
    rms = float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - R) ** 2)))
    return c, rms


def init_from_roi(gray, rect):
    x, y, w, h = [int(v) for v in rect]
    crop = gray[y:y + h, x:x + w]
    _, m = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, OPEN_K)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None

    def inside(c):
        bx, by, bw, bh = cv2.boundingRect(c)
        return bx > 0 and by > 0 and bx + bw < w and by + bh < h
    good = [c for c in cs if inside(c)] or cs
    c = max(good, key=cv2.contourArea)
    return c + np.array([[[x, y]]], dtype=c.dtype)


# ----------------------------------------------------------------- tracker
class RobotTracker:
    def __init__(self, gray, contour, max_speed=None, dark_margin=12.0):
        a, circ, sol, cen = contour_stats(contour)
        self.R = float(np.sqrt(a / np.pi))
        self.A_ref, self.circ_ref, self.sol_ref = a, circ, sol
        self.robot_int = core_median(gray, contour, self.R)
        self.bg = ring_median(gray, contour, self.R)
        self.max_speed = max_speed if max_speed else 0.5 * self.R   # px / frame
        self.base_margin = dark_margin

        # loose limits while learning, tightened in lock()
        self.tol, self.circ_tol, self.sol_tol = 0.6, 0.30, 0.15
        self.dark_margin = max(25.0, dark_margin)
        self.locked = False

        self.pos = cen
        self.vel = np.zeros(2)
        self.lost = 0
        self.hist = [(a, circ, sol, self.robot_int, self.bg)]

    # --- candidate search -------------------------------------------------
    def find(self, gray):
        H, W = gray.shape
        ref = self.pos + self.vel if self.lost == 0 else self.pos.copy()
        speed = float(np.linalg.norm(self.vel))
        # reachable distance: grows with time since last confirmed sighting
        gate = 1.5 * self.R + self.max_speed * (self.lost + 1) + 4 * speed
        half = int(gate + 2 * self.R)
        x0, y0 = max(0, int(ref[0]) - half), max(0, int(ref[1]) - half)
        x1, y1 = min(W, int(ref[0]) + half), min(H, int(ref[1]) + half)
        win = gray[y0:y1, x0:x1]
        if win.size == 0:
            return None
        wh, ww = win.shape

        dark_lim = self.robot_int + self.dark_margin
        hi = max(dark_lim + 8, self.robot_int + 0.6 * (self.bg - self.robot_int))
        off = np.array([[[x0, y0]]], dtype=np.int32)

        best = None
        for T in np.linspace(dark_lim, hi, 10):
            m = (win < T).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, OPEN_K)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cs:
                a, circ, sol, cen = contour_stats(c)
                if cen is None or a < 15:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                # clipped by the search window (not the frame) -> reject
                if (bx == 0 and x0 > 0) or (by == 0 and y0 > 0) or \
                   (bx + bw >= ww and x1 < W) or (by + bh >= wh and y1 < H):
                    continue
                touches_frame = (bx == 0 and x0 == 0) or (by == 0 and y0 == 0) or \
                                (bx + bw >= ww and x1 == W) or (by + bh >= wh and y1 == H)

                if a / self.A_ref > 1 + self.tol:
                    continue
                if core_median(win, c, self.R) > dark_lim:
                    continue                    # not dark enough -> not the robot
                cfull = c + off

                if not touches_frame:
                    # ---------------- full robot -----------------
                    if abs(a / self.A_ref - 1) > self.tol:
                        continue
                    if circ < self.circ_ref - self.circ_tol:
                        continue
                    if sol < self.sol_ref - self.sol_tol:
                        continue
                    center = cen + np.array([x0, y0])
                    dist = float(np.linalg.norm(center - ref))
                    if dist > gate:
                        continue
                    score = (abs(a / self.A_ref - 1) + 0.3 * dist / self.R
                             + abs(circ - self.circ_ref) + abs(sol - self.sol_ref))
                    kind = "full"
                else:
                    # ---------------- partial robot ----------------
                    if a < 0.08 * self.A_ref:
                        continue
                    pts = cfull.reshape(-1, 2).astype(float)
                    free = pts[(pts[:, 0] >= 2) & (pts[:, 0] <= W - 3) &
                               (pts[:, 1] >= 2) & (pts[:, 1] <= H - 3)]
                    if len(free) < 8:
                        continue
                    center, rms = fit_circle_fixed_radius(free, self.R, ref)
                    if rms > 0.12 * self.R:
                        continue                # visible edge is not a circular arc
                    if not (-self.R <= center[0] <= W + self.R and
                            -self.R <= center[1] <= H + self.R):
                        continue
                    dist = float(np.linalg.norm(center - ref))
                    if dist > gate:
                        continue
                    score = 3 * rms / self.R + 0.3 * dist / self.R + 0.2
                    kind = "partial"

                if best is None or score < best["score"]:
                    best = dict(score=score, kind=kind, contour=cfull, center=center)
        return best

    # --- state update -----------------------------------------------------
    def update(self, gray, found):
        if found is None:
            self.lost += 1
            self.vel = np.zeros(2)          # do not coast: stay at last confirmed spot
            return None
        c, center = found["contour"], found["center"]
        if self.lost == 0:
            pred = self.pos + self.vel
            resid = center - pred
            self.pos = pred + 0.7 * resid
            self.vel = self.vel + 0.3 * resid
        else:
            self.pos, self.vel = center.copy(), np.zeros(2)
        self.lost = 0
        if found["kind"] == "full" and not self.locked:
            a, circ, sol, _ = contour_stats(c)
            R = np.sqrt(a / np.pi)
            self.hist.append((a, circ, sol, core_median(gray, c, R),
                              ring_median(gray, c, R)))
        return c

    def lock(self):
        h = np.array(self.hist)
        a = np.median(h[:, 0])
        mad = np.median(np.abs(h[:, 0] - a)) * 1.4826
        self.A_ref = float(a)
        self.R = float(np.sqrt(a / np.pi))
        self.circ_ref = float(np.median(h[:, 1]))
        self.sol_ref = float(np.median(h[:, 2]))
        self.robot_int = float(np.median(h[:, 3]))
        self.bg = float(np.median(h[:, 4]))
        self.tol = float(np.clip(3 * mad / a + 0.10, 0.15, 0.40))
        self.circ_tol = float(max(0.08, 3 * np.std(h[:, 1])))
        self.sol_tol = float(max(0.05, 3 * np.std(h[:, 2])))
        self.dark_margin = float(max(self.base_margin, 4 * np.std(h[:, 3])))
        self.locked = True
        print(f"[locked] area={a:.0f}px R~{self.R:.1f}px circ={self.circ_ref:.2f}"
              f"+/-{self.circ_tol:.2f} solidity={self.sol_ref:.2f}+/-{self.sol_tol:.2f}\n"
              f"         area tol +/-{self.tol * 100:.0f}%  robot darkness={self.robot_int:.0f}"
              f" (accept <= {self.robot_int + self.dark_margin:.0f})  bg={self.bg:.0f}\n"
              f"         max speed={self.max_speed:.1f}px/frame")


# ----------------------------------------------------------------- UI
def pick_initial(cap, start, n_frames):
    idx = max(0, start)
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            idx = max(0, idx - 10)
            continue
        view = frame.copy()
        cv2.putText(view, f"frame {idx} | a/d: -/+10 frames | s: select robot | q: quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("setup", view)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            sys.exit(0)
        if k == ord("d"):
            idx = min(idx + 10, n_frames - 1)
        elif k == ord("a"):
            idx = max(idx - 10, 0)
        elif k == ord("s"):
            r = cv2.selectROI("setup", frame, showCrosshair=False, fromCenter=False)
            if r[2] == 0 or r[3] == 0:
                continue
            c = init_from_roi(prep(frame), r)
            if c is None:
                print("No dark blob found in that rectangle, try again.")
                continue
            a, circ, _, _ = contour_stats(c)
            view = frame.copy()
            cv2.drawContours(view, [c], -1, (0, 255, 0), 1)
            cv2.putText(view, f"area={a:.0f}px circ={circ:.2f} | y: confirm  n: re-select",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("setup", view)
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 in (ord("y"), 13):
                    cv2.destroyWindow("setup")
                    return idx, frame, c
                if k2 in (ord("n"), ord("q")):
                    break


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="F:/Mohit's BTP/Python Code/video_exp/videos/a2.mp4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--learn-frames", type=int, default=30)
    ap.add_argument("--max-speed", type=float, default=None,
                    help="max robot speed in px/frame (default 0.5 * radius)")
    ap.add_argument("--dark-margin", type=float, default=12.0,
                    help="allowed gray-level rise above learned robot darkness")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    idx0, frame0, c0 = pick_initial(cap, args.start, n_frames)
    tracker = RobotTracker(prep(frame0), c0, args.max_speed, args.dark_margin)

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    rows, contours = [], {}
    idx, first = idx0, True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = prep(frame)

        if first:
            contour, kind, center, first = c0, "full", tracker.pos, False
        else:
            found = tracker.find(gray)
            contour = tracker.update(gray, found)
            kind = found["kind"] if found else None
            center = found["center"] if found else None

        if not tracker.locked and (idx - idx0) + 1 >= args.learn_frames:
            tracker.lock()

        if contour is None:
            status = "ABSENT"
        elif kind == "partial":
            status = "PARTIAL"
        else:
            status = "track" if tracker.locked else "learn"

        if contour is not None:
            contours[idx] = contour.reshape(-1, 2)
            rows.append([idx, idx / fps, center[0], center[1],
                         cv2.contourArea(contour), status])
        else:
            rows.append([idx, idx / fps, float("nan"), float("nan"),
                         float("nan"), status])

        if not args.no_display:
            col = {"ABSENT": (0, 0, 255), "PARTIAL": (0, 255, 255)}.get(status, (0, 255, 0))
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, col, 1)
                cv2.circle(frame, tuple(int(v) for v in center), 3, (0, 0, 255), -1)
                if status == "PARTIAL":
                    cv2.circle(frame, tuple(int(v) for v in center),
                               int(tracker.R), (0, 255, 255), 1)
            label = "NO ROBOT" if status == "ABSENT" else status
            cv2.putText(frame, f"{idx} {label}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        idx += 1

    cap.release()
    cv2.destroyAllWindows()

    base = os.path.splitext(args.video)[0]
    with open(base + "_track.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x", "y", "area_px", "status"])
        w.writerows(rows)
    np.savez(base + "_contours.npz",
             frames=np.array(list(contours.keys())),
             contours=np.array(list(contours.values()), dtype=object))
    n_abs = sum(r[-1] == "ABSENT" for r in rows)
    n_par = sum(r[-1] == "PARTIAL" for r in rows)
    print(f"Done: {len(rows)} frames | {n_par} partial | {n_abs} absent. "
          f"Saved {base}_track.csv / _contours.npz")


if __name__ == "__main__":
    main()