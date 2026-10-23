#!/usr/bin/env python3
"""
Interactive micro-robot tracker (dark blob on lighter background), v2 + BENCHMARKING.

Same tracking logic as v2 (nothing algorithmic was changed), but every stage is timed
with time.perf_counter_ns() and a full report is printed and saved to results.json.

What is measured
----------------
Per frame (every frame of the run):
  read      cap.read()                      (video decode; replaced by camera grab in a live setup)
  prep      grayscale + Gaussian blur
  find      candidate search (tracker.find) -> broken down further into:
              find_setup, threshold, morph_open, find_contours, contour_stats,
              core_median, partial_fit, find_other
  update    state update (filter + learning stats)
  lock      one-off tracker.lock()
  log       storing row/contour
  draw      overlay drawing            (display mode only)
  show      imshow + waitKey(1)        (display mode only; waitKey(1) always blocks >= 1 ms)
  iter      whole loop iteration
  pipeline  prep + find + update       <-- the compute latency your controller will see
  e2e       read + pipeline
  gc        time spent in Python garbage collection during that frame
  counters  thresholds run, contours examined, search-window pixels, ...

Aggregates: mean/std/min/p50/p90/p95/p99/p99.9/max, achievable FPS, frame budget
(1/fps), number & streaks of over-budget frames, jitter, per-status timing,
drift over the run (quarters), histogram, slowest frames, hot-spot ranking.

Extra micro-benchmarks (repeated N times on a static frame, after the run):
  decode-only speed, each helper function, numpy-threshold vs cv2.threshold,
  full find(), full prep+find().

Usage
-----
Interactive ROI:     python track_robot_bench.py --video robot.mp4
Non-interactive:     python track_robot_bench.py --video robot.mp4 --start 0 --roi 310,220,40,40 --no-display
Pure compute timing: add --no-display (display + waitKey add several ms and are not part of "pipeline")

Outputs: results.json, <video>_timing.csv (per frame), <video>_track.csv, <video>_contours.npz
"""
import argparse
import csv
import gc
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

OPEN_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_now = time.perf_counter_ns

# ------------------------------------------------------------ profiling state
SEC = defaultdict(int)   # ns accumulated per section inside find(), reset every frame
CNT = defaultdict(int)   # counters, reset every frame


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
        t = _now()
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
            SEC["find_setup"] += _now() - t
            return None
        wh, ww = win.shape
        CNT["window_px"] = int(win.size)

        dark_lim = self.robot_int + self.dark_margin
        hi = max(dark_lim + 8, self.robot_int + 0.6 * (self.bg - self.robot_int))
        off = np.array([[[x0, y0]]], dtype=np.int32)
        SEC["find_setup"] += _now() - t

        best = None
        for T in np.linspace(dark_lim, hi, 10):
            CNT["thresholds"] += 1
            t = _now()
            m = (win < T).astype(np.uint8) * 255
            SEC["threshold"] += _now() - t

            t = _now()
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, OPEN_K)
            SEC["morph_open"] += _now() - t

            t = _now()
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            SEC["find_contours"] += _now() - t
            CNT["contours"] += len(cs)

            for c in cs:
                t = _now()
                a, circ, sol, cen = contour_stats(c)
                SEC["contour_stats"] += _now() - t
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
                t = _now()
                cm = core_median(win, c, self.R)
                SEC["core_median"] += _now() - t
                CNT["core_median_calls"] += 1
                if cm > dark_lim:
                    continue                    # not dark enough -> not the robot
                cfull = c + off
                CNT["candidates_scored"] += 1

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
                    t = _now()
                    pts = cfull.reshape(-1, 2).astype(float)
                    free = pts[(pts[:, 0] >= 2) & (pts[:, 0] <= W - 3) &
                               (pts[:, 1] >= 2) & (pts[:, 1] <= H - 3)]
                    if len(free) < 8:
                        SEC["partial_fit"] += _now() - t
                        continue
                    center, rms = fit_circle_fixed_radius(free, self.R, ref)
                    SEC["partial_fit"] += _now() - t
                    CNT["partial_fits"] += 1
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


def pick_initial_noninteractive(cap, start, roi):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Could not read frame {start}")
    c = init_from_roi(prep(frame), roi)
    if c is None:
        sys.exit("No dark blob found in the given --roi")
    return start, frame, c


# ----------------------------------------------------------------- benchmark utils
class GCWatch:
    """Measures time spent inside Python's garbage collector."""
    def __init__(self):
        self.t0 = 0
        self.total_ns = 0
        self.count = 0
        self.max_ns = 0

    def __call__(self, phase, info):
        if phase == "start":
            self.t0 = _now()
        else:
            d = _now() - self.t0
            self.total_ns += d
            self.count += 1
            self.max_ns = max(self.max_ns, d)


def stats(a_ms):
    a = np.asarray(a_ms, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    p = np.percentile(a, [50, 90, 95, 99, 99.9])
    return dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()), min=float(a.min()),
                p50=float(p[0]), p90=float(p[1]), p95=float(p[2]), p99=float(p[3]),
                p999=float(p[4]), max=float(a.max()), sum=float(a.sum()))


def max_streak(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def bench(fn, n, warm=10):
    for _ in range(warm):
        fn()
    ts = np.empty(n)
    for i in range(n):
        t = _now()
        fn()
        ts[i] = (_now() - t) / 1e6
    return stats(ts)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def system_info():
    info = dict(platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), python=sys.version.split()[0],
                opencv=cv2.__version__, numpy=np.__version__,
                cpu_count_logical=os.cpu_count(), cv2_threads=cv2.getNumThreads(),
                cv2_use_optimized=bool(cv2.useOptimized()))
    try:
        info["cv2_cpu_features"] = cv2.getCPUFeaturesLine()
    except Exception:
        pass
    if psutil:
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        try:
            f = psutil.cpu_freq()
            if f:
                info["cpu_freq_mhz_current"] = f.current
                info["cpu_freq_mhz_max"] = f.max
        except Exception:
            pass
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    else:
        info["note"] = "pip install psutil for CPU freq / RAM / memory-usage stats"
    return info


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6 if psutil else None


# ----------------------------------------------------------------- micro benchmarks
def micro_benchmarks(cap, idx0, frame0, c0, args):
    n = args.micro
    out = {}
    gray0 = prep(frame0)
    tr = RobotTracker(gray0, c0, args.max_speed, args.dark_margin)
    R = tr.R
    cen = tr.pos

    # window like find() uses
    gate = 1.5 * R + tr.max_speed + 0
    half = int(gate + 2 * R)
    H, W = gray0.shape
    x0, y0 = max(0, int(cen[0]) - half), max(0, int(cen[1]) - half)
    x1, y1 = min(W, int(cen[0]) + half), min(H, int(cen[1]) + half)
    win = np.ascontiguousarray(gray0[y0:y1, x0:x1])
    dark_lim = tr.robot_int + tr.dark_margin
    T = float(dark_lim + 4)
    mask = ((win < T).astype(np.uint8) * 255)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea) if cs else c0
    pts = c.reshape(-1, 2).astype(float)

    out["_info"] = dict(frame_shape=list(frame0.shape), search_window_shape=list(win.shape),
                        robot_radius_px=R, contour_points=int(len(pts)), repeats=n)
    out["prep (gray+blur) full frame"] = bench(lambda: prep(frame0), n)
    out["  cvtColor only"] = bench(lambda: cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), n)
    out["  GaussianBlur 5x5 only"] = bench(lambda: cv2.GaussianBlur(gray0, (5, 5), 0), n)
    out["threshold: numpy (win<T).astype*255 [current]"] = bench(
        lambda: (win < T).astype(np.uint8) * 255, n)
    out["threshold: cv2.threshold BINARY_INV [alt]"] = bench(
        lambda: cv2.threshold(win, T - 1, 255, cv2.THRESH_BINARY_INV), n)
    out["morphologyEx OPEN 3x3 (window)"] = bench(
        lambda: cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_K), n)
    out["findContours (window)"] = bench(
        lambda: cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE), n)
    out["contour_stats"] = bench(lambda: contour_stats(c), n)
    out["core_median (window)"] = bench(lambda: core_median(win, c, R), n)
    out["ring_median (window)"] = bench(lambda: ring_median(win, c, R), n)
    out["fit_circle_fixed_radius"] = bench(
        lambda: fit_circle_fixed_radius(pts, R, cen), n)
    out["tracker.find() on static frame"] = bench(lambda: tr.find(gray0), n)
    out["prep + find (full pipeline, static frame)"] = bench(
        lambda: tr.find(prep(frame0)), n)

    # decode only
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
    ts = []
    m = min(args.decode_frames, max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - idx0))
    for _ in range(m):
        t = _now()
        ok, _f = cap.read()
        ts.append((_now() - t) / 1e6)
        if not ok:
            break
    out["video decode only (sequential cap.read)"] = stats(ts)
    return out


# ----------------------------------------------------------------- reporting
def fmt_row(name, s, w=44):
    if s is None:
        return f"  {name:<{w}} (no data)"
    return (f"  {name:<{w}} {s['mean']:8.3f} {s['std']:8.3f} {s['min']:8.3f} {s['p50']:8.3f} "
            f"{s['p95']:8.3f} {s['p99']:8.3f} {s['max']:8.3f}")


def print_header(title, w=44):
    print(f"\n{title}")
    print(f"  {'':<{w}} {'mean':>8} {'std':>8} {'min':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}   [ms]")


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
    ap.add_argument("--no-display", action="store_true",
                    help="headless: gives the true compute timing (no imshow/waitKey)")
    # benchmarking options
    ap.add_argument("--roi", type=str, default=None,
                    help="x,y,w,h : skip the interactive selection (reproducible benchmarks)")
    ap.add_argument("--results", type=str, default="results.json")
    ap.add_argument("--warmup", type=int, default=5,
                    help="first N loop iterations excluded from steady-state statistics")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--micro", type=int, default=200,
                    help="repeats for micro-benchmarks (0 = skip)")
    ap.add_argument("--decode-frames", type=int, default=300)
    ap.add_argument("--no-per-frame", action="store_true",
                    help="do not store the per-frame arrays in results.json")
    ap.add_argument("--topk", type=int, default=10, help="slowest frames to list")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    budget_ms = 1000.0 / fps
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------ setup (timed separately)
    t_setup = _now()
    if args.roi:
        roi = [int(v) for v in args.roi.split(",")]
        idx0, frame0, c0 = pick_initial_noninteractive(cap, args.start, roi)
        setup_interactive = False
    else:
        idx0, frame0, c0 = pick_initial(cap, args.start, n_frames)
        setup_interactive = True
    setup_ms = (_now() - t_setup) / 1e6

    t = _now()
    tracker = RobotTracker(prep(frame0), c0, args.max_speed, args.dark_margin)
    tracker_init_ms = (_now() - t) / 1e6

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)

    # ------------------------------------------------ main loop
    gcw = GCWatch()
    gc.callbacks.append(gcw)
    rss_start = rss_mb()
    rss_peak = rss_start or 0.0
    cpu0, wall0 = time.process_time(), time.perf_counter()

    cols = defaultdict(list)
    rows, contours = [], {}
    idx, first, it = idx0, True, 0
    lock_ms, lock_iter = None, None

    while True:
        if args.max_frames and it >= args.max_frames:
            break
        SEC.clear()
        CNT.clear()
        gc_before = gcw.total_ns
        t_iter = _now()

        t = _now()
        ok, frame = cap.read()
        t_read = _now() - t
        if not ok:
            break

        t = _now()
        gray = prep(frame)
        t_prep = _now() - t

        t_find = t_update = 0
        if first:
            contour, kind, center, first = c0, "full", tracker.pos, False
        else:
            t = _now()
            found = tracker.find(gray)
            t_find = _now() - t
            t = _now()
            contour = tracker.update(gray, found)
            t_update = _now() - t
            kind = found["kind"] if found else None
            center = found["center"] if found else None

        t_lock = 0
        if not tracker.locked and (idx - idx0) + 1 >= args.learn_frames:
            t = _now()
            tracker.lock()
            t_lock = _now() - t
            lock_ms, lock_iter = t_lock / 1e6, it

        t = _now()
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
        t_log = _now() - t

        t_draw = t_show = 0
        stop = False
        if not args.no_display:
            t = _now()
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
            t_draw = _now() - t
            t = _now()
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                stop = True
            t_show = _now() - t

        t_iter_total = _now() - t_iter

        # ---- record
        f_sections = {k: v for k, v in SEC.items()}
        find_other = t_find - sum(f_sections.values()) if t_find else 0
        cols["frame"].append(idx)
        cols["status"].append(status)
        cols["read"].append(t_read / 1e6)
        cols["prep"].append(t_prep / 1e6)
        cols["find"].append(t_find / 1e6)
        cols["update"].append(t_update / 1e6)
        cols["lock"].append(t_lock / 1e6)
        cols["log"].append(t_log / 1e6)
        cols["draw"].append(t_draw / 1e6)
        cols["show"].append(t_show / 1e6)
        cols["iter"].append(t_iter_total / 1e6)
        cols["pipeline"].append((t_prep + t_find + t_update) / 1e6)
        cols["e2e"].append((t_read + t_prep + t_find + t_update) / 1e6)
        cols["gc"].append((gcw.total_ns - gc_before) / 1e6)
        for k in ("find_setup", "threshold", "morph_open", "find_contours",
                  "contour_stats", "core_median", "partial_fit"):
            cols["f_" + k].append(f_sections.get(k, 0) / 1e6)
        cols["f_find_other"].append(max(find_other, 0) / 1e6)
        for k in ("thresholds", "contours", "core_median_calls", "candidates_scored",
                  "partial_fits", "window_px"):
            cols["n_" + k].append(CNT.get(k, 0))

        if psutil and it % 100 == 0:
            rss_peak = max(rss_peak, rss_mb())

        it += 1
        idx += 1
        if stop:
            break

    wall_loop = time.perf_counter() - wall0
    cpu_loop = time.process_time() - cpu0
    gc.callbacks.remove(gcw)
    cap.release()
    cv2.destroyAllWindows()
    rss_end = rss_mb()
    if rss_end:
        rss_peak = max(rss_peak, rss_end)

    # ------------------------------------------------ saving original outputs (timed)
    t = _now()
    base = os.path.splitext(args.video)[0]
    with open(base + "_track.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x", "y", "area_px", "status"])
        w.writerows(rows)
    t_csv = (_now() - t) / 1e6
    t = _now()
    np.savez(base + "_contours.npz",
             frames=np.array(list(contours.keys())),
             contours=np.array(list(contours.values()), dtype=object))
    t_npz = (_now() - t) / 1e6
    n_abs = sum(r[-1] == "ABSENT" for r in rows)
    n_par = sum(r[-1] == "PARTIAL" for r in rows)
    print(f"Done: {len(rows)} frames | {n_par} partial | {n_abs} absent. "
          f"Saved {base}_track.csv / _contours.npz")

    # ------------------------------------------------ analysis
    A = {k: np.asarray(v) for k, v in cols.items() if k != "status"}
    status_arr = np.asarray(cols["status"])
    N = len(status_arr)
    wu = min(args.warmup, max(N - 1, 0))
    steady = np.arange(N) >= wu                       # steady-state mask

    stage_keys = ["read", "prep", "find", "update", "log", "draw", "show", "iter",
                  "pipeline", "e2e"]
    find_keys = ["find_setup", "threshold", "morph_open", "find_contours",
                 "contour_stats", "core_median", "partial_fit", "find_other"]

    timing = {k: stats(A[k][steady]) for k in stage_keys}
    timing["gc_per_frame"] = stats(A["gc"][steady])
    find_break = {k: stats(A["f_" + k][steady]) for k in find_keys}

    pipe = A["pipeline"][steady]
    e2e = A["e2e"][steady]
    it_ms = A["iter"][steady]
    over_pipe = pipe > budget_ms
    over_e2e = e2e > budget_ms
    over_iter = it_ms > budget_ms
    mean_pipe = float(pipe.mean()) if pipe.size else float("nan")

    budget = dict(
        video_fps=fps, frame_budget_ms=budget_ms,
        pipeline_mean_ms=mean_pipe,
        pipeline_budget_used_mean_pct=100 * mean_pipe / budget_ms,
        pipeline_budget_used_p99_pct=100 * float(np.percentile(pipe, 99)) / budget_ms if pipe.size else None,
        pipeline_budget_used_max_pct=100 * float(pipe.max()) / budget_ms if pipe.size else None,
        pipeline_frames_over_budget=int(over_pipe.sum()),
        pipeline_frames_over_budget_pct=100 * float(over_pipe.mean()) if pipe.size else None,
        pipeline_longest_over_budget_streak=max_streak(over_pipe),
        e2e_frames_over_budget=int(over_e2e.sum()),
        e2e_frames_over_budget_pct=100 * float(over_e2e.mean()) if e2e.size else None,
        iter_frames_over_budget=int(over_iter.sum()),
        iter_frames_over_budget_pct=100 * float(over_iter.mean()) if it_ms.size else None,
        max_sustainable_fps_pipeline_mean=1000.0 / mean_pipe if mean_pipe > 0 else None,
        max_sustainable_fps_pipeline_p99=1000.0 / float(np.percentile(pipe, 99)) if pipe.size else None,
        max_sustainable_fps_e2e_mean=1000.0 / float(e2e.mean()) if e2e.size and e2e.mean() > 0 else None,
        max_sustainable_fps_iter_mean=1000.0 / float(it_ms.mean()) if it_ms.size and it_ms.mean() > 0 else None,
        realtime_ok_p99=bool(pipe.size and np.percentile(pipe, 99) < budget_ms),
        realtime_ok_max=bool(pipe.size and pipe.max() < budget_ms),
    )

    jitter = dict(
        pipeline_std_ms=float(pipe.std()) if pipe.size else None,
        pipeline_mean_abs_frame_to_frame_diff_ms=float(np.mean(np.abs(np.diff(pipe)))) if pipe.size > 1 else None,
        pipeline_p99_minus_p50_ms=float(np.percentile(pipe, 99) - np.percentile(pipe, 50)) if pipe.size else None,
        pipeline_max_over_median=float(pipe.max() / np.median(pipe)) if pipe.size and np.median(pipe) > 0 else None,
    )

    # time per status
    per_status = {}
    for s in ("learn", "track", "PARTIAL", "ABSENT"):
        mk = steady & (status_arr == s)
        if mk.any():
            per_status[s] = dict(
                frames=int(mk.sum()),
                pipeline=stats(A["pipeline"][mk]),
                find=stats(A["find"][mk]),
                mean_thresholds=float(A["n_thresholds"][mk].mean()),
                mean_contours=float(A["n_contours"][mk].mean()),
                mean_candidates_scored=float(A["n_candidates_scored"][mk].mean()),
            )

    # hot-spot ranking (inside the pipeline)
    hot = []
    for k in ["prep"] + ["f_" + x for x in find_keys] + ["update"]:
        m = float(A[k][steady].mean()) if steady.any() else 0.0
        hot.append((k.replace("f_", "find."), m, 100 * m / mean_pipe if mean_pipe > 0 else 0.0))
    hot.sort(key=lambda r: -r[1])
    hotspots = [dict(stage=n, mean_ms=m, pct_of_pipeline=p) for n, m, p in hot]

    # drift over run (quarters)
    trend = []
    ps = A["pipeline"][steady]
    if ps.size >= 8:
        for i, ch in enumerate(np.array_split(ps, 4)):
            trend.append(dict(quarter=i + 1, mean_ms=float(ch.mean()), p99_ms=float(np.percentile(ch, 99))))

    # histogram
    hist = None
    if pipe.size:
        cnt, edges = np.histogram(pipe, bins=20)
        hist = dict(bin_edges_ms=edges, counts=cnt)

    # slowest frames
    order = np.argsort(-A["pipeline"])
    order = [i for i in order if steady[i]][:args.topk]
    worst = []
    for i in order:
        worst.append(dict(
            frame=int(A["frame"][i]), status=str(status_arr[i]),
            pipeline_ms=float(A["pipeline"][i]), prep_ms=float(A["prep"][i]),
            find_ms=float(A["find"][i]), update_ms=float(A["update"][i]),
            gc_ms=float(A["gc"][i]), thresholds=int(A["n_thresholds"][i]),
            contours=int(A["n_contours"][i]), candidates_scored=int(A["n_candidates_scored"][i]),
            window_px=int(A["n_window_px"][i])))

    counters = {k: stats(A["n_" + k][steady]) for k in
                ("thresholds", "contours", "core_median_calls", "candidates_scored",
                 "partial_fits", "window_px")}

    first_frames = [dict(loop_iter=i, frame=int(A["frame"][i]), status=str(status_arr[i]),
                         pipeline_ms=float(A["pipeline"][i]), iter_ms=float(A["iter"][i]))
                    for i in range(min(wu, N))]

    # ------------------------------------------------ micro benchmarks
    micro = None
    if args.micro > 0:
        print(f"\nRunning micro-benchmarks ({args.micro} repeats each)...")
        cap2 = cv2.VideoCapture(args.video)
        micro = micro_benchmarks(cap2, idx0, frame0, c0, args)
        cap2.release()

    # ------------------------------------------------ assemble results
    results = dict(
        meta=dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), script=os.path.basename(__file__),
                  args=vars(args), display_enabled=not args.no_display,
                  note=("display mode includes imshow+waitKey in 'iter' but NOT in 'pipeline'; "
                        "use --no-display for clean compute timing")),
        system=system_info(),
        video=dict(path=args.video, width=vid_w, height=vid_h, fps=fps, total_frames=n_frames,
                   start_frame=idx0, frames_processed=N, warmup_excluded=wu),
        tracker=dict(radius_px=tracker.R, area_px=tracker.A_ref, circ_ref=tracker.circ_ref,
                     sol_ref=tracker.sol_ref, max_speed_px_per_frame=tracker.max_speed,
                     locked=tracker.locked, lock_at_loop_iter=lock_iter),
        outcome=dict(frames=N, partial=int(n_par), absent=int(n_abs),
                     statuses={s: int((status_arr == s).sum()) for s in np.unique(status_arr)}),
        budget=budget,
        timing_ms=timing,
        find_breakdown_ms=find_break,
        hotspots=hotspots,
        jitter=jitter,
        per_status=per_status,
        counters_per_frame=counters,
        drift_quarters=trend,
        pipeline_histogram=hist,
        slowest_frames=worst,
        first_frames_cold_start=first_frames,
        one_off_ms=dict(setup=setup_ms, setup_was_interactive=setup_interactive,
                        tracker_init=tracker_init_ms, lock=lock_ms,
                        save_csv=t_csv, save_npz=t_npz),
        process=dict(loop_wall_s=wall_loop, loop_cpu_s=cpu_loop,
                     cpu_over_wall=cpu_loop / wall_loop if wall_loop > 0 else None,
                     processing_speed_x_realtime=(N / fps) / wall_loop if wall_loop > 0 else None,
                     loop_effective_fps=N / wall_loop if wall_loop > 0 else None,
                     rss_start_mb=rss_start, rss_end_mb=rss_end, rss_peak_mb=rss_peak or None),
        gc=dict(collections=gcw.count, total_ms=gcw.total_ns / 1e6, max_pause_ms=gcw.max_ns / 1e6),
        micro_benchmarks_ms=micro,
    )
    if not args.no_per_frame:
        results["per_frame_ms"] = {k: A[k] for k in A}
        results["per_frame_ms"]["status"] = cols["status"]

    with open(args.results, "w") as f:
        json.dump(clean(results), f, indent=2)

    # per-frame CSV
    tcsv = base + "_timing.csv"
    keys = [k for k in A]
    with open(tcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["status"] + keys)
        for i in range(N):
            w.writerow([status_arr[i]] + [A[k][i] for k in keys])

    # ------------------------------------------------ print report
    line = "=" * 100
    print(f"\n{line}\nBENCHMARK REPORT   ({N} frames, {wu} warm-up excluded, "
          f"{'DISPLAY ON' if not args.no_display else 'headless'})\n{line}")
    print(f"Video: {vid_w}x{vid_h} @ {fps:.1f} fps   frame budget = {budget_ms:.2f} ms")

    print_header("Per-frame timing (steady state)")
    labels = dict(read="read (decode)", prep="prep (gray+blur)", find="find (candidate search)",
                  update="update (filter/learn)", log="log (store row/contour)",
                  draw="draw overlay", show="imshow+waitKey", iter="WHOLE LOOP ITERATION",
                  pipeline="PIPELINE = prep+find+update", e2e="E2E = read+pipeline")
    for k in stage_keys:
        print(fmt_row(labels[k], timing[k]))
    print(fmt_row("gc time per frame", timing["gc_per_frame"]))

    print_header("Inside find()")
    for k in find_keys:
        print(fmt_row(k, find_break[k]))

    print("\nHot-spots (share of mean pipeline time)")
    for h in hotspots[:8]:
        bar = "#" * int(round(h["pct_of_pipeline"] / 2.5))
        print(f"  {h['stage']:<22} {h['mean_ms']:7.3f} ms  {h['pct_of_pipeline']:5.1f}%  {bar}")

    b = budget
    print("\nReal-time budget")
    print(f"  pipeline mean / p99 / max      : {mean_pipe:.3f} / {np.percentile(pipe, 99):.3f} / {pipe.max():.3f} ms"
          if pipe.size else "  (no data)")
    print(f"  budget used (mean / p99 / max) : {b['pipeline_budget_used_mean_pct']:.1f}% / "
          f"{b['pipeline_budget_used_p99_pct']:.1f}% / {b['pipeline_budget_used_max_pct']:.1f}%")
    print(f"  frames over budget (pipeline)  : {b['pipeline_frames_over_budget']} "
          f"({b['pipeline_frames_over_budget_pct']:.2f}%), longest streak {b['pipeline_longest_over_budget_streak']}")
    print(f"  frames over budget (read+pipe) : {b['e2e_frames_over_budget']} ({b['e2e_frames_over_budget_pct']:.2f}%)")
    print(f"  frames over budget (whole iter): {b['iter_frames_over_budget']} ({b['iter_frames_over_budget_pct']:.2f}%)")
    print(f"  max sustainable FPS (pipeline) : mean-based {b['max_sustainable_fps_pipeline_mean']:.0f}, "
          f"p99-based {b['max_sustainable_fps_pipeline_p99']:.0f}")
    print(f"  real-time OK at {fps:.0f} fps?   : p99 -> {b['realtime_ok_p99']}, worst case -> {b['realtime_ok_max']}")
    print(f"  jitter: std {jitter['pipeline_std_ms']:.3f} ms, p99-p50 {jitter['pipeline_p99_minus_p50_ms']:.3f} ms, "
          f"max/median {jitter['pipeline_max_over_median']:.1f}x" if pipe.size else "")

    print("\nBy status (pipeline ms, mean / p99 / max)")
    for s, d in per_status.items():
        p = d["pipeline"]
        print(f"  {s:<8} frames={d['frames']:<6} {p['mean']:7.3f} / {p['p99']:7.3f} / {p['max']:7.3f}   "
              f"thresholds/frame={d['mean_thresholds']:.1f} contours/frame={d['mean_contours']:.1f}")

    if trend:
        print("\nDrift over run (pipeline mean ms per quarter): " +
              "  ".join(f"Q{t['quarter']}={t['mean_ms']:.3f}" for t in trend))

    print(f"\nSlowest {len(worst)} frames (pipeline)")
    for w_ in worst:
        print(f"  frame {w_['frame']:<6} {w_['status']:<8} pipeline={w_['pipeline_ms']:7.3f} "
              f"(prep {w_['prep_ms']:.3f}, find {w_['find_ms']:.3f}, update {w_['update_ms']:.3f}, "
              f"gc {w_['gc_ms']:.3f})  contours={w_['contours']} window_px={w_['window_px']}")

    o = results["one_off_ms"]
    p_ = results["process"]
    print("\nOne-off / process")
    print(f"  tracker init {o['tracker_init']:.2f} ms | lock {o['lock'] if o['lock'] is None else round(o['lock'], 2)} ms | "
          f"save csv {o['save_csv']:.1f} ms | save npz {o['save_npz']:.1f} ms")
    print(f"  loop wall {p_['loop_wall_s']:.2f}s, cpu {p_['loop_cpu_s']:.2f}s, "
          f"speed {p_['processing_speed_x_realtime']:.2f}x real-time, {p_['loop_effective_fps']:.1f} fps effective")
    if psutil:
        print(f"  memory RSS start/peak/end: {p_['rss_start_mb']:.0f} / {p_['rss_peak_mb']:.0f} / {p_['rss_end_mb']:.0f} MB")
    print(f"  GC: {gcw.count} collections, total {gcw.total_ns / 1e6:.2f} ms, longest pause {gcw.max_ns / 1e6:.3f} ms")

    if micro:
        print_header("Micro-benchmarks (static frame, repeated)", w=48)
        for k, v in micro.items():
            if k == "_info":
                print(f"  [frame {v['frame_shape']}, search window {v['search_window_shape']}, "
                      f"R={v['robot_radius_px']:.1f}px, contour pts={v['contour_points']}]")
                continue
            print(fmt_row(k, v, w=48))

    print(f"\nSaved: {args.results}  |  {tcsv}")
    if not args.no_display:
        print("NOTE: display was ON. Re-run with --no-display for clean compute numbers.")


if __name__ == "__main__":
    main()