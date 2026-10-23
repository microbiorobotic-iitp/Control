"""
Live micro-robot tracker (dark blob on lighter background) - drop-in for your control loop.

Public API
----------
setup_tracker(cam, win, ...)   -> interactive: live view, draw box, learn from live frames, confirm
RobotTracker.step(gray)        -> dict(kind, contour, center, R) or None (robot absent)
RobotTracker.time_lost()       -> seconds since robot was last confirmed
RobotTracker.reset_clock()     -> call right before the control loop starts

Gates (a candidate must pass ALL):  interior darkness, area, circularity, solidity,
reachable position (max speed x elapsed time).  Blob touching the image border is treated
as a PARTIAL robot: centre from a fixed-radius circle fit on the visible arc.
"""
import time

import cv2
import numpy as np

OPEN_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# ----------------------------------------------------------------- helpers
def prep(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def contour_stats(c):
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    circ = 4 * np.pi * a / (p * p) if p > 0 else 0.0
    hull = cv2.contourArea(cv2.convexHull(c))
    sol = a / hull if hull > 0 else 0.0
    m = cv2.moments(c)
    cen = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] > 0 else None
    return a, circ, sol, cen


def core_median(gray, contour, R):
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
    c = np.array(c0, float)
    for _ in range(iters):
        d = pts - c
        r = np.linalg.norm(d, axis=1) + 1e-9
        J = -d / r[:, None]
        step, *_ = np.linalg.lstsq(J, -(r - R), rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-3:
            break
    rms = float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - R) ** 2)))
    return c, rms


def init_from_roi(gray_blur, rect):
    x, y, w, h = [int(v) for v in rect]
    crop = gray_blur[y:y + h, x:x + w]
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
    def __init__(self, gray_blur, contour, max_speed=None, dark_margin=12.0, now=None):
        """max_speed in px/second (default 20*radius)."""
        a, circ, sol, cen = contour_stats(contour)
        self.R = float(np.sqrt(a / np.pi))
        self.A_ref, self.circ_ref, self.sol_ref = a, circ, sol
        self.robot_int = core_median(gray_blur, contour, self.R)
        self.bg = ring_median(gray_blur, contour, self.R)
        self.max_speed = max_speed if max_speed else 20.0 * self.R
        self.base_margin = dark_margin

        self.tol, self.circ_tol, self.sol_tol = 0.6, 0.30, 0.15   # loose while learning
        self.dark_margin = max(25.0, dark_margin)
        self.locked = False

        self.pos = cen
        self.vel = np.zeros(2)                       # px/s
        self.lost = 0
        now = time.perf_counter() if now is None else now
        self.t_last = self.t_good = now
        self.last_kind = "full"
        self.hist = [(a, circ, sol, self.robot_int, self.bg)]
        self.last_contour = contour

    # --- timing -------------------------------------------------------------
    def reset_clock(self, now=None):
        now = time.perf_counter() if now is None else now
        self.t_last = self.t_good = now

    def time_lost(self, now=None):
        now = time.perf_counter() if now is None else now
        return 0.0 if self.lost == 0 else now - self.t_good

    # --- candidate search ---------------------------------------------------
    def _find(self, gray, now):
        H, W = gray.shape
        dt = float(np.clip(now - self.t_last, 1e-3, 0.5))
        ref = self.pos + self.vel * dt if self.lost == 0 else self.pos.copy()
        elapsed = max(now - self.t_good, dt)
        speed = float(np.linalg.norm(self.vel))
        gate = 1.5 * self.R + max(self.max_speed, 2 * speed) * elapsed
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
                if (bx == 0 and x0 > 0) or (by == 0 and y0 > 0) or \
                   (bx + bw >= ww and x1 < W) or (by + bh >= wh and y1 < H):
                    continue                           # clipped by search window only
                touches_frame = (bx == 0 and x0 == 0) or (by == 0 and y0 == 0) or \
                                (bx + bw >= ww and x1 == W) or (by + bh >= wh and y1 == H)
                if a / self.A_ref > 1 + self.tol:
                    continue
                if core_median(win, c, self.R) > dark_lim:
                    continue
                cfull = c + off

                if not touches_frame:
                    if abs(a / self.A_ref - 1) > self.tol:
                        continue
                    if circ < self.circ_ref - self.circ_tol or sol < self.sol_ref - self.sol_tol:
                        continue
                    center = cen + np.array([x0, y0])
                    dist = float(np.linalg.norm(center - ref))
                    if dist > gate:
                        continue
                    score = (abs(a / self.A_ref - 1) + 0.3 * dist / self.R
                             + abs(circ - self.circ_ref) + abs(sol - self.sol_ref))
                    kind = "full"
                else:
                    if a < 0.08 * self.A_ref:
                        continue
                    pts = cfull.reshape(-1, 2).astype(float)
                    free = pts[(pts[:, 0] >= 2) & (pts[:, 0] <= W - 3) &
                               (pts[:, 1] >= 2) & (pts[:, 1] <= H - 3)]
                    if len(free) < 8:
                        continue
                    center, rms = fit_circle_fixed_radius(free, self.R, ref)
                    if rms > 0.12 * self.R:
                        continue
                    if not (-self.R <= center[0] <= W + self.R and -self.R <= center[1] <= H + self.R):
                        continue
                    dist = float(np.linalg.norm(center - ref))
                    if dist > gate:
                        continue
                    score = 3 * rms / self.R + 0.3 * dist / self.R + 0.2
                    kind = "partial"

                if best is None or score < best["score"]:
                    best = dict(score=score, kind=kind, contour=cfull, center=center)
        return best

    # --- state update ---------------------------------------------------------
    def _update(self, gray, found, now):
        dt = float(np.clip(now - self.t_last, 1e-3, 0.5))
        if found is None:
            self.lost += 1
            self.vel = np.zeros(2)       # never coast: stay at last confirmed spot
            return
        c, center = found["contour"], found["center"]
        if self.lost == 0:
            pred = self.pos + self.vel * dt
            resid = center - pred
            self.pos = pred + 0.7 * resid
            self.vel = self.vel + 0.3 * resid / dt
        else:
            self.pos, self.vel = center.copy(), np.zeros(2)
        self.lost = 0
        self.t_good = now
        if found["kind"] == "full" and not self.locked:
            a, circ, sol, _ = contour_stats(c)
            R = np.sqrt(a / np.pi)
            self.hist.append((a, circ, sol, core_median(gray, c, R), ring_median(gray, c, R)))

    def step(self, gray, now=None):
        """gray: raw grayscale frame. Returns dict or None if robot absent."""
        now = time.perf_counter() if now is None else now
        g = prep(gray)
        found = self._find(g, now)
        self._update(g, found, now)
        self.t_last = now
        self.last_kind = found["kind"] if found else None
        self.last_contour = found["contour"] if found else None
        if found is None:
            return None
        return dict(kind=found["kind"], contour=found["contour"],
                    center=found["center"].copy(), R=self.R)

    def lock(self):
        h = np.array(self.hist)
        a = np.median(h[:, 0])
        mad = np.median(np.abs(h[:, 0] - a)) * 1.4826
        self.A_ref, self.R = float(a), float(np.sqrt(a / np.pi))
        self.circ_ref, self.sol_ref = float(np.median(h[:, 1])), float(np.median(h[:, 2]))
        self.robot_int, self.bg = float(np.median(h[:, 3])), float(np.median(h[:, 4]))
        self.tol = float(np.clip(3 * mad / a + 0.10, 0.15, 0.40))
        self.circ_tol = float(max(0.08, 3 * np.std(h[:, 1])))
        self.sol_tol = float(max(0.05, 3 * np.std(h[:, 2])))
        self.dark_margin = float(max(self.base_margin, 4 * np.std(h[:, 3])))
        self.locked = True
        print(f"[tracker locked] area={a:.0f}px R~{self.R:.1f}px circ={self.circ_ref:.2f}"
              f"+/-{self.circ_tol:.2f} solidity={self.sol_ref:.2f}+/-{self.sol_tol:.2f}\n"
              f"                 area tol +/-{self.tol * 100:.0f}%  robot darkness={self.robot_int:.0f} "
              f"(accept <= {self.robot_int + self.dark_margin:.0f})  bg={self.bg:.0f}  "
              f"max speed={self.max_speed:.0f}px/s")


# ----------------------------------------------------------------- interactive setup
def _text(img, s, y=25, col=(0, 255, 255)):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


def setup_tracker(cam, win, learn_frames=30, max_speed=None, dark_margin=12.0):
    """
    1. live view -> press S to freeze and drag a box around the robot (ENTER to accept box)
    2. tracker learns size/shape/darkness from the next `learn_frames` LIVE frames
    3. live view with the detected outline -> Y confirm / N re-select
    Returns a locked RobotTracker, or None if the user quits (Q).
    """
    while True:
        # --- phase 1: live view, wait for S
        while True:
            frame = cam.snapshot()
            view = frame.copy()
            _text(view, "Live view | S: select robot | Q: quit")
            cv2.imshow(win, view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                return None
            if k == ord("s"):
                break

        rect = cv2.selectROI(win, frame, showCrosshair=False, fromCenter=False)
        if rect[2] == 0 or rect[3] == 0:
            continue
        g0 = prep(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        c0 = init_from_roi(g0, rect)
        if c0 is None:
            print("No dark blob found in that box, try again.")
            continue
        tr = RobotTracker(g0, c0, max_speed, dark_margin)

        # --- phase 2: learn from upcoming live frames
        for i in range(learn_frames):
            frame = cam.snapshot()
            res = tr.step(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            view = frame.copy()
            if res is not None:
                cv2.drawContours(view, [res["contour"]], -1, (0, 255, 0), 1)
            _text(view, f"Learning robot shape {i + 1}/{learn_frames} ...")
            cv2.imshow(win, view)
            cv2.waitKey(1)
        if len(tr.hist) < max(5, learn_frames // 2):
            print("Could not follow the robot during learning, please re-select.")
            continue
        tr.lock()
        tr.reset_clock()

        # --- phase 3: confirm on live view
        while True:
            frame = cam.snapshot()
            res = tr.step(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            view = frame.copy()
            if res is not None:
                cv2.drawContours(view, [res["contour"]], -1, (0, 255, 0), 1)
                cv2.circle(view, tuple(int(v) for v in res["center"]), 3, (0, 0, 255), -1)
                _text(view, f"Is this the robot?  Y: confirm | N: re-select   ({res['kind']})")
            else:
                _text(view, "NO ROBOT FOUND   N: re-select", col=(0, 0, 255))
            cv2.imshow(win, view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("y"), 13):
                return tr
            if k == ord("n"):
                break
            if k == ord("q"):
                return None