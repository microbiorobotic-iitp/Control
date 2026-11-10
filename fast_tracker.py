"""
FastTracker - dark-blob tracker built for speed.

IDEA: do all the thinking ONCE (learn()), then every frame is just
        crop -> blur -> ONE per-pixel compare against a threshold map -> contour -> centroid.
learn() builds a uint8 threshold map T(x,y):
        T = local_background(x,y) * (1 - thr_frac*(1 - robot_dark/robot_bg))
  - local_background is estimated WITH THE ROBOT AND STATIC DARK THINGS MASKED OUT (normalised convolution),
    so the robot never has to be removed from the workspace.
  - excluded pixels (user mask + every other dark blob seen while learning = static) get T = 0, which can
    never be "darker than T", so they disappear from the frame without any per-frame masking cost.
  - the threshold sits half-way between background and robot darkness (half-max edge) => unbiased size.
Per frame there is no shape/circularity comparison, only an area sanity gate.

API compatible with the old tracker: step(gray) -> dict(kind, contour, center, R, ...) or None,
time_lost(), reset_clock().   center / R are in PIXELS; *_um keys are in micrometres.
"""
import time

import cv2
import numpy as np

from benchmark import NullTimer
from config import Cfg

pc = time.perf_counter


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


def _filled(shape, contour):
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    return m


class FastTracker:
    def __init__(self, um_per_px, cfg=None, timer=None):
        self.cfg = cfg or Cfg()
        self.um_per_px = float(um_per_px)
        self.tm = timer or NullTimer()
        self.rng = np.random.default_rng(0)
        self.excl_user = None        # uint8 255 = excluded by the user
        self.excl_auto = None        # uint8 255 = excluded automatically (static dark blobs)
        self.T_base = self.T = None
        self.contour0 = None
        self.valid = False
        self.info = ""
        self.last_shift = 0.0
        self.drift = 0
        self.ref0 = None
        self._bgc = None
        self.bg_msg = None
        self.collect_ns = 0

    # ------------------------------------------------------------------ state helpers
    def invalidate(self, why=""):
        self.valid = False
        self.info = why

    def set_scale(self, um_per_px):
        self.um_per_px = float(um_per_px)

    def reset_clock(self, now=None):
        now = pc() if now is None else now
        self.t_last = self.t_good = now

    def force_full_search(self):
        self.lost = max(self.lost, self.cfg.full_search_after)

    def time_lost(self, now=None):
        now = pc() if now is None else now
        return 0.0 if self.lost == 0 else now - self.t_good

    def set_user_mask(self, mask):
        """mask: uint8, 255 = never look here. Cheap: can be changed any time, even while tracking."""
        self.excl_user = None if mask is None else (mask > 0).astype(np.uint8) * 255
        self._compose()

    def _compose(self):
        if self.T_base is None:
            return
        T = self.T_base.copy()
        for m in (self.excl_user, self.excl_auto):
            if m is not None:
                T[m > 0] = 0
        self.T = T

    def exclusion_total(self):
        parts = [m for m in (self.excl_user, self.excl_auto) if m is not None]
        return np.maximum.reduce(parts) if parts else None

    # ------------------------------------------------------------------ learning (slow part, once)
    @staticmethod
    def _robot_contour(g, rect):
        x, y, w, h = [int(v) for v in rect]
        crop = g[y:y + h, x:x + w]
        _, m = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        inside = [c for c in cs if (lambda b: b[0] > 0 and b[1] > 0 and b[0] + b[2] < w and b[1] + b[3] < h)
                  (cv2.boundingRect(c))]
        c = max(inside or cs, key=cv2.contourArea)
        return c + np.array([[[x, y]]], dtype=c.dtype)

    def _estimate_bg(self, g, excluded, R):
        """Smooth local background, ignoring `excluded` pixels (normalised convolution at reduced size)."""
        H, W = g.shape
        s = 4 if min(H, W) >= 480 else 2
        sw, sh = max(1, W // s), max(1, H // s)
        small = cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        ex = cv2.resize(excluded, (sw, sh), interpolation=cv2.INTER_AREA)
        valid = (ex < 100).astype(np.float32)
        sigma = float(np.clip(self.cfg.bg_sigma_diams * 2 * R / s, 4, 80))
        num = cv2.GaussianBlur(small * valid, (0, 0), sigma)
        den = cv2.GaussianBlur(valid, (0, 0), sigma)
        fallback = float(np.median(small[valid > 0])) if valid.any() else 128.0
        bg = np.where(den > 1e-3, num / np.maximum(den, 1e-3), fallback).astype(np.float32)
        return cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)

    def learn(self, gray, rect, now=None):
        """gray: raw grayscale frame; rect: (x, y, w, h) box the user drew around the robot.
        Returns (ok, message)."""
        cfg = self.cfg
        H, W = gray.shape
        g = cv2.GaussianBlur(gray, (5, 5), 0)
        c = self._robot_contour(g, rect)
        if c is None:
            return False, "no dark blob found in the box"
        area = cv2.contourArea(c)
        if area < 12:
            return False, "blob in the box is tiny - draw the box tighter around the robot"
        R = float(np.sqrt(area / np.pi))
        rmask = _filled(g.shape, c)
        k = max(3, int(R * 0.5)) | 1
        core = cv2.erode(rmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        core = core if core.any() else rmask
        robot_int = float(np.median(g[core > 0]))
        m = cv2.moments(c)
        cen = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
        kd = int(2 * R * 0.6) | 1
        robot_dil = cv2.dilate(rmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        user = self.excl_user if self.excl_user is not None else np.zeros_like(rmask)
        auto = np.zeros_like(rmask)

        for p in range(max(1, cfg.bg_passes)):
            bg = self._estimate_bg(g, np.maximum.reduce([user, auto, robot_dil]), R)
            bg_robot = float(np.median(bg[core > 0]))
            if bg_robot - robot_int < 1.5 * cfg.min_margin:
                return False, (f"robot is not dark enough vs. its surroundings "
                               f"({bg_robot - robot_int:.0f} grey levels). Increase light / lower exposure")
            ratio = max(robot_int, 1.0) / bg_robot
            T = bg * (1.0 - cfg.thr_frac * (1.0 - ratio))
            T = np.minimum(T, bg - cfg.min_margin)
            if cfg.auto_exclude and p < cfg.bg_passes - 1:
                dark = ((g < T) & (user == 0)).astype(np.uint8)
                n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
                ids = np.bincount(lab[core > 0].ravel(), minlength=n)
                robot_lab = int(np.argmax(ids[1:]) + 1) if ids[1:].any() else -1
                keep = st[:, cv2.CC_STAT_AREA] >= 0.1 * area
                keep[0] = False
                if robot_lab > 0:
                    keep[robot_lab] = False
                auto = np.maximum(auto, (keep[lab]).astype(np.uint8) * 255)
                gk = cfg.auto_exclude_grow_px * 2 + 1
                auto = cv2.dilate(auto, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gk, gk)))
                auto[robot_dil > 0] = 0

        self.bg_smooth, self.ratio = bg, ratio
        self.excl_auto = auto if auto.any() else None
        self.T_base = np.clip(T, 0, 255).astype(np.uint8)
        self._compose()

        # reference pixels for global-brightness drift compensation
        ok = (self.T > 0) & (robot_dil == 0)
        ys, xs = np.nonzero(ok[::6, ::6])
        if len(ys) > 400:
            sel = self.rng.choice(len(ys), 400, replace=False)
            ys, xs = ys[sel], xs[sel]
        self.ref_ys, self.ref_xs = ys * 6, xs * 6
        self.ref0 = float(np.median(gray[self.ref_ys, self.ref_xs])) if len(ys) else None

        self.R, self.A_ref = R, area
        self.robot_int, self.bg_robot = robot_int, bg_robot
        self.diameter_um = 2 * R * self.um_per_px
        self.contour0 = c
        self.pos, self.vel, self.prev_meas = cen, np.zeros(2), None
        self.lost = 0
        now = pc() if now is None else now
        self.t_last = self.t_good = now
        self.valid = True
        self.info = (f"learned: diameter={self.diameter_um:.0f} um ({2 * R:.1f} px), robot grey={robot_int:.0f}, "
                     f"local bg={bg_robot:.0f}, threshold ~{robot_int + cfg.thr_frac * (bg_robot - robot_int):.0f} "
                     f"| auto-excluded {'yes' if self.excl_auto is not None else 'none'}")
        return True, self.info

    # ------------------------------------------------------------------ optional: exact static background
    # The robot never has to be removed: while it MOVES for ~1.5 s, the per-pixel temporal median of the
    # frames is a robot-free background (walls and shadows exactly as they are). The threshold map is then
    # rebuilt from it, which also lets the robot cross shadows. Tracking keeps running while collecting.
    def start_bg_collect(self, n=45):
        self._bgc = dict(n=n, frames=[], contours=[], centers=[], steps=0)
        self.bg_msg = None

    @property
    def collecting(self):
        return self._bgc is not None

    @property
    def bg_progress(self):
        return (len(self._bgc["frames"]), self._bgc["n"]) if self._bgc else (0, 0)

    def _bg_collect(self, gray, out):
        b = self._bgc
        b["steps"] += 1
        if out is not None and out["kind"] == "full":
            k = self.cfg.blur_k          # MUST match the live path or sharp static edges leave residuals
            b["frames"].append(cv2.GaussianBlur(gray, (k, k), 0) if k >= 3 else gray.copy())
            b["contours"].append(out["contour"])
            b["centers"].append(out["center"])
        if len(b["frames"]) >= b["n"] or b["steps"] > 4 * b["n"]:
            self.bg_msg = self._finish_bg_collect()

    def _finish_bg_collect(self):
        b, self._bgc = self._bgc, None
        n = len(b["frames"])
        if n < 10:
            return False, "background refine: robot was found in too few frames"
        extent = float(np.ptp(np.array(b["centers"]), axis=0).max())
        if extent < 6 * self.R:
            return False, (f"background refine: robot only moved {extent:.0f}px (< 3 diameters) - "
                           f"move it around during collection and press B again")
        med = np.median(np.stack(b["frames"]), axis=0).astype(np.float32)
        cover = np.zeros(med.shape, np.uint16)
        for c in b["contours"]:
            bx, by, bw, bh = cv2.boundingRect(c)
            patch = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(patch, [c], -1, 1, -1, offset=(-bx, -by))
            cover[by:by + bh, bx:bx + bw] += patch
        contam = cv2.dilate((cover > 0.4 * n).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        med = np.where(contam, self.bg_smooth, med)          # pixels the robot sat on: use the smooth estimate
        T = med * (1.0 - self.cfg.thr_frac * (1.0 - self.ratio))
        T = np.minimum(T, med - self.cfg.min_margin)
        self.T_base = np.clip(T, 0, 255).astype(np.uint8)
        self.excl_auto = None                                # static things are now in the background itself
        self._compose()
        return True, (f"background refined from {n} frames (robot travelled {extent:.0f}px); "
                      f"{100 * contam.mean():.2f}% of pixels were robot-contaminated and filled")

    # ------------------------------------------------------------------ per frame (fast part)
    def _search(self, gray, x0, y0, x1, y1, shift, ref):
        cfg, tm = self.cfg, self.tm
        H, W = gray.shape
        crop = gray[y0:y1, x0:x1]
        b = cv2.GaussianBlur(crop, (cfg.blur_k, cfg.blur_k), 0) if cfg.blur_k >= 3 else crop
        if shift:
            b = cv2.add(b, shift)
        tm.lap("blur")
        Tc = self.T[y0:y1, x0:x1]
        m = cv2.compare(b, Tc, cv2.CMP_LT)
        tm.lap("threshold")
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tm.lap("contours")
        wh, ww = crop.shape
        A = self.A_ref
        cands, clipped = [], False
        for c in cs:
            a = cv2.contourArea(c)
            if a < cfg.partial_min_frac * A or a > cfg.area_max_frac * A:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if (bx == 0 and x0 > 0) or (by == 0 and y0 > 0) or (bx + bw >= ww and x1 < W) or (by + bh >= wh and y1 < H):
                clipped = True                              # cut by the search window only
                continue
            frame_edge = (bx == 0 and x0 == 0) or (by == 0 and y0 == 0) or (bx + bw >= ww and x1 == W) or (by + bh >= wh and y1 == H)
            if not frame_edge and a < cfg.area_min_frac * A:
                continue
            cands.append((a, c, (bx, by, bw, bh), frame_edge))
        if not cands:
            tm.lap("select")
            return None, clipped
        if len(cands) > 1:                                  # several blobs: take the one nearest the prediction
            def dist(t):
                bx, by, bw, bh = t[2]
                return np.hypot(x0 + bx + bw / 2 - ref[0], y0 + by + bh / 2 - ref[1])
            cands.sort(key=dist)
        a, c, (bx, by, bw, bh), frame_edge = cands[0]
        tm.lap("select")

        if not frame_edge:
            # sub-pixel centre: darkness-weighted centroid inside the blob
            xa, ya, xb, yb = max(bx - 1, 0), max(by - 1, 0), min(bx + bw + 1, ww), min(by + bh + 1, wh)
            w = cv2.subtract(Tc[ya:yb, xa:xb], b[ya:yb, xa:xb])
            buf = np.zeros(w.shape, np.uint8)
            cv2.drawContours(buf, [c], -1, 1, -1, offset=(-xa, -ya))
            w = w * buf
            mm = cv2.moments(w)
            if mm["m00"] > 0:
                center = np.array([x0 + xa + mm["m10"] / mm["m00"], y0 + ya + mm["m01"] / mm["m00"]])
            else:
                mc = cv2.moments(c)
                center = np.array([x0 + mc["m10"] / mc["m00"], y0 + mc["m01"] / mc["m00"]])
            kind, R = "full", float(np.sqrt(a / np.pi))
        else:
            pts = c.reshape(-1, 2).astype(float) + [x0, y0]
            free = pts[(pts[:, 0] >= 2) & (pts[:, 0] <= W - 3) & (pts[:, 1] >= 2) & (pts[:, 1] <= H - 3)]
            if len(free) < 8:
                tm.lap("centroid")
                return None, clipped
            center, rms = fit_circle_fixed_radius(free, self.R, ref)
            if rms > 0.15 * self.R:
                tm.lap("centroid")
                return None, clipped
            kind, R = "partial", self.R
        tm.lap("centroid")
        return dict(kind=kind, contour=c + np.array([[[x0, y0]]], dtype=c.dtype), center=center, R=R), clipped

    def step(self, gray, now=None):
        """gray: raw uint8 grayscale frame.  Returns dict or None (robot not found)."""
        if not self.valid:
            return None
        cfg, tm = self.cfg, self.tm
        t_start = time.perf_counter_ns()
        now = pc() if now is None else now
        H, W = gray.shape
        if now < self.t_last:                                # video looped / clock reset
            self.t_last = self.t_good = now
        dt = float(np.clip(now - self.t_last, 1e-3, 0.5))
        ref = self.pos + self.vel * dt if self.lost == 0 else self.pos.copy()
        elapsed = max(now - self.t_good, dt)
        vmax = max(cfg.max_speed_um_s / self.um_per_px, 2.0 * float(np.linalg.norm(self.vel)))
        half = int(cfg.win_R_factor * self.R + vmax * elapsed)
        full = self.lost >= cfg.full_search_after or 2 * half >= 0.8 * min(H, W)
        if full:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, y0 = max(0, int(ref[0]) - half), max(0, int(ref[1]) - half)
            x1, y1 = min(W, int(ref[0]) + half), min(H, int(ref[1]) + half)
        shift = 0
        if cfg.drift_comp and self.ref0 is not None:
            shift = int(np.clip(round(self.ref0 - float(np.median(gray[self.ref_ys, self.ref_xs]))), -60, 60))
            self.drift = shift
        tm.lap("drift")

        found, clipped = self._search(gray, x0, y0, x1, y1, shift, ref)
        if found is None and clipped and not full:           # robot is bigger than the window -> whole frame
            found, _ = self._search(gray, 0, 0, W, H, shift, ref)
            x0, y0, x1, y1 = 0, 0, W, H

        out = None
        if found is None:
            self.lost += 1
            self.vel = np.zeros(2)                           # never coast: stay at last confirmed spot
        else:
            center = found["center"]
            raw_v = np.zeros(2)
            if self.lost == 0:
                pred = self.pos + self.vel * dt
                resid = center - pred
                self.pos = pred + cfg.alpha * resid
                self.vel = self.vel + cfg.beta * resid / dt
                if self.prev_meas is not None:
                    raw_v = (center - self.prev_meas) / dt
            else:
                self.pos, self.vel = center.copy(), np.zeros(2)
            self.prev_meas = center.copy()
            self.lost = 0
            self.t_good = now
            upp = self.um_per_px
            n = cfg.boundary_points
            cc = found["contour"].reshape(-1, 2)
            boundary = cc[np.linspace(0, len(cc) - 1, n).astype(int)]
            v_um = self.vel * upp
            out = dict(kind=found["kind"], contour=found["contour"], center=center.copy(), R=found["R"],
                       center_um=center * upp, R_um=found["R"] * upp, boundary=boundary,
                       vel_um_s=v_um, speed_um_s=float(np.linalg.norm(v_um)),
                       vel_raw_um_s=raw_v * upp, window=(x0, y0, x1, y1), t=now, drift=shift)
        self.t_last = now
        tm.lap("update")
        tm.add("TRACKER_TOTAL", time.perf_counter_ns() - t_start)
        self.collect_ns = 0
        if self._bgc is not None:
            t_c = time.perf_counter_ns()
            self._bg_collect(gray, out)
            self.collect_ns = time.perf_counter_ns() - t_c
            tm.add("bg_collect", self.collect_ns)
        return out
