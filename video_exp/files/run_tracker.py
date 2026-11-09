#!/usr/bin/env python3
"""
Live micro-robot tracker.

  python run_tracker.py --video test.avi                      (interactive, plays at native speed)
  python run_tracker.py --video test.avi --fast --no-gui --init 615 577 50 50     (pure benchmark)
  python run_tracker.py --camera 0 --mag 2

Keys (click the video window first):
  R  pick robot (draw a box around it, ENTER)   -> learns size / darkness, builds the threshold map
  K  mask editor (paint where the robot can never be)       S  save mask to mask.png
  C  camera panel: exposure / gain / lock auto-exposure (live camera only)
  [ ]  change magnification (tracker is invalidated -> re-mask + pick robot again)
  SPACE  tracking ON/OFF (resumes with a full-frame search)     P  freeze video
  B  refine background from the robot's own motion (keep it moving ~2 s; fixes robot crossing shadows)
  V  show/hide mask outlines   T  trail   H  print timing table   ENTER  accept after learning   Q/ESC  quit
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

import ui
from benchmark import Timer
from camera import CameraSource, VideoSource
from config import Cfg, Scale
from fast_tracker import FastTracker

_ns = time.perf_counter_ns
WIN = "micro-robot tracker"


class ShiftMonitor:
    """Warns when the whole workspace has moved: phase correlation of 1/4-size images every N frames.
    The robot (old + current position) is blanked in both images so only STATIC structure is compared."""

    def __init__(self, every, warn_px):
        self.every, self.warn, self.ref = every, warn_px, None
        self.dx = self.dy = self.resp = 0.0
        self.alarm = False

    @staticmethod
    def _small(g):
        s = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        return s - s.mean()

    @staticmethod
    def _blank(img, c, R):
        cv2.circle(img, (int(c[0] / 4), int(c[1] / 4)), int(R * 2 / 4) + 3, 0, -1)

    def reset(self, gray, c, R):
        self.ref = self._small(gray)
        self._blank(self.ref, c, R)
        self.c0, self.R = np.array(c, float), R
        self.win = cv2.createHanningWindow(self.ref.shape[::-1], cv2.CV_32F)
        self.alarm, self.dx, self.dy = False, 0.0, 0.0

    def check(self, gray, idx, c):
        if not self.every or self.ref is None or idx % self.every or c is None:
            return
        cur, ref = self._small(gray), self.ref.copy()
        for img in (cur, ref):
            self._blank(img, self.c0, self.R)
            self._blank(img, c, self.R)
        (dx, dy), self.resp = cv2.phaseCorrelate(ref, cur, self.win)
        self.dx, self.dy = dx * 4, dy * 4
        self.alarm = self.resp > 0.05 and float(np.hypot(self.dx, self.dy)) > self.warn


def outlines_of(mask):
    if mask is None or not mask.any():
        return []
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cs


def put(view, text, y, col=(0, 255, 255)):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(view, (3, y - h - 4), (13 + w, y + b), (0, 0, 0), -1)      # solid backing = always legible
    cv2.putText(view, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)


def draw(frame, s, res, tr, hud, outl_user, outl_auto, show_mask, trail, learn_contour):
    view = frame.copy() if s == 1 else cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    sc = lambda a: (np.asarray(a, np.float32) * s).astype(np.int32)
    if show_mask:
        for cs, col in ((outl_user, (0, 0, 255)), (outl_auto, (0, 128, 255))):
            if cs:
                cv2.polylines(view, [sc(c) for c in cs], True, col, 1)
    if learn_contour is not None:
        cv2.polylines(view, [sc(learn_contour)], True, (0, 255, 0), 2)
    if res is not None:
        x0, y0, x1, y1 = res["window"]
        cv2.rectangle(view, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (90, 90, 90), 1)
        if len(trail) > 1:
            cv2.polylines(view, [sc(np.array(trail))], False, (255, 200, 0), 1)
        cv2.polylines(view, [sc(res["contour"])], True, (0, 255, 0) if res["kind"] == "full" else (0, 200, 255), 1)
        for p in res["boundary"]:
            cv2.circle(view, tuple(sc(p)), 2, (0, 255, 255), -1)
        c = res["center"]
        cv2.circle(view, tuple(sc(c)), 3, (0, 0, 255), -1)
        v_px = res["vel_um_s"] / tr.um_per_px
        cv2.arrowedLine(view, tuple(sc(c)), tuple(sc(c + v_px * 0.25)), (255, 255, 0), 2, tipLength=0.3)
    for i, (txt, col) in enumerate(hud):
        put(view, txt, 20 + 20 * i, col)
    return view


def evaluate_gt(path, rows, fps, upp):
    gt = np.loadtxt(path, delimiter=",", skiprows=1)
    est = {r[0]: r for r in rows if r[1] is not None}
    err, verr = [], []
    for i in range(1, len(gt) - 1):
        f = int(gt[i, 0])
        if f in est:
            _, c, v = est[f]
            err.append(np.hypot(*(c - gt[i, 1:3])))
            vg = (gt[i + 1, 1:3] - gt[i - 1, 1:3]) * fps / 2
            if f > 20:
                verr.append(np.hypot(*(v / upp - vg)))
    err, verr = np.array(err), np.array(verr)
    print(f"\n[ground truth] detected {len(err)}/{len(gt) - 2} frames | position error px: mean {err.mean():.2f}  "
          f"median {np.median(err):.2f}  p95 {np.percentile(err, 95):.2f}  max {err.max():.2f}")
    if len(verr):
        print(f"[ground truth] velocity error px/s: median {np.median(verr):.1f}  p95 {np.percentile(verr, 95):.1f}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--camera", type=int)
    ap.add_argument("--mag", default=None, help="key from magnification.json, e.g. 2")
    ap.add_argument("--init", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="robot box (skips the GUI selection)")
    ap.add_argument("--mask", help="PNG mask to load (white = excluded)")
    ap.add_argument("--csv", default="track_out.csv")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--fast", action="store_true", help="video: don't wait for real time")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--preview-scale", type=float, default=0.0, help="0 = auto")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--gt", help="ground-truth csv from gen_test_video.py")
    ap.add_argument("--max-speed", type=float, help="um/s")
    ap.add_argument("--refine-bg", action="store_true", help="start background refinement right after --init")
    ap.add_argument("--no-auto-exclude", action="store_true", help="do not auto-mask other dark blobs")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    a = ap.parse_args()

    cfg = Cfg()
    if a.max_speed:
        cfg.max_speed_um_s = a.max_speed
    if a.no_auto_exclude:
        cfg.auto_exclude = False
    scale = Scale(key=a.mag)
    tm = Timer()
    src = VideoSource(a.video, a.loop) if a.video else CameraSource(a.camera, a.width, a.height)
    scale.check_resolution(*src.size)
    if not scale.verified:
        print(f"[warn] magnification {scale.key}x um/px={scale.um_per_px} is NOT verified in magnification.json")
    tr = FastTracker(scale.um_per_px, cfg, tm)
    mon = ShiftMonitor(cfg.shift_check_every, cfg.shift_warn_px)
    gui = not a.no_gui
    if not gui and not a.init:
        sys.exit("--no-gui needs --init X Y W H")
    pv = a.preview_scale or min(1.0, 1280.0 / src.size[0])
    if gui:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    mask = None
    if a.mask and os.path.exists(a.mask):
        mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
        tr.set_user_mask(mask)

    first = src.read()
    if first is None:
        sys.exit("no frames")
    hold = first
    state, tracking_on, frozen_user = "SETUP", True, False
    msg, msg_col = "", (0, 255, 255)
    show_mask, show_trail = True, True
    trail = deque(maxlen=90)
    outl_user, outl_auto, dirty = [], [], True
    rows, fcsv = [], open(a.csv, "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["frame", "t_s", "kind", "x_px", "y_px", "x_um", "y_um", "vx_um_s", "vy_um_s", "speed_um_s",
                 "vrawx_um_s", "vrawy_um_s", "R_um", "mag", "drift"])
    n_found = n_lost = n_proc = 0
    fps_ema, t_prev = 0.0, time.perf_counter()
    res = None

    def do_learn(rect, frame, gray, t):
        nonlocal state, msg, msg_col, dirty
        ok, m = tr.learn(gray, rect, t)
        msg, msg_col = m, ((0, 255, 0) if ok else (0, 0, 255))
        print("[learn]", m)
        if ok:
            mon.reset(gray, tr.pos, tr.R)
            dirty = True
            trail.clear()
            state = "TRACK" if not gui else "CONFIRM"
        return ok

    if a.init:
        f0 = cv2.cvtColor(hold[0], cv2.COLOR_BGR2GRAY)
        if not do_learn(a.init, hold[0], f0, hold[1]):
            sys.exit("learning failed")
        state = "TRACK"
        if a.refine_bg:
            tr.start_bg_collect()

    while True:
        it0 = time.perf_counter()
        frozen = state == "CONFIRM" or (state == "SETUP" and not src.is_live) or frozen_user
        tm.start()
        if not frozen:
            item = src.read()
            if item is None:
                break
            hold = item
        tm.lap("grab")
        frame, t, idx = hold
        t_a = _ns()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tm.lap("gray")

        res = None
        if state == "TRACK" and tracking_on and tr.valid and not frozen:
            res = tr.step(gray, t)
            tm.add("FRAME_TOTAL", _ns() - t_a - tr.collect_ns)
            n_proc += 1
            if res is not None:
                n_found += 1
                trail.append(res["center"])
                c, cu, v, vr = res["center"], res["center_um"], res["vel_um_s"], res["vel_raw_um_s"]
                wr.writerow([idx, f"{t:.4f}", res["kind"], f"{c[0]:.3f}", f"{c[1]:.3f}", f"{cu[0]:.2f}", f"{cu[1]:.2f}",
                             f"{v[0]:.1f}", f"{v[1]:.1f}", f"{res['speed_um_s']:.1f}", f"{vr[0]:.1f}", f"{vr[1]:.1f}",
                             f"{res['R_um']:.2f}", scale.key, res["drift"]])
                rows.append((idx, c.copy(), v.copy()))
            else:
                n_lost += 1
                wr.writerow([idx, f"{t:.4f}", "lost"] + [""] * 11 + [scale.key, ""])
                rows.append((idx, None, None))
            t_m = _ns()
            was = mon.alarm
            mon.check(gray, idx, None if res is None else res['center'])
            if mon.alarm and not was:
                print(f'[shift] frame {idx}: workspace moved ~({mon.dx:+.1f},{mon.dy:+.1f}) px (resp {mon.resp:.2f}) - re-pick robot [R]')
            if mon.every and idx % mon.every == 0:
                tm.add("shift_check", _ns() - t_m)
            if a.max_frames and n_proc >= a.max_frames:
                break

        tm.lap("log")
        if tr.bg_msg is not None:
            msg, msg_col = tr.bg_msg[1], ((0, 255, 0) if tr.bg_msg[0] else (0, 0, 255))
            print('[bg]', msg)
            tr.bg_msg, dirty = None, True
        if gui:
            if dirty:
                outl_user, outl_auto = outlines_of(tr.excl_user), outlines_of(tr.excl_auto)
                dirty = False
            now = time.perf_counter()
            fps_ema = 0.9 * fps_ema + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            hud = [(f"{state} | tracking {'ON' if tracking_on else 'OFF'} | {scale.key}x = {scale.um_per_px:.3f} um/px"
                    f"{'' if scale.verified else ' (UNVERIFIED)'} | {fps_ema:.0f} fps", (0, 255, 255))]
            if res is not None:
                hud.append((f"x={res['center_um'][0]:.0f} y={res['center_um'][1]:.0f} um | speed={res['speed_um_s']:.0f} um/s | "
                            f"{res['kind']} | tracker {tm.data['TRACKER_TOTAL'][-1] / 1e6:.2f} ms | drift {res['drift']:+d}", (0, 255, 0)))
            if state == "SETUP":
                hud.append(("SETUP: [K] paint mask  [R] pick robot" + ("  [C] camera" if src.is_live else ""), (0, 255, 255)))
            elif state == "CONFIRM":
                hud.append(("Is the green outline the robot?  ENTER accept | R re-pick | K edit mask", (0, 255, 255)))
            elif state == "TRACK" and tracking_on and res is None:
                hud.append((f"ROBOT NOT FOUND (lost {tr.time_lost(t):.2f}s) - searching; if workspace moved press R", (0, 0, 255)))
            if tr.collecting:
                hud.append((f"collecting background {tr.bg_progress[0]}/{tr.bg_progress[1]} - keep the robot MOVING", (0, 165, 255)))
            if mon.alarm:
                hud.append((f"WORKSPACE MOVED ~({mon.dx:+.1f},{mon.dy:+.1f}) px - mask/threshold map out of date: press R", (0, 0, 255)))
            if msg:
                hud.append((msg, msg_col))
            view = draw(frame, pv, res, tr, hud, outl_user, outl_auto, show_mask, trail if show_trail else [],
                        tr.contour0 if state == "CONFIRM" else None)
            tm.lap("draw")
            delay = 1 if (src.is_live or a.fast) else max(1, int(1000 / src.fps - (time.perf_counter() - it0) * 1000))
            cv2.imshow(WIN, view)
            k = cv2.waitKey(delay) & 0xFF
            tm.lap("show")
        else:
            k = 255
            if not (src.is_live or a.fast):
                time.sleep(max(0.0, 1 / src.fps - (time.perf_counter() - it0)))

        # ---------------- keys
        if k in (ord("q"), 27):
            break
        elif k in (13, 10) and state == "CONFIRM":
            state, msg = "TRACK", ""
            tr.reset_clock(t)
        elif k == ord(" "):
            tracking_on = not tracking_on
            if tracking_on and tr.valid:
                tr.force_full_search()
                tr.reset_clock(t)
            trail.clear()
        elif k == ord("p"):
            frozen_user = not frozen_user
        elif k == ord("v"):
            show_mask = not show_mask
        elif k == ord("t"):
            show_trail = not show_trail
        elif k == ord("h"):
            print(tm.summary())
        elif k == ord("b") and tr.valid and state == "TRACK":
            tr.start_bg_collect()
            tracking_on = True
        elif k == ord("r"):
            view_f = frame.copy() if pv == 1 else cv2.resize(frame, None, fx=pv, fy=pv)
            cv2.putText(view_f, "Draw a box around the robot, press ENTER (c = cancel)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            rect = cv2.selectROI(WIN, view_f, showCrosshair=False)
            if rect[2] > 0 and rect[3] > 0:
                do_learn([v / pv for v in rect], frame, gray, t)
        elif k == ord("k"):
            m_in = tr.exclusion_total()          # user + automatic zones: everything you see is editable
            big = frame if pv == 1 else cv2.resize(frame, None, fx=pv, fy=pv)
            m_small = None if m_in is None else cv2.resize(m_in, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST)
            m_new = ui.edit_mask(WIN, big, m_small)
            m_full = cv2.resize(m_new, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) if pv != 1 else m_new
            tr.excl_auto = None                  # the edited mask now replaces the automatic one
            tr.set_user_mask(m_full)
            dirty = True
        elif k == ord("s") and tr.excl_user is not None:
            cv2.imwrite("mask.png", tr.excl_user)
            msg, msg_col = "mask saved to mask.png", (0, 255, 0)
        elif k == ord("c") and src.is_live:
            ui.tune_camera(WIN, src, lambda: (src.read() or (None,))[0])
            msg = ""
        elif k in (ord("["), ord("]")):
            scale.step(-1 if k == ord("[") else 1)
            old = tr.um_per_px
            tr.set_scale(scale.um_per_px)
            tr.invalidate("magnification changed")
            tr.set_user_mask(None)
            tr.excl_auto = None
            dirty, state, res = True, "SETUP", None
            hint = f" robot should now be ~{2 * tr.R * old / scale.um_per_px:.0f}px across" if hasattr(tr, "R") else ""
            msg, msg_col = f"magnification -> {scale.key}x ({scale.um_per_px:.3f} um/px);{hint}. Redo mask [K], pick robot [R]", (0, 165, 255)

    fcsv.close()
    src.release()
    if gui:
        cv2.destroyAllWindows()
    print(f"\nframes tracked: {n_proc}  found: {n_found}  lost: {n_lost}   -> {a.csv}")
    if n_proc:
        print(tm.summary())
    if a.gt and rows:
        evaluate_gt(a.gt, rows, src.fps, tr.um_per_px)


if __name__ == "__main__":
    main()