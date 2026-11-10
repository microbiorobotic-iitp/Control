"""Interactive tracker set-up for the existing Camera class (replaces tracker.setup_tracker).

Live view with keys:
  R  pick robot (drag a box, ENTER)      K  mask editor      B  refine background (keep robot moving ~2 s)
  [ ]  change magnification (clears mask, robot must be re-picked)
  E  lock camera exposure     D  camera driver dialog
  ENTER  accept (needs the robot to be found)      Q  quit (returns None)
"""
import time

import cv2

import ui
from fast_tracker import FastTracker
from vision import to_gray


def _put(img, text, y, col=(0, 255, 255)):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (3, y - h - 4), (13 + w, y + b), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def _outlines(mask):
    if mask is None or not mask.any():
        return []
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cs


def robot_geometry(tracker):
    """Robot geometry for meta.json (replaces recording.robot_geometry, which needs the old tracker)."""
    g = dict(area_px=float(tracker.A_ref), radius_px=float(tracker.R), diameter_px=float(2 * tracker.R),
             um_per_px=float(tracker.um_per_px), diameter_um=float(2 * tracker.R * tracker.um_per_px),
             area_um2=float(tracker.A_ref * tracker.um_per_px ** 2),
             robot_darkness=float(tracker.robot_int), background_level=float(tracker.bg_robot),
             threshold_frac=float(tracker.cfg.thr_frac), centre_px=tracker.pos.tolist())
    c = tracker.contour0
    if c is not None:
        g['perimeter_px'] = float(cv2.arcLength(c, True))
        g['hull_area_px'] = float(cv2.contourArea(cv2.convexHull(c)))
        if len(c) >= 5:
            _, (a1, a2), ang = cv2.fitEllipse(c)
            g['ellipse_major_px'], g['ellipse_minor_px'] = float(max(a1, a2)), float(min(a1, a2))
            g['ellipse_aspect'] = float(max(a1, a2) / max(1e-9, min(a1, a2)))
    return g


def setup_tracker(cam, win, scale, cfg, tracker=None, tick=None):
    """Returns a valid FastTracker (the same object if one was passed in), or None if the user quit.
    `tick`: optional function called every frame (navigate uses it to keep re-sending the held duty).
    `scale` (tracker_config.Scale) is changed in place when [ ] is pressed: call profiles.apply_scale afterwards."""
    tr = tracker if tracker is not None else FastTracker(scale.um_per_px, cfg)
    tr.set_scale(scale.um_per_px)
    msg, col = ('Press R and drag a box around the robot' if not tr.valid else ''), (0, 255, 255)
    outl_u, outl_a, dirty = [], [], True
    cv2.setMouseCallback(win, lambda *a: None)
    while True:
        frame = cam.snapshot()
        if tick is not None:
            tick()
        now = time.perf_counter()
        gray = to_gray(frame)
        res = tr.step(gray, now) if tr.valid else None
        if tr.bg_msg is not None:
            msg, col = tr.bg_msg[1], ((0, 255, 0) if tr.bg_msg[0] else (0, 0, 255))
            print('[bg]', msg)
            tr.bg_msg, dirty = None, True
        if dirty:
            outl_u, outl_a, dirty = _outlines(tr.excl_user), _outlines(tr.excl_auto), False
        view = frame.copy()
        for cs, c in ((outl_u, (0, 0, 255)), (outl_a, (0, 128, 255))):
            if cs:
                cv2.polylines(view, cs, True, c, 1)
        if res is not None:
            cv2.polylines(view, [res['contour']], True, (0, 255, 0), 1)
            cv2.circle(view, tuple(int(v) for v in res['center']), 2, (0, 0, 255), -1)
        _put(view, f"SETUP  {scale.key}x = {scale.um_per_px:.3f} um/px{'' if scale.verified else ' (UNVERIFIED)'}  "
                   f"{'robot OK' if res is not None else 'robot NOT found'}", 20)
        _put(view, "R pick robot | K mask | B bg-refine | [ ] magnification | E lock exposure | ENTER accept | Q quit", 40)
        if tr.collecting:
            _put(view, f"collecting background {tr.bg_progress[0]}/{tr.bg_progress[1]} - keep the robot MOVING", 60, (0, 165, 255))
        elif msg:
            _put(view, msg, 60, col)
        cv2.imshow(win, view)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('q'), 27):
            return None
        elif k in (13, 10):
            if tr.valid and res is not None:
                tr.reset_clock(time.perf_counter())
                return tr
            msg, col = 'cannot accept: robot not found - press R', (0, 0, 255)
        elif k == ord('r'):
            rect = cv2.selectROI(win, frame, showCrosshair=False)
            if rect[2] > 0 and rect[3] > 0:
                ok, m = tr.learn(gray, rect, time.perf_counter())
                msg, col = m, ((0, 255, 0) if ok else (0, 0, 255))
                print('[learn]', m)
                dirty = True
        elif k == ord('k'):
            m_new = ui.edit_mask(win, frame, tr.exclusion_total())
            tr.excl_auto = None
            tr.set_user_mask(m_new)
            dirty = True
        elif k == ord('b') and tr.valid:
            tr.start_bg_collect()
        elif k in (ord('['), ord(']')):
            scale.step(-1 if k == ord('[') else 1)
            old = tr.um_per_px
            tr.set_scale(scale.um_per_px)
            hint = f' robot should now be ~{2 * tr.R * old / scale.um_per_px:.0f}px across.' if hasattr(tr, 'R') else ''
            tr.invalidate('magnification changed')
            tr.set_user_mask(None)
            tr.excl_auto = None
            dirty = True
            msg, col = f'magnification -> {scale.key}x ({scale.um_per_px:.3f} um/px).{hint} Redo mask [K] and pick robot [R]', (0, 165, 255)
        elif k == ord('e') and hasattr(cam, 'lock_exposure'):
            ok, m = cam.lock_exposure()
            msg, col = m, ((0, 255, 0) if ok else (0, 0, 255))
            print('[camera]', m)
        elif k == ord('d') and hasattr(cam, 'open_driver_dialog'):
            cam.open_driver_dialog()