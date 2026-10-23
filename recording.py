"""
recording.py - background recording of video + duty cycles, usable from manual.py AND navigate.py.

Folder layout
-------------
experiments/
  20261014_153000_myexp/
    meta.json            <- your meta answers, robot geometry, settings
    run_01/
      video.avi          <- raw frames (no overlay)
      log.csv            <- ONE ROW PER VIDEO FRAME (row i  <->  frame i of the video)
      events.csv         <- key presses / changes with timestamps
      run_info.json      <- n_frames, dropped frames, measured fps, ...
    run_02/ ...

Correspondence: log.csv column `frame` is the index of the frame inside video.avi.
`t_s` is seconds since the run started, `t_wall` is the unix time (to sync with other devices).
The duty written in a row is the duty that was ACTIVE while that frame was captured.

Saving happens in a worker thread; the control loop only does a frame.copy() + queue.put.
"""
import csv
import json
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np

COIL_NAMES = ('xplus', 'xminus', 'yplus', 'yminus')
DUTY_COLUMNS = ['duty_xplus', 'duty_xminus', 'duty_yplus', 'duty_yminus', 'duty_x', 'duty_y']


# ----------------------------------------------------------------- small helpers
def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=_jsonable)


def ask_meta(questions):
    """questions = [(key, 'question text', default), ...] -> {key: answer}"""
    print('\n--- Experiment info (just press Enter to keep the default) ---')
    answers = {}
    for key, question, default in questions:
        suffix = f' [{default}]' if default != '' else ''
        a = input(f'{question}{suffix}: ').strip()
        answers[key] = a if a else default
    return answers


def duties_from_vector(dir_vec, magnitude):
    """Same logic as hardware.currents(): (direction, magnitude) -> duty per coil. Use this in navigate.py."""
    m = max(0.0, min(1.0, magnitude))
    x, y = dir_vec[0] * m, dir_vec[1] * m
    return dict(xplus=max(x, 0.0), xminus=max(-x, 0.0), yplus=max(y, 0.0), yminus=max(-y, 0.0))


def duty_columns(duties):
    """{'xplus':..,'xminus':..,'yplus':..,'yminus':..} -> dict with the DUTY_COLUMNS keys (incl. signed net duty)."""
    d = {n: float(duties.get(n, 0.0)) for n in COIL_NAMES}
    return {
        'duty_xplus': d['xplus'], 'duty_xminus': d['xminus'],
        'duty_yplus': d['yplus'], 'duty_yminus': d['yminus'],
        'duty_x': d['xplus'] - d['xminus'], 'duty_y': d['yplus'] - d['yminus'],
    }


# ----------------------------------------------------------------- robot geometry
def robot_geometry(tracker, mm_per_px=None):
    """Geometrical properties of the robot from a locked RobotTracker (call after setup_tracker)."""
    g = dict(
        area_px=float(tracker.A_ref),
        radius_px=float(tracker.R),                       # equivalent radius sqrt(A/pi)
        diameter_px=float(2 * tracker.R),
        perimeter_ideal_px=float(2 * np.pi * tracker.R),  # perimeter of a perfect circle with same area
        circularity=float(tracker.circ_ref),
        solidity=float(tracker.sol_ref),
        robot_darkness=float(tracker.robot_int),
        background_level=float(tracker.bg),
        centre_px=tracker.pos.tolist(),
    )
    c = tracker.last_contour
    if c is not None:
        g['perimeter_px'] = float(cv2.arcLength(c, True))
        g['hull_area_px'] = float(cv2.contourArea(cv2.convexHull(c)))
        x, y, w, h = cv2.boundingRect(c)
        g['bbox_w_px'], g['bbox_h_px'] = int(w), int(h)
        _, r = cv2.minEnclosingCircle(c)
        g['min_enclosing_radius_px'] = float(r)
        if len(c) >= 5:
            _, (a1, a2), ang = cv2.fitEllipse(c)
            g['ellipse_major_px'], g['ellipse_minor_px'] = float(max(a1, a2)), float(min(a1, a2))
            g['ellipse_aspect'] = float(max(a1, a2) / max(1e-9, min(a1, a2)))
            g['ellipse_angle_deg'] = float(ang)
    if mm_per_px:
        k = float(mm_per_px)
        g['mm_per_px'] = k
        g['area_mm2'] = g['area_px'] * k * k
        g['radius_mm'] = g['radius_px'] * k
        g['diameter_mm'] = g['diameter_px'] * k
        if 'perimeter_px' in g:
            g['perimeter_mm'] = g['perimeter_px'] * k
    return g


# ----------------------------------------------------------------- experiment folder
class Experiment:
    def __init__(self, name, answers=None, geometry=None, settings=None, root='experiments'):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir = os.path.join(root, f'{stamp}_{name}')
        os.makedirs(self.dir, exist_ok=True)
        self.n_runs = 0
        self.meta = dict(experiment_name=name,
                         started=datetime.now().isoformat(timespec='seconds'),
                         answers=answers or {}, robot_geometry=geometry, settings=settings or {})
        self.save_meta()

    def update_meta(self, **kw):
        self.meta.update(kw)
        self.save_meta()

    def save_meta(self):
        save_json(os.path.join(self.dir, 'meta.json'), self.meta)

    def new_run_dir(self):
        self.n_runs += 1
        d = os.path.join(self.dir, f'run_{self.n_runs:02d}')
        os.makedirs(d, exist_ok=True)
        return d


# ----------------------------------------------------------------- recorder
class Recorder:
    def __init__(self, folder, columns, fps=30, codec='MJPG', ext='avi', max_queue=120, info=None):
        """
        columns : your per-frame columns, e.g. DUTY_COLUMNS + ['x_px', 'y_px'].
                  'frame', 't_s', 't_wall' are added automatically in front.
        max_queue: frames buffered if the disk is slow (each ~1-3 MB). When full, frame AND row
                  are dropped together, so frame<->row correspondence is never broken.
        """
        self.folder = folder
        self.columns = ['frame', 't_s', 't_wall'] + list(columns)
        self.fps, self.codec = fps, codec
        self.video_path = os.path.join(folder, f'video.{ext}')
        self.info = dict(info or {})
        self.dropped = 0
        self.written = 0
        self.error = None
        self._t0 = time.perf_counter()
        self._first_t = self._last_t = None
        self._q = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ---- called from the control loop (cheap) ----
    @property
    def backlog(self):
        return self._q.qsize()

    def elapsed(self):
        return time.perf_counter() - self._t0

    def log(self, frame, t=None, **fields):
        """t = time.perf_counter() taken right after cam.snapshot() (default: now)."""
        now = time.perf_counter()
        t = now if t is None else t
        row = dict(fields)
        row['t_s'] = t - self._t0
        row['t_wall'] = time.time() - (now - t)
        try:
            self._q.put_nowait(('frame', frame.copy(), row))
        except queue.Full:
            self.dropped += 1
            return
        if self._first_t is None:
            self._first_t = row['t_s']
        self._last_t = row['t_s']

    def event(self, text):
        try:
            self._q.put_nowait(('event', self.elapsed(), text))
        except queue.Full:
            pass

    def close(self):
        """Flush everything to disk and return a summary dict."""
        self._q.put(None)
        self._thread.join()
        dur = (self._last_t - self._first_t) if self._first_t is not None else 0.0
        summary = dict(self.info, n_frames=self.written, dropped_frames=self.dropped,
                       duration_s=dur, nominal_fps=self.fps,
                       measured_fps=(self.written - 1) / dur if dur > 0 and self.written > 1 else None,
                       video_file=os.path.basename(self.video_path), codec=self.codec, error=self.error)
        save_json(os.path.join(self.folder, 'run_info.json'), summary)
        return summary

    # ---- worker thread ----
    def _worker(self):
        vw = None
        video_ok = True
        f_log = open(os.path.join(self.folder, 'log.csv'), 'w', newline='')
        f_ev = open(os.path.join(self.folder, 'events.csv'), 'w', newline='')
        wl = csv.DictWriter(f_log, fieldnames=self.columns, restval='', extrasaction='ignore')
        we = csv.writer(f_ev)
        wl.writeheader()
        we.writerow(['t_s', 'event'])
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                if item[0] == 'event':
                    we.writerow([f'{item[1]:.4f}', item[2]])
                    continue
                _, frame, row = item
                if vw is None and video_ok:
                    h, w = frame.shape[:2]
                    vw = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*self.codec),
                                         self.fps, (w, h))
                    if not vw.isOpened():
                        video_ok, vw = False, None
                        self.error = f'could not open VideoWriter with codec {self.codec}'
                        print('[recorder] WARNING:', self.error)
                if vw is not None:
                    vw.write(frame)
                row['frame'] = self.written
                wl.writerow(row)
                self.written += 1
                if self.written % 50 == 0:
                    f_log.flush()
                    f_ev.flush()
        except Exception as e:                      # never kill the experiment because of disk problems
            self.error = repr(e)
            print('[recorder] ERROR:', e)
        finally:
            if vw is not None:
                vw.release()
            f_log.close()
            f_ev.close()