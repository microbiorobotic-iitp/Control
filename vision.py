"""vision.py - thin wrapper around FastTracker. Same function names as before, so calibration scripts
that call get_bot_pos() / track_bot() keep working."""
import time

import cv2
import numpy as np

_tracker = None


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def set_tracker(t):
    global _tracker
    _tracker = t
    for name in ('last_kind', 'last_contour'):          # display.py expects these; older FastTracker files lack them
        if not hasattr(t, name):
            setattr(t, name, None)


def get_tracker():
    return _tracker


def track(gray, now=None):
    """Full result dict (center/center_um/vel_um_s/kind/contour/...) or None if the robot is not found."""
    res = _tracker.step(gray, now)
    _tracker.last_kind = res['kind'] if res is not None else None
    _tracker.last_contour = res['contour'] if res is not None else None
    return res


def track_bot(gray, now=None):
    """Old interface: (centres Nx2 px, radii N px). Empty if the robot is absent."""
    res = _tracker.step(gray, now)
    if res is None:
        return np.empty((0, 2)), np.empty(0)
    return res['center'][None, :], np.array([res['R']])


def get_bot_pos(cam, allow_partial=False):
    for _ in range(5):
        res = _tracker.step(to_gray(cam.snapshot()))
        centres, radii = (np.empty((0, 2)), np.empty(0)) if res is None else \
            (res['center'][None, :], np.array([res['R']]))
        cam._detections = (centres.copy(), radii.copy(), time.time())
        if res is not None and (allow_partial or res['kind'] == 'full'):
            return res['center'].copy()
    return np.array([np.nan, np.nan])