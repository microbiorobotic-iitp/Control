import cv2
import numpy as np
import time

import config as C


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _is_dark(gray, x, y, r):
    inner = np.zeros_like(gray)
    cv2.circle(inner, (int(x), int(y)), int(r), 255, -1)
    outer = np.zeros_like(gray)
    cv2.circle(outer, (int(x), int(y)), int(r * 1.5), 255, -1)
    outer = cv2.subtract(outer, inner)
    return gray[inner > 0].mean() < gray[outer > 0].mean()


def find_bots(gray):
    """Equivalent of imfindcircles(..., 'ObjectPolarity','Dark'). Returns (centres Nx2, radii N), strongest first."""
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=C.RMIN,
                               param1=C.HOUGH_PARAM1, param2=C.HOUGH_PARAM2,
                               minRadius=C.RMIN, maxRadius=C.RMAX)
    if circles is None:
        return np.empty((0, 2)), np.empty(0)
    kept = np.array([c for c in circles[0] if _is_dark(gray, *c)]).reshape(-1, 3)
    return kept[:, :2], kept[:, 2]

def get_bot_pos(cam, allow_partial=False):
    for _ in range(5):
        frame_gray = to_gray(cam.snapshot())
        res = _tracker.step(frame_gray)
        if res is None:
            centres, radii = np.empty((0, 2)), np.empty(0)
        else:
            centres, radii = res['center'][None, :], np.array([res['R']])
        cam._detections = (centres.copy(), radii.copy(), time.time())
        if res is not None and (allow_partial or res['kind'] == 'full'):
            return res['center'].copy()
    return np.array([np.nan, np.nan])

_tracker = None

def set_tracker(t):
    global _tracker
    _tracker = t

def get_tracker():
    return _tracker

def track_bot(gray):
    """Same return shape as find_bots(): (centres Nx2, radii N). Empty if robot absent."""
    res = _tracker.step(gray)
    if res is None:
        return np.empty((0, 2)), np.empty(0)
    return res['center'][None, :], np.array([res['R']])