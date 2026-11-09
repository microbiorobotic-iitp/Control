"""Frame sources. Both return (bgr_frame, t_seconds, index) from read(), or None at end/timeout."""
import os
import threading
import time

import cv2
import numpy as np


class VideoSource:
    is_live = False

    def __init__(self, path, loop=False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.loop, self.idx = loop, 0

    def read(self):
        ok, f = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.idx = 0
            ok, f = self.cap.read()
        if not ok:
            return None
        self.idx += 1
        return f, (self.idx - 1) / self.fps, self.idx - 1          # video time, not wall time

    def release(self):
        self.cap.release()


class CameraSource:
    """Webcam with a background grab thread (always returns the NEWEST frame, no stale buffer)."""
    is_live = True

    def __init__(self, index=0, width=None, height=None, fps=30, fourcc="MJPG"):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError(f"cannot open camera {index}")
        if fourcc:                      # MJPG is what lets USB2 cameras reach 30 fps at high resolution
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._cond, self._latest, self._n, self._taken, self._run = threading.Condition(), None, 0, 0, True
        self.dropped = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._cond:
                self._latest = (f, time.perf_counter(), self._n)
                self._n += 1
                self._cond.notify_all()

    def read(self, timeout=1.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._n > self._taken, timeout):
                return None
            f, t, n = self._latest
            self.dropped += n - self._taken
            self._taken = n + 1
        return f, t, n

    # ---- camera controls -------------------------------------------------
    def get(self, prop):
        return self.cap.get(prop)

    def set(self, prop, val):
        return self.cap.set(prop, val)

    def mean_brightness(self, n=3):
        vals = []
        for _ in range(n):
            r = self.read()
            if r:
                vals.append(float(cv2.cvtColor(r[0], cv2.COLOR_BGR2GRAY).mean()))
        return float(np.mean(vals)) if vals else float("nan")

    def lock_exposure(self, exposure=None):
        """Switch auto-exposure OFF (keeping the exposure the camera had chosen unless `exposure` is given),
        also auto white-balance and auto-focus. Then VERIFY that the camera really obeys manual exposure.
        Returns (ok, message).  Exposure units: DirectShow = log2(seconds) (-5 ~ 1/32 s, -8 ~ 1/256 s);
        V4L2 = 100 us units."""
        cap = self.cap
        cur = cap.get(cv2.CAP_PROP_EXPOSURE) if exposure is None else exposure
        manual_codes = (0.25, 1.0) if os.name == "nt" else (1.0, 0.25)   # DSHOW quirk: 0.25 = manual
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        ok = False
        for code in manual_codes:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, code)
            cap.set(cv2.CAP_PROP_EXPOSURE, cur)
            time.sleep(0.4)
            if abs(cap.get(cv2.CAP_PROP_EXPOSURE) - cur) < 0.51:
                ok = True
                break
        if not ok:
            return False, "camera/driver ignored the manual-exposure request"
        # empirical check: a change of exposure must change the image brightness
        b0 = self.mean_brightness()
        step = 1.0 if os.name == "nt" else cur * 0.5
        cap.set(cv2.CAP_PROP_EXPOSURE, cur - step)
        time.sleep(0.5)
        b1 = self.mean_brightness()
        cap.set(cv2.CAP_PROP_EXPOSURE, cur)
        time.sleep(0.4)
        if not (abs(b1 - b0) > 0.03 * max(b0, 1.0)):
            return False, ("exposure value is accepted but the image does not respond -> auto-exposure is "
                           "probably still active in the driver/firmware (try the driver dialog, key D)")
        return True, f"manual exposure locked at {cur} (brightness responds to exposure changes)"

    def auto_exposure_on(self):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if os.name == "nt" else 3.0)

    def open_driver_dialog(self):
        """DirectShow only: the camera maker's own property page (often has an 'Auto' exposure checkbox)."""
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def release(self):
        self._run = False
        time.sleep(0.05)
        self.cap.release()
