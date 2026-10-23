import threading
import cv2
import numpy as np
import pyfirmata2
import time

import config as C


class Arduino:
    STEPS = 2048                                  # 7812.5 Hz
    CHANNEL = {9: 0, 10: 1, 11: 2, 3: 3}

    def __init__(self, port, pins):
        self.board = pyfirmata2.Arduino(port)
        self.dir = {}
        self.pwm = {}
        for name in ('xplus', 'xminus', 'yplus', 'yminus'):
            self.dir[name] = self.board.get_pin(f"d:{int(pins[name + '_dir'][1:])}:o")
            self.pwm[name] = self.board.get_pin(f"d:{int(pins[name][1:])}:p")

    def write_dir(self, name, value):
        self.dir[name].write(int(value))

    def write_pwm(self, name, duty):
        self.pwm[name].write(float(duty))

    def close(self):
        self.board.exit()


def currents(ard, dir_vec, magnitude):
    magnitude = max(0.0, min(1.0, magnitude))
    x_val = dir_vec[0] * magnitude
    y_val = dir_vec[1] * magnitude

    if x_val >= 0:
        ard.write_dir('xplus', 0)
        ard.write_pwm('xplus', x_val)
        ard.write_dir('xminus', 0)
        ard.write_pwm('xminus', 0.0)
    else:
        ard.write_dir('xplus', 0)
        ard.write_pwm('xplus', 0.0)
        ard.write_dir('xminus', 0)
        ard.write_pwm('xminus', abs(x_val))

    if y_val >= 0:
        ard.write_dir('yplus', 0)
        ard.write_pwm('yplus', y_val)
        ard.write_dir('yminus', 0)
        ard.write_pwm('yminus', 0.0)
    else:
        ard.write_dir('yplus', 0)
        ard.write_pwm('yplus', 0.0)
        ard.write_dir('yminus', 1)
        ard.write_pwm('yminus', abs(y_val))


class Camera:
    """one thread keep reading the frames in the background, whenever take snapshot is called another thred blocks it and reads the saved frame, hence returning the closest frame possible at the time of calling."""

    def __init__(self, index=C.CAMERA_INDEX, size=C.FRAME_SIZE):

        # cv2.CAP_DSHOW -> faster startup (Direct show)
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._frame = None
        self._count = 0
        self._seen = 0
        self._running = True

         # for seeing live video without disturbing the live performance of the control logic
        self._overlay = []
        self._show = False

        # to stop moving the bot out of the wall
        self._key_event = threading.Event()
        self._detections = None

        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # it keeps capturing frames in the background so that frame can be provided as soon as it is asked for. 
    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()  # this have to wait for the camera hardware to provide the snapshot
            if ok:
                with self._cond:  # as now this thread is dealing with the variables it blocks those variables
                    self._frame = frame
                    self._count += 1
                    self._cond.notify_all()  # notifies all the waiting threads (if there is any) that the frame has now been updated

    # this function returns the image which needs to be analyzed
    def snapshot(self, timeout=5):
        with self._cond:  # locks the variable so that the loop thread doesn't change it now

            # if there is new frame then go to next line
            # if there is no new frame, then sleep until the _loop notifies that there is a new frame
            # if wait is longer than 5s then there is an error
            if not self._cond.wait_for(lambda: self._count > self._seen, timeout):
                raise RuntimeError('Camera timed out')
            self._seen = self._count
            return self._frame.copy()

    def warmup(self, n=C.WARMUP_FRAMES):
        for _ in range(n):
            self.snapshot()

    # this code just copy the existing newest frame
    def latest(self):
        with self._cond:
            return None if self._frame is None else self._frame.copy()

    # start the preview window in its own thread
    def start_preview(self, window='Calibration', fps=30):
        self._show = True
        self._preview_thread = threading.Thread(
            target=self._preview_loop, args=(window, fps), daemon=True)
        self._preview_thread.start()

    def _preview_loop(self, window, fps):
        delay = max(1, int(1000 / fps))                 # waitKey(0) would block forever
        while self._show and self._running:
            frame = self.latest()
            if frame is not None:
                det = self._detections
                if det is not None and time.time() - det[2] < 0.5:
                    centres, radii, _ = det
                    for i, (c, r) in enumerate(zip(centres, radii)):
                        colour = (0, 255, 0) if i == 0 else (0, 0, 255)
                        cv2.circle(frame, (int(c[0]), int(c[1])), int(r), colour, 2)
                        cv2.circle(frame, (int(c[0]), int(c[1])), 2, colour, -1)
                for i, text in enumerate(list(self._overlay)):
                    cv2.putText(frame, text, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                cv2.imshow(window, frame)

            key = cv2.waitKey(delay) & 0xFF             # NEW: refreshes window + paces loop
            if key == 32:                               # SPACE = wall hit
                self._key_event.set()
            elif key == 27:                             # Esc closes preview only
                self._show = False
        cv2.destroyAllWindows()

    def stop_preview(self):
        self._show = False

    def release(self):
        self._show = False
        self._running = False
        self._thread.join(timeout=1)
        if hasattr(self, '_preview_thread'):
            self._preview_thread.join(timeout=1)   # NEW
        self.cap.release()