"""sim_hardware.py - drop-in replacement for hardware.py to test navigate.py WITHOUT the lab setup.

    import sys, sim_hardware
    sys.modules['hardware'] = sim_hardware          # must happen before importing navigate
    import navigate; navigate.main()

A simulated robot (dark disk) moves according to the PWM / direction that navigate.py sends through currents():
    steady speed = G_true * (pwm - phi_true)  (first-order lag tau), plus a constant sideways drift,
    and an optional 'cell' load that reduces the gain as the robot is pulled away from PLANT.attach_pos.
Frames are rendered in real time at 30 fps, so dt / velocity estimates behave like the real camera.
Edit PLANT.* to mimic a fluid (a wrong G or phi is a good way to see what the PID has to correct).
"""
import time

import cv2
import numpy as np


class Plant:
    def __init__(self):
        self.upp = 4.66                 # um per px
        self.G_true = 4000.0            # um/s per unit PWM
        self.phi_true = 0.02            # PWM at which motion starts
        self.tau = 0.15                 # s, first-order lag
        self.drift_um_s = np.array([0.0, 4.0])      # constant sideways disturbance (image axes)
        self.noise_px = 0.0
        self.load_len_um = 0.0          # >0: gain falls as 1/(1 + s/load_len) after attach_pos is set
        self.attach_pos = None
        self.pos = np.array([200.0, 300.0])         # px, image coordinates
        self.vel = np.zeros(2)          # px/s
        self.cmd_dir = np.zeros(2)      # coil axes (y up)
        self.cmd_pwm = 0.0
        self.t_last = None

    def advance(self, now):
        if self.t_last is None:
            self.t_last = now
            return
        dt = min(now - self.t_last, 0.2)
        self.t_last = now
        g = self.G_true
        if self.load_len_um > 0 and self.attach_pos is not None:
            s = np.linalg.norm(self.pos - self.attach_pos) * self.upp
            g = g / (1.0 + s / self.load_len_um)
        speed = g * max(self.cmd_pwm - self.phi_true, 0.0) / self.upp                  # px/s
        d_img = np.array([self.cmd_dir[0], -self.cmd_dir[1]])
        v_target = speed * d_img + self.drift_um_s / self.upp
        self.vel += (v_target - self.vel) * (1 - np.exp(-dt / self.tau))
        self.pos += self.vel * dt
        self.pos = np.clip(self.pos, [20, 20], [620, 460])


PLANT = Plant()


class SimCamera:
    W, H, FPS = 640, 480, 30

    def __init__(self, *a, **k):
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        bg = 190 - 35 * (((xx - self.W / 2) / self.W) ** 2 + ((yy - self.H / 2) / self.H) ** 2) * 2
        bg += cv2.GaussianBlur(np.random.default_rng(3).normal(0, 1, (self.H, self.W)).astype(np.float32), (0, 0), 20) * 90
        bg[:, :30] = 55                                           # dark wall bar on the left
        self.bg = bg
        self.rng = np.random.default_rng(1)
        self._next = None
        self._n = 0

    def _render(self):
        img = np.zeros((self.H, self.W), np.float32)
        p = PLANT.pos + self.rng.normal(0, PLANT.noise_px, 2)
        r_px = 40.0 / PLANT.upp                                   # a 80 um robot
        cv2.circle(img, (int(round(p[0] * 8)), int(round(p[1] * 8))), int(round(r_px * 8)), 1.0, -1, cv2.LINE_AA, shift=3)
        img = cv2.GaussianBlur(img, (0, 0), 1.0)
        f = self.bg * (1 - 0.7 * np.clip(img, 0, 1)) + self.rng.normal(0, 2.0, img.shape)
        return cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def snapshot_t(self, timeout=5):
        now = time.perf_counter()
        if self._next is None:
            self._next = now
        self._next += 1.0 / self.FPS
        if self._next > now:
            time.sleep(self._next - now)
        else:
            self._next = now
        PLANT.advance(time.perf_counter())
        self._n += 1
        return self._render(), time.perf_counter()

    def snapshot(self, timeout=5):
        return self.snapshot_t(timeout)[0]

    def warmup(self, n=3):
        for _ in range(n):
            self.snapshot()

    def release(self):
        pass


Camera = SimCamera


class Arduino:
    def __init__(self, *a, **k):
        pass


def currents(ard, direction, magnitude):
    PLANT.cmd_dir = np.array(direction, float)
    PLANT.cmd_pwm = float(magnitude)
