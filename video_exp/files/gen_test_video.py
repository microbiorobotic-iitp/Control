"""Synthetic test video + ground truth:  python gen_test_video.py out.avi [--shift-at N] [--w 1280 --h 720]
Static textured background with vignette, wall bar and soft shadow, sensor noise, one dark motion-blurred
robot that moves fast and briefly leaves through the top edge."""
import argparse

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--w", type=int, default=1280)
ap.add_argument("--h", type=int, default=720)
ap.add_argument("--frames", type=int, default=450)
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--contrast", type=float, default=0.72, help="robot darkening 0..1 (0.72 = 72 percent darker than bg)")
ap.add_argument("--shift-at", type=int, default=-1)
ap.add_argument("--drift", type=float, default=0.0, help="total brightness drift in grey levels over the clip")
a = ap.parse_args()
W, H, N = a.w, a.h, a.frames
rng = np.random.default_rng(1)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
bg = 195 - 40 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2
tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 25) * 120
bg += tex
bg[:, : int(W * 0.04)] = 55                                   # wall bar
cv2.line(bg, (int(W * .04), 0), (int(W * .04), H), 35, 3)
sh = np.zeros((H, W), np.float32)
cv2.ellipse(sh, (int(W * .86), int(H * .85)), (int(W * .08), int(H * .06)), 20, 0, 360, 1.0, -1)
sh = cv2.GaussianBlur(sh, (0, 0), 6)
bg = bg * (1 - 0.55 * sh)                                      # soft static shadow
R = max(6, int(H * 0.0125))
base_R = R


def robot_pos(i):
    t = i / a.fps
    x = W * (0.5 + 0.38 * np.sin(2 * np.pi * t / 6.0))
    y = H * (0.5 + 0.40 * np.sin(2 * np.pi * t / 4.1 + 1.0))
    if 9.0 < t < 9.8:                                          # slide out through the top edge
        y = y - (H * 0.8) * np.sin(np.pi * (t - 9.0) / 0.8)
    return x, y


vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"MJPG"), a.fps, (W, H))
gt = []
sub = 6
for i in range(N):
    img = np.zeros((H, W), np.float32)
    for s in range(sub):                                       # exposure = 1/3 of frame time
        x, y = robot_pos(i + (s / sub - 0.5) * 0.33)
        tmp = np.zeros((H, W), np.float32)
        cv2.circle(tmp, (int(round(x * 8)), int(round(y * 8))), R * 8, 1.0, -1, cv2.LINE_AA, shift=3)
        img += tmp / sub
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    frame = bg * (1 - a.contrast * np.clip(img, 0, 1))
    if a.shift_at >= 0 and i >= a.shift_at:
        frame = cv2.warpAffine(frame, np.float32([[1, 0, 7], [0, 1, 4]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    frame = frame + a.drift * i / N + rng.normal(0, 2.0, (H, W))
    f8 = np.clip(frame, 0, 255).astype(np.uint8)
    vw.write(cv2.cvtColor(f8, cv2.COLOR_GRAY2BGR))
    x, y = robot_pos(i)
    gt.append((i, x, y))
vw.release()
np.savetxt(a.out.rsplit(".", 1)[0] + "_gt.csv", np.array(gt), delimiter=",", header="frame,x_px,y_px", comments="")
print(f"wrote {a.out}: {W}x{H}, {N} frames, robot radius {base_R}px; start box ~ x={int(robot_pos(0)[0])-25} y={int(robot_pos(0)[1])-25} w=50 h=50")
