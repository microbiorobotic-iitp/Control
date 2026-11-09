# Micro-robot tracker

Tracks one dark, roughly circular micro-robot in a microscope/webcam view and reports its
centre, boundary points, velocity and radius in **micrometres**. Built for speed: after a one-time
learning step, each frame costs about 0.4 ms (1080p, tracker only).

Contents

1. [Quick start](#1-quick-start)
2. [Controls](#2-controls)
3. [Command-line options](#3-command-line-options)
4. [Parameters (`config.py`)](#4-parameters-configpy)
5. [Values hard-coded in the code](#5-values-hard-coded-in-the-code)
6. [`magnification.json`](#6-magnificationjson)
7. [How the algorithm works](#7-how-the-algorithm-works)
8. [File structure](#8-file-structure)
9. [Output (CSV and the `step()` result)](#9-output)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Quick start

```
pip install opencv-python numpy

# interactive, on a video file (plays at the video's own speed)
python run_tracker.py --video my_video.avi --mag 2

# live camera
python run_tracker.py --camera 0 --mag 2

# pure benchmark, no window, as fast as possible (needs the robot box: X Y W H in pixels)
python run_tracker.py --video my_video.avi --fast --no-gui --init 615 577 50 50
```

Typical interactive session:

1. The first frame (video) or live view (camera) appears in state **SETUP**.
2. Press **K**, paint everything the robot can never be in (walls, dark corners), press **ENTER**.
3. Press **R**, drag a box tightly around the robot, press **ENTER**.
4. A green outline appears (state **CONFIRM**). If it is the robot, press **ENTER**; otherwise **R** again.
5. Tracking starts. Optionally press **B** and keep the robot moving ~2 s to refine the background
   (fixes the robot crossing static shadows).

---

## 2. Controls

Click the video window first so it has keyboard focus.

### Main window

| Key | Action |
|---|---|
| **R** | Pick the robot: drag a box around it, press ENTER (press C to cancel). Learns size, darkness and builds the threshold map. Use again whenever the workspace moved or lighting changed. |
| **K** | Open the mask editor (see below). |
| **S** | Save the current mask to `mask.png` (load it next time with `--mask mask.png`). |
| **C** | Camera panel (live camera only): exposure, gain, brightness, contrast, lock auto-exposure. |
| **B** | Refine the background from the robot's own motion. Keep the robot moving for ~2 s (45 frames where it is found). Tracking continues during collection; one ~0.3 s pause happens at the end. |
| **[**  /  **]** | Previous / next magnification from `magnification.json`. This invalidates the tracker and clears the mask (the field of view changed): redo **K** and **R**. |
| **SPACE** | Tracking ON/OFF. Turning it back ON forces a full-frame search. |
| **P** | Freeze / unfreeze the video. |
| **V** | Show / hide mask outlines (red = your mask, orange = automatic). |
| **T** | Show / hide the trail. |
| **H** | Print the timing table to the console. |
| **ENTER** | Accept the learned robot (in CONFIRM state). |
| **Q** or **ESC** | Quit (prints the timing summary and writes the CSV). |

States shown in the top-left: **SETUP** (no valid tracker yet; video is held on one frame, live camera keeps running),
**CONFIRM** (check the green outline), **TRACK**.

### Mask editor (after pressing K)

The editor shows everything that is currently masked in red, **including the automatic zones**, so
everything you see can be erased. When you accept, the edited result becomes your mask and the automatic zones are cleared.

| Input | Action |
|---|---|
| Left-drag | Draw (exclude) |
| Right-drag | Opposite of the left button (erase) |
| **E** | Swap the left button between draw and erase. **Use this if your right mouse button or touchpad does not register.** The window text and the brush colour show the mode (yellow ring = draw, green ring = erase). |
| **[**  /  **]** | Smaller / larger brush |
| **P** | Polygon mode. Left-click vertices, then **K** = keep INSIDE only (everything outside is excluded), or **X** = exclude inside. **ESC** leaves polygon mode. |
| **Z** | Undo (up to 25 steps) |
| **C** | Clear the whole mask |
| **ENTER** | Accept |
| **ESC** | Cancel (restores the mask you started with) |

Note: pressing **R** afterwards re-runs automatic exclusion, which can mask dark blobs again.
Start with `--no-auto-exclude` (or `auto_exclude = False`) if you want only your own mask.

### Camera panel (after pressing C)

| Input | Action |
|---|---|
| Sliders | `exposure(-log2 s)`: slider value v sets exposure = -v (DirectShow units, e.g. 6 = 1/64 s). `gain`, `brightness`, `contrast`: raw driver values 0-255. Ranges depend on the camera; some cameras ignore some controls. |
| **E** | Lock exposure: switches auto-exposure OFF at the value the camera has now, turns off auto white-balance and auto-focus, then **verifies** that brightness really responds to exposure. Prints the result. |
| **A** | Auto-exposure back ON |
| **D** | Open the camera's own driver dialog (Windows/DirectShow only) |
| **ENTER** / **ESC** | Close the panel |

Saturated pixels (>= 250) are painted red and a histogram is drawn. Aim for no red on the robot's surroundings.

---

## 3. Command-line options

| Option | Meaning |
|---|---|
| `--video PATH` or `--camera N` | Input (exactly one is required). |
| `--mag KEY` | Magnification key from `magnification.json` (e.g. `2`, `0.65`). Default: the `"default"` entry. |
| `--init X Y W H` | Robot box in pixels; skips the interactive robot selection. Required with `--no-gui`. |
| `--mask FILE.png` | Load a mask (white = excluded). |
| `--csv FILE` | Output CSV (default `track_out.csv`). |
| `--no-gui` | No windows (benchmarking, batch runs). |
| `--fast` | Video: do not wait for real time. Always use this for benchmarking. |
| `--loop` | Restart the video at the end. |
| `--preview-scale F` | Display scale. `0` = auto (shrinks to 1280 px wide). Tracking always runs at full resolution. |
| `--max-frames N` | Stop after N tracked frames. |
| `--max-speed V` | Robot's maximum speed in um/s (overrides `max_speed_um_s`). |
| `--no-auto-exclude` | Do not automatically mask other dark blobs at setup. |
| `--refine-bg` | Start the background refinement right after `--init` (for headless runs). |
| `--gt FILE` | Ground-truth CSV (from `gen_test_video.py`): prints the position/velocity error at the end. |
| `--width N`, `--height N` | Requested camera resolution (live camera). |

---

## 4. Parameters (`config.py`)

Edit the defaults in the `Cfg` class in `config.py`. For a quick experiment you can also set them in
`run_tracker.py` right after `cfg = Cfg()`, e.g. `cfg.thr_frac = 0.6`.
Most changes take effect on the next run. Parameters marked **(re-learn)** are used while learning,
so press **R** again after changing them in a running session (and **B** if you used it).

Lengths are in micrometres, speeds in um/s.

### Motion

| Parameter | Default | What it does | Raise it | Lower it |
|---|---|---|---|---|
| `max_speed_um_s` (also `--max-speed`) | 5000 | Fastest speed the robot can have. Sets how far from its last position the search looks: window half-size = `win_R_factor * R + max(max_speed, 2 * current_speed) * time_since_last_seen`. | Robot that is sometimes lost when fast: the window is too small. Slightly slower (bigger crop), more risk of picking up a stray blob. | Faster, and fewer wrong blobs, but the robot is lost if it exceeds this. **If the robot is lost during fast moves, check this first.** |
| `win_R_factor` | 3.0 | Extra window margin in robot radii. | Helps with erratic motion and big acceleration. | Smaller crop, a little faster. |
| `full_search_after` | 5 | Number of consecutive lost frames before the whole frame is searched. | Stays local longer (better if the robot is briefly hidden). | Recovers sooner from a real loss, but a full-frame search is slower and can find stray blobs. |

### Segmentation (what counts as "robot-dark")

The threshold at each pixel is `T = background * (1 - thr_frac * (1 - robot_dark / robot_background))`, then limited to
`T <= background - min_margin`. A pixel belongs to the robot if its blurred value is below `T`.

| Parameter | Default | What it does | Raise it | Lower it |
|---|---|---|---|---|
| `thr_frac` | 0.5 | Where the threshold sits between the local background (0) and the robot's darkness (1). 0.5 is the half-way (half-maximum) edge, which gives an unbiased size. | Stricter: smaller blobs, fewer false detections, but a fainter/blurry robot may break up or vanish. | More permissive: bigger blobs (size looks inflated), finds low-contrast robots, but picks up noise and halos. Try 0.3-0.4 for a low-contrast robot. |
| `min_margin` | 8 | Minimum number of grey levels the threshold must be below the background. Also: setup refuses if the robot is less than `1.5 * min_margin` grey levels darker than its surroundings. | Rejects noise blobs (raise if your camera is noisy). | Allows lower-contrast robots; set it no lower than your noise level (about 3-4x the noise standard deviation). |
| `blur_k` | 3 | Gaussian blur kernel on the search window (odd, >= 3; 0, 1 or 2 turns blur off). | 5 for a noisy camera: smoother, slightly slower, softer edges. | Sharper edges, noisier. If you change it, redo **B** (the background collection uses the same kernel on purpose). |
| `bg_sigma_diams` **(re-learn)** | 4.0 | Smoothness of the estimated background, in robot diameters. | Smoother background: ignores big structure but cannot follow strong vignetting/gradients. | Follows local brightness variations more closely, but large dark features can leak into the background estimate. |
| `bg_passes` **(re-learn)** | 3 | Number of refinement passes of background estimation + automatic exclusion at setup (minimum 1). | More passes: cleaner background near shadows. | 1 = single pass, no automatic exclusion is refined. |

### Automatic masking

| Parameter | Default | What it does | Raise it / turn on | Lower it / turn off |
|---|---|---|---|---|
| `auto_exclude` (also `--no-auto-exclude`) **(re-learn)** | True | At setup, every dark blob other than the robot (area >= 10% of the robot's) is assumed static and excluded. | - | **Turn off if the robot ever crosses those zones**, or if you want to define the mask entirely yourself. |
| `auto_exclude_grow_px` **(re-learn)** | 4 | How many pixels the automatic zones are expanded (covers soft shadow edges). | Larger safety margin around shadows and walls. | Smaller if the automatic zones clip the robot near walls. |

### Candidate acceptance

There is no shape comparison. Only the blob area is checked, relative to the area learned at setup.

| Parameter | Default | What it does | Raise it | Lower it |
|---|---|---|---|---|
| `area_min_frac` | 0.30 | Smallest accepted blob area (fraction of robot area). | Rejects more small noise blobs; a partly-faded robot may be missed. | Accepts smaller pieces (risk: noise). |
| `area_max_frac` | 4.0 | Largest accepted blob area. Above 1 on purpose, so motion-blur streaks are accepted. | Fast robots with long blur streaks are still accepted. | Rejects merged blobs (for example robot + shadow) more strictly; a motion-blurred robot may be rejected. |
| `partial_min_frac` | 0.08 | Smallest visible area accepted when the robot is cut by the image border. | Needs more of the robot visible. | Tracks the robot while it is almost out of view. |

### Velocity filter (alpha-beta)

Each frame: `predicted = pos + vel*dt`, `error = measured - predicted`, `pos += alpha*error`, `vel += beta*error/dt`.

| Parameter | Default | What it does | Raise it | Lower it |
|---|---|---|---|---|
| `alpha` | 0.85 | How much the position trusts the new measurement (0-1). | Follows the raw measurement (1 = no smoothing). | Smoother position, more lag. |
| `beta` | 0.50 | How quickly the velocity estimate reacts. | Faster response to speed changes, noisier. | Smoother velocity, laggier. Keep it at or below `alpha`. |

The CSV also contains the unfiltered velocity (`vrawx_um_s`, `vrawy_um_s`), so you can compare.

### Brightness and workspace handling

| Parameter | Default | What it does | When to change |
|---|---|---|---|
| `drift_comp` | True | Measures the median brightness of ~400 reference background pixels every frame and shifts the image to cancel global brightness changes (what auto-exposure or flickering light does). | Leave on. Turn off only if exposure is truly locked and you want to save ~0.17 ms/frame. It only corrects global, additive changes. |
| `shift_check_every` | 30 | Frames between workspace-moved checks (phase correlation, ~1.3 ms each). `0` disables. | Lower = noticed sooner; higher = cheaper. |
| `shift_warn_px` | 2.0 | Warning threshold for the shift (full-resolution pixels). The estimate is rough (it can under-read), so treat it as a detector only. | Raise if it false-alarms on your images; lower for more sensitivity. If it false-alarms often, also raise the `resp > 0.05` cut-off in `ShiftMonitor.check` (`run_tracker.py`). |

### Output

| Parameter | Default | What it does |
|---|---|---|
| `boundary_points` | 12 | Number of boundary points sampled from the contour (drawn as yellow dots and returned as `boundary`). More points cost almost nothing. |

---

## 5. Values hard-coded in the code

These are not in `Cfg`, but you can edit them in place if needed.

| Where | Value | Meaning |
|---|---|---|
| `fast_tracker.py`, `learn()` | `area < 12` | Blob under 12 px² in the box is rejected ("draw the box tighter"). |
| `learn()` | `GaussianBlur((5,5))` | Smoothing used only during learning. |
| `learn()` | Otsu threshold inside your box | Initial robot outline. Your box should contain the robot plus a bit of background, and nothing else dark. |
| `learn()` | blobs `>= 0.1 * area` | Smallest dark blob that gets auto-excluded. |
| `learn()` | 400 points, stride 6 | Reference pixels for drift compensation. |
| `step()` | `clip(shift, -60, 60)` | Largest brightness correction in grey levels. |
| `step()` | `2 * half >= 0.8 * min(H, W)` | If the search window would cover most of the image, the whole frame is used. |
| `step()` | `dt` limited to 1 ms - 0.5 s | Protects the velocity estimate from timing glitches. |
| `step()`, `vmax` | `max(max_speed, 2 * |v|)` | The window also grows if the robot is currently faster than `max_speed_um_s`. |
| `_search()` | at least 8 border-free contour points, circle-fit error `<= 0.15 * R` | Partial (image-border) detection quality limits. |
| `start_bg_collect(n=45)` | 45 frames | Length of background collection. Longer = cleaner median, needs a longer-moving robot. |
| `_finish_bg_collect()` | travel `>= 6 * R` (about 3 diameters) | Otherwise the refinement is refused ("robot only moved ..."). |
| `_finish_bg_collect()` | `cover > 0.4 * n`, dilated 5x5 | Pixels the robot sat on for more than 40% of frames are filled from the smooth estimate. |
| `camera.py` | `MJPG`, buffer size 1 | MJPG lets USB 2 cameras reach 30 fps at high resolution; buffer 1 avoids stale frames. |
| `camera.py`, `lock_exposure()` | brightness must change by `> 3%` when exposure changes by 1 stop | The test that detects "driver accepts the value but auto-exposure is still active". |
| `run_tracker.py`, `ShiftMonitor` | 1/4 scale, robot blanked with radius `2R/4 + 3` | How the workspace-shift check ignores the robot. |
| `run_tracker.py` | preview auto-scale to 1280 px wide | Display only. |

---

## 6. `magnification.json`

```json
{
  "capture_resolution": null,
  "default": "2",
  "magnifications": {
    "2": {"um_per_px": 4.66, "verified": true},
    "3": {"um_per_px": 3.107, "verified": false}
  }
}
```

* `magnifications`: one entry per zoom setting. The key is what you pass to `--mag`. `um_per_px` is the pixel
  size. `verified` is just a flag; the program prints a warning and shows "(UNVERIFIED)" when it is false.
  **Only 2x (4.66 um/px) was given; the other values were computed as `9.32 / magnification` and must be measured**
  (image a ruler or a known-size object at each setting and put the real values here).
* `default`: the entry used when `--mag` is not given.
* `capture_resolution`: set it to `[width, height]` you calibrated at (for example `[1280, 720]`).
  If the frames differ, a warning is printed, because um/px is only valid at the calibrated resolution.
* Adding a magnification: add a new key and `um_per_px`. The `[` and `]` keys step through the keys in numeric order.

---

## 7. How the algorithm works

### Idea

Do the expensive thinking **once** (learning), so that every frame is only a handful of cheap
OpenCV calls. The robot never has to be removed from the workspace.

### Step A: learning (key R; ~tens of ms)

1. Blur the current frame. Inside your box, Otsu-threshold and take the largest blob. That is the robot.
   From it: area, equivalent radius `R`, and its core darkness `robot_dark`.
2. Estimate the **local background** over the whole frame while ignoring the robot, your mask and (after the first
   pass) the automatically detected static dark blobs. This is a normalised convolution (a blur that skips excluded
   pixels), computed at 1/2 or 1/4 resolution. It fills in what is under the robot by interpolation.
3. Compute `ratio = robot_dark / background_at_robot` and the **threshold map**
   `T(x,y) = background(x,y) * (1 - thr_frac * (1 - ratio))`, limited to `background - min_margin`.
   Because it is relative to the local background, it handles vignetting and uneven lighting.
4. **Automatic exclusion** (passes 1 to `bg_passes - 1`): everything darker than `T` that is *not* the robot
   is static (only the robot moves), so it is added to the exclusion mask and the background is re-estimated without it.
5. Final map: `T` is stored as 8-bit, and set to **0** wherever the pixel is excluded (your mask + automatic zones).
   A pixel can never be "darker than 0", so excluded areas vanish with zero per-frame cost.
6. Pick ~400 reference background pixels for drift compensation; store robot size, darkness, position.

### Step B (optional): background refinement (key B)

While tracking, collect 45 frames in which the robot is found. Take the per-pixel **median**: the robot moves, so it
drops out and a robot-free background remains (walls and shadows exactly as they are). Pixels where the robot lingered
(> 40% of frames) are filled from the smooth estimate. The threshold map is rebuilt from this exact background and the
automatic exclusion is dropped. This lets the robot cross static shadows.

### Step C: every frame (about 0.4 ms)

```
frame (BGR) -> gray
  1. drift      brightness of ~400 reference pixels vs. learning time -> integer shift
  2. window     crop around the predicted position (size from max speed and time since last seen);
                whole frame if lost for a while
  3. blur       3x3 Gaussian on the crop, then add the drift shift (saturating)
  4. threshold  mask = crop < T[crop]            (one cv2.compare; excluded pixels never pass)
  5. contours   findContours on the mask
  6. select     keep blobs whose area is in range; if several, the one nearest the prediction;
                a blob cut only by the *window* edge triggers one full-frame retry
  7. centre     darkness-weighted centroid inside the blob (sub-pixel);
                blob touching the *image* border -> circle fit with the known radius ("partial")
  8. update     alpha-beta filter -> position and velocity in um and um/s; boundary points; result dict
```

If nothing is found, the tracker stays at the last confirmed position (it does not coast) and the window
keeps growing with the time lost, up to a full-frame search.

### Step D: monitoring (main program)

* **Workspace shift:** every `shift_check_every` frames, phase correlation of 1/4-size images with the robot blanked
  out in both; a warning is shown if the static background moved.
* **Timing:** every stage is timed (`benchmark.py`); the table is printed at exit and with key H.
* **Magnification change:** the tracker is invalidated and the mask cleared; you redo K and R.

### State machine (`run_tracker.py`)

```
SETUP --(R, box ok)--> CONFIRM --(ENTER)--> TRACK
   ^                       |  (R again)        |
   |                       v                   |
   +---- [ ] magnification changed ------------+
TRACK: SPACE toggles tracking ON/OFF, B refines the background, K edits the mask
```

---

## 8. File structure

| File | Role |
|---|---|
| `run_tracker.py` | Main program: sources, state machine, keys, drawing, CSV, workspace-shift monitor, ground-truth evaluation. |
| `fast_tracker.py` | `FastTracker`: learning (threshold map, automatic exclusion), background refinement, per-frame `step()`. This is the core. |
| `camera.py` | `VideoSource` (files) and `CameraSource` (webcam with a background grab thread, exposure locking, driver dialog). |
| `ui.py` | Mask editor and camera panel. |
| `config.py` | `Cfg` (all tunables) and `Scale` (reads `magnification.json`). |
| `benchmark.py` | `Timer` (per-stage timing table). |
| `magnification.json` | um-per-pixel for every zoom setting. |
| `gen_test_video.py` | Makes a synthetic test video + ground truth (`python gen_test_video.py out.avi [--w --h --frames --contrast --drift --shift-at]`). |

Using the tracker from your own code (same keys as the old tracker):

```python
tr = FastTracker(um_per_px, Cfg())
ok, info = tr.learn(gray_frame, (x, y, w, h))   # once
tr.set_user_mask(mask_uint8_or_None)            # optional, any time
res = tr.step(gray_frame)                       # every frame; None if the robot is not found
```

---

## 9. Output

### CSV (`--csv`, one row per tracked frame)

`frame, t_s, kind, x_px, y_px, x_um, y_um, vx_um_s, vy_um_s, speed_um_s, vrawx_um_s, vrawy_um_s, R_um, mag, drift`

* `kind`: `full`, `partial` (robot cut by the image border; centre from a circle fit) or `lost` (other columns empty).
* `x_um, y_um`: position from the image's top-left corner.
* `vx/vy/speed`: filtered velocity; `vraw*`: unfiltered frame-to-frame velocity.
* `t_s`: video time for files, camera time for live.
* `drift`: brightness correction applied this frame (grey levels). If it keeps growing, exposure is not locked.

### `step()` result (a dict, or `None` if the robot is not found)

`kind, contour, center` (px), `R` (px), `center_um, R_um, boundary` (Nx2 px), `vel_um_s, speed_um_s, vel_raw_um_s, window, t, drift`.

---

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Robot lost during fast moves | `max_speed_um_s` too low (first check); then `win_R_factor`; then `area_max_frac` for blur streaks. Shorten the camera exposure to reduce blur. |
| Tracker jumps to a wrong blob | Raise `min_margin` or `thr_frac`; mask the region (K); check that the learned diameter in the console is right (your box included something else). |
| "robot is not dark enough" at setup | Contrast under `1.5 * min_margin` grey levels. Improve lighting or lower exposure, or lower `min_margin` if the noise allows. |
| Robot vanishes near walls or shadows | It is inside an excluded zone. Open K and erase the zone (the automatic zones are editable too), or use `--no-auto-exclude`, then press B. |
| Eraser does not work in the mask editor | Press **E** to make the left button erase (your right button may not register). Press **C** to clear everything. |
| Size looks inflated or too small | Move `thr_frac` (0.5 is unbiased for a blurred edge; lower = bigger). |
| Everything is wrong after zooming or bumping the table | Redo K and R (and B). The threshold map is tied to the old geometry. |
| Velocity too noisy / too laggy | Lower / raise `alpha` and `beta`. |
| HUD shows a growing `drift` value | Exposure is not locked. Try C then E; if it reports that the image does not respond, use the driver dialog (D) or the camera vendor's tool. The tracker still works through `drift_comp`. |
| Workspace-moved warning keeps appearing | Raise `shift_warn_px` (and the `resp` cut-off), or set `shift_check_every = 0`. |
| Slow display, not slow tracking | Use `--preview-scale 0.5`, or `--no-gui`. Press H to see which stage is slow. |
| Wrong um values | `magnification.json`: only 2x is measured. Calibrate the others and check `capture_resolution`. |
