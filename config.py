EXPERIMENT_DIR = 'experiments'
RECORD_CODEC = 'MJPG'


# ---------------- vision ----------------
RMIN = 10
RMAX = 60
HOUGH_PARAM1 = 100   # Canny high threshold used by HoughCircles
HOUGH_PARAM2 = 25    # accumulator threshold (lower = more sensitive, ~imfindcircles Sensitivity 0.9)
CAMERA_INDEX = 0
FRAME_SIZE = (640, 480)
WARMUP_FRAMES = 35

# ---------------- hardware ----------------
ARDUINO_PORT = 'COM6'
PINS = dict(
    yminus_dir='D7', yminus='D11',
    xminus_dir='D12', xminus='D10',
    yplus_dir='D6', yplus='D9',
    xplus_dir='D13', xplus='D3',
)

# ---------------- tracking ----------------
TRACK_LEARN_FRAMES = 30     # live frames used to learn robot size/shape/darkness
TRACK_DARK_MARGIN  = 12     # allowed gray-level rise above learned robot darkness
TRACK_MAX_SPEED    = None   # px/SECOND the robot can physically move (None = 20*radius)
LOST_TIMEOUT       = 0.3    # seconds without robot before motors are cut

# ---------------- navigation ----------------
# K_FALLBACK = 900
# K_FIXED = 30000
# ROA = 10
# LOOKAHEAD = 40           # px, how far ahead on the line the bot steers toward
# CRUISE_SPEED = 35        # px/sec, velocity setpoint
# DECEL_RADIUS = 10        # px, start slowing within this distance of the final goal
# MIN_SPEED = 5            # px/sec floor while decelerating

# USE_FEEDFORWARD = True
# FF_GAIN_SCALE = 1.0
# FF_MAX_MAG = 0.9          # must be <= MAX_PWM
# TRIM_MAX = 1.0            # max PWM the PID may add or subtract around the calibrated base
# MAX_EXCESS_PWM = 1.0      # cap on the v/k term in the calibrated drive
# PWM_SLEW = 0.3            # max PWM increase per second (calibration ramp was much slower)

# KP_V = 2e-3
# KI_V = 3e-3
# KD_V = 0
# MAX_INTEGRAL_V = 100
# SIMPLE_TEST_PWM = None   # back to closed loop
ABORT_LOST_S = 1.5
FREEZE_RADIUS = 20       # below this, direction noise dominates - don't trust it

MAX_PWM = 1.0
MIN_PWM = 0.00
VEL_FILTER_ALPHA_INIT = 0.5   # 1 = no filtering, lower = smoother/laggier
ALPHA_POS = 0.3               # separate from vel filter alpha, tune independently
LINE_TIME_CONST = 1           # seconds - how long a persistent deflection takes to "move" the line

VISUALIZE = True
SHOW_DEBUG = True

CALIB_FILE = 'sensitivity_a.pkl'
ROBOT_ID = 'a'
LOG_DIR = "F:/Mohit's BTP/Python Code/analysis/logs"

# ---------------- calibration ----------------
TEST_MAGNITUDE = 0.15    # PWM used for gain/restoring tests
TEST_DURATION = 2.0      # sec, how long each push lasts
SAMPLE_DT = 0.01         # sec, spacing between position samples while pushing/relaxing

RAMP_START      = 0.01
SWEEP_MAX       = 0.5
RAMP_STEP       = 0.005
MODEL_BINS      = 12
N_REPEATS       = 1      # loops per direction
RELAX_DURATION  = 2.0    # relax 2 s
RAMP_MAX = 0.5
MOVE_THRESH_PX = 3

RELAX_DURATION = 2       # sec, free-drift window after each push
RETURN_TOL_PX = 5        # how close to pos0 counts as "returned home"

N_LOOPS = 3              # repetitions per axis
RUN_PIN_DIAG = False
STORE_RAW_TRACES = True
MOVING_SPEED_PX_S = 8   # speed along the commanded direction that counts as "moving"


RETREAT_STEP_PWM = 0.005  # step used when lowering PWM back to 0 (raise, e.g. 0.002, to go faster)
STEP_HOLD = 0.3            # sec each PWM level is held (needs >= 3 samples: 0.4 / SAMPLE_DT = 5)
SWEEP_MAX = 0.5            # absolute safety cap on PWM during the wall sweep
FINE_START_BACKOFF = 0.02  # fine ramp starts this far below the coarse threshold
WALL_MARGIN_PX = 40        # bot counts as "at the wall" this far inside the edge
WALL_LOST_STEPS = 2        # consecutive steps with bot not visible => treated as out of workspace
WALL_NEAR_MARGIN_PX = 80   # "vanished near the wall" if last position was this close to the edge

MODEL_BINS = 12                 # position bins per ray
USE_HEADING_CORRECTION = False
HEADING_CORR_GAIN = 1.0


# ---------------- manual coil control ----------------
KEY_HOLD_TIMEOUT = 0.75   # sec; repeats of a toggle key within this gap count as "held"
DEFAULT_PWM = 0.0  # value with which the PWM will start 
# MANUAL_RAMP_STEP = 0.002
MANUAL_RAMP_STEP = 0.002


# ---------------- manual coil control with recording ----------------
RECORD_FPS = 30
# RECORD_CODEC, EXPERIMENT_DIR, MIN_RAMP_STEP, MM_PER_PX


"""All tunables in one place. Lengths are in micrometres, speeds in um/s, time in s."""
import json
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Cfg:
    # ---- physics / motion
    max_speed_um_s: float = 5000.0     # fastest the robot can possibly go (sets search window)
    # ---- segmentation
    blur_k: int = 3                    # Gaussian blur on the search window (odd; 0 = off)
    thr_frac: float = 0.8              # threshold = this fraction of the way from local bg to robot darkness
    min_margin: float = 4.0            # threshold is always >= this many grey levels below local bg
    bg_sigma_diams: float = 4.0        # background smoothness in robot diameters (set-up only)
    bg_passes: int = 3                 # set-up: bg / auto-exclusion refinement passes
    auto_exclude: bool = True          # set-up: every OTHER dark blob in the learn frame is static -> exclude
    auto_exclude_grow_px: int = 4
    # ---- candidate acceptance (only an area sanity gate, no shape comparison)
    area_min_frac: float = 0.30
    area_max_frac: float = 4.0         # >1 so motion-blur streaks are accepted
    partial_min_frac: float = 0.08     # min visible area when the robot is cut by the image border
    # ---- search window
    win_R_factor: float = 3.0
    full_search_after: int = 5         # frames lost -> search the whole frame
    # ---- velocity filter (alpha-beta)
    alpha: float = 0.85
    beta: float = 0.50
    # ---- misc
    drift_comp: bool = True            # compensate global brightness drift (exposure not truly locked)
    boundary_points: int = 12
    shift_check_every: int = 30        # frames between workspace-shift checks (0 = off)
    shift_warn_px: float = 2.0


class Scale:
    """um-per-pixel for the current zoom setting, from magnification.json."""

    def __init__(self, path=None, key=None):
        path = path or os.path.join(HERE, "magnification.json")
        with open(path) as f:
            d = json.load(f)
        self.table = d["magnifications"]
        self.keys = sorted(self.table, key=float)
        self.capture_resolution = d.get("capture_resolution")
        self.key = key if key is not None else d.get("default", self.keys[0])
        if self.key not in self.table:
            raise KeyError(f"magnification {self.key!r} not in {self.keys}")

    @property
    def um_per_px(self):
        return float(self.table[self.key]["um_per_px"])

    @property
    def verified(self):
        return bool(self.table[self.key].get("verified", False))

    def step(self, d):
        i = self.keys.index(self.key) + d
        self.key = self.keys[max(0, min(len(self.keys) - 1, i))]
        return self.key

    def check_resolution(self, w, h):
        r = self.capture_resolution
        if r and (int(r[0]), int(r[1])) != (w, h):
            print(f"[warn] frame is {w}x{h} but magnification.json was calibrated at {r[0]}x{r[1]}: "
                  f"um/px is WRONG unless you rescale it")
