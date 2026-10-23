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
    xminus_dir='D7', xminus='D11',
    yminus_dir='D12', yminus='D10',
    yplus_dir='D13', yplus='D9',
    xplus_dir='D2', xplus='D3',
)

# ---------------- tracking ----------------
TRACK_LEARN_FRAMES = 30     # live frames used to learn robot size/shape/darkness
TRACK_DARK_MARGIN  = 12     # allowed gray-level rise above learned robot darkness
TRACK_MAX_SPEED    = None   # px/SECOND the robot can physically move (None = 20*radius)
LOST_TIMEOUT       = 0.3    # seconds without robot before motors are cut

# ---------------- navigation ----------------
K_FALLBACK = 900
K_FIXED = 30000
ROA = 10
LOOKAHEAD = 40           # px, how far ahead on the line the bot steers toward
CRUISE_SPEED = 35        # px/sec, velocity setpoint
DECEL_RADIUS = 10        # px, start slowing within this distance of the final goal
MIN_SPEED = 5            # px/sec floor while decelerating

USE_FEEDFORWARD = True
FF_GAIN_SCALE = 1.0
FF_MAX_MAG = 0.9          # must be <= MAX_PWM
TRIM_MAX = 1.0            # max PWM the PID may add or subtract around the calibrated base
MAX_EXCESS_PWM = 1.0      # cap on the v/k term in the calibrated drive
PWM_SLEW = 0.3            # max PWM increase per second (calibration ramp was much slower)

KP_V = 2e-3
KI_V = 3e-3
KD_V = 0
MAX_INTEGRAL_V = 100
SIMPLE_TEST_PWM = None   # back to closed loop
ABORT_LOST_S = 1.5
FREEZE_RADIUS = 20       # below this, direction noise dominates - don't trust it

MAX_PWM = 1.0
MIN_PWM = 0.00
VEL_FILTER_ALPHA_INIT = 0.5   # 1 = no filtering, lower = smoother/laggier
ALPHA_POS = 0.4               # separate from vel filter alpha, tune independently
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
MANUAL_RAMP_STEP = 0.002


# ---------------- manual coil control with recording ----------------
RECORD_FPS = 30
# RECORD_CODEC, EXPERIMENT_DIR, MIN_RAMP_STEP, MM_PER_PX