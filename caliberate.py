from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import config as C
from hardware import Arduino, Camera, currents
from storage import save_calibration
import sys
import cv2
from tracker import setup_tracker
from vision import set_tracker, get_bot_pos
import time


# region Caliberation Tools



def _collect(cam, duration, dt, stop=None):
    """Collects bot positions for `duration` s at ~dt; exits early if `stop` event is set."""
    t, xy = [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        if stop is not None and stop.is_set():
            break
        pos = get_bot_pos(cam)
        if np.all(np.isfinite(pos)):
            t.append(time.perf_counter() - t0)
            xy.append(pos)
        time.sleep(dt)
    return np.array(t), np.array(xy).reshape(-1, 2)


def fit_velocity(t, xy):
    """Linear fit over the whole trace; falls back to endpoint difference if too few points."""
    if len(xy) < 3:
        if len(xy) >= 2:
            dt = t[-1] - t[0]
            return (xy[-1] - xy[0]) / dt if dt > 0 else np.zeros(2)
        return np.zeros(2)
    px = np.polyfit(t, xy[:, 0], 1)
    py = np.polyfit(t, xy[:, 1], 1)
    return np.array([px[0], py[0]])

def test_individual_pins(cam, ard, magnitude, duration):
    """Actuates each coil ALONE and measures displacement, to separate wiring faults from restoring force."""
    dirs = {'xplus': [1, 0], 'xminus': [-1, 0], 'yplus': [0, 1], 'yminus': [0, -1]}
    results = []
    for name, dirvec in dirs.items():
        pos0 = get_bot_pos(cam)
        currents(ard, dirvec, magnitude)
        time.sleep(duration)
        currents(ard, [0, 0], 0)
        time.sleep(0.3)

        delta = get_bot_pos(cam) - pos0
        results.append(dict(pin=name, disp_x=delta[0], disp_y=delta[1], disp_mag=np.linalg.norm(delta)))

        currents(ard, [-v for v in dirvec], magnitude)
        time.sleep(duration)
        currents(ard, [0, 0], 0)
        time.sleep(0.3)
    return results


def _toward_wall(pos, d_img, margin):
    """True if pos is within `margin` px of an image edge that the motion points TOWARD."""
    W, H = C.FRAME_SIZE
    return bool((d_img[0] >  0.1 and pos[0] > W - margin) or
                (d_img[0] < -0.1 and pos[0] < margin) or
                (d_img[1] >  0.1 and pos[1] > H - margin) or
                (d_img[1] < -0.1 and pos[1] < margin))


def _hold_and_log(cam, ard, dirvec, mag, hold, dt, t_start, phase, stop=None):
    """Apply one PWM level for ``hold`` seconds and summarise the response.
 
    Parameters
    ----------
    cam, ard : Camera, Arduino
        Hardware handles.
    dirvec : array-like of shape (2,)
        Direction vector passed to ``currents``.
    mag : float
        Duty cycle applied during this step.
    hold : float
        Duration (s) the duty cycle is held while the bot is tracked.
    dt : float
        Sampling interval (s) between camera reads.
    t_start : float
        ``time.perf_counter()`` value at the start of the whole sweep, used
        to give each step a global timestamp.
    phase : int
        0 for the ramp-up leg, 1 for the retreat (ramp-down) leg.
 
    Returns
    -------
    record : dict or None
        Keys ``t, mag, pos, vel, phase``: step mid-time, duty cycle, mean
        position (2,), fitted velocity (2,) in px/s and phase flag.
        None if the bot was not visible during the whole step.
    last_pos : ndarray or None
        Last valid position seen in the step (None if bot not visible).
    """
    currents(ard, dirvec, mag)
    t, xy = _collect(cam, hold, dt, stop)
    if len(xy) == 0:
        return None, None
    record = dict(t=time.perf_counter() - t_start - 0.5 * hold, mag=mag,
                  pos=xy.mean(axis=0), vel=fit_velocity(t, xy), phase=phase)
    return record, xy[-1]
 
 
def sweep_direction(cam, ard, dirvec, label=''):
    """Ramp the duty cycle up until the bot hits the wall, then ramp it down.
 
    Leg 1 (ramp up): starting at ``start_mag`` the duty cycle is raised by
    ``C.RAMP_STEP`` every ``C.STEP_HOLD`` seconds. Each step is logged, so a
    "moves, stalls, needs more PWM, moves again" pattern caused by the
    meniscus shows up as velocity-vs-PWM / position-vs-PWM data. The leg ends
    when the bot reaches the wall (see :func:`is_at_wall`), disappears from
    the camera for ``C.WALL_LOST_STEPS`` consecutive steps, or ``C.SWEEP_MAX``
    is exceeded.
 
    Leg 2 (retreat): the duty cycle is lowered by ``C.RETREAT_STEP_PWM`` per
    step, all the way to 0 (not cut abruptly), so the meniscus can pull the
    bot back. It runs to 0 even if the bot has not returned home.
 
    Parameters
    ----------
    cam, ard : Camera, Arduino
        Hardware handles.
    dirvec : array-like of shape (2,)
        Direction vector passed to ``currents``.
    start_mag : float
        Duty cycle at which the fine ramp starts (usually a bit below the
        coarse threshold from :func:`ramp_find_threshold`).
 
    Returns
    -------
    dict
        ``home``        : (2,) position before the sweep
        ``wall_mag``    : duty cycle at which the wall was reached (nan if never)
        ``wall_pos``    : (2,) last position of the ramp-up leg
        ``returned``    : bool, bot ended within ``C.RETURN_TOL_PX`` of home
        ``t, mag, pos, vel, phase`` : per-step arrays (pos/vel are (N, 2))
    """
    d = np.asarray(dirvec, dtype=float)
    d_img = np.array([d[0], -d[1]])        # image coords (y down), same flip as your speed calc

    home = get_bot_pos(cam)
    t_start = time.perf_counter()
    records = []
    mag, applied = C.RAMP_START, C.RAMP_START
    lost, threshold, last_pos = 0, np.nan, home
    wall_reason = 'sweep_max'

    # ---- leg 1: ramp up until the wall is detected automatically ----
    while mag <= C.SWEEP_MAX:
        applied = mag
        cam._overlay = [f'{label}: PWM={mag:.4f}']
        rec, pos = _hold_and_log(cam, ard, dirvec, mag, C.STEP_HOLD, C.SAMPLE_DT, t_start, 0)
        if rec is None:                                    # tracker: no full robot visible
            lost += 1
            if lost >= C.WALL_LOST_STEPS:
                near = np.all(np.isfinite(last_pos)) and \
                    _toward_wall(last_pos, d_img, C.WALL_NEAR_MARGIN_PX)
                wall_reason = 'left_view' if near else 'TRACKING_LOST'
                break
        else:
            lost = 0
            records.append(rec)
            last_pos = pos
            if _toward_wall(pos, d_img, C.WALL_MARGIN_PX):
                wall_reason = 'wall_margin'
                break
            if float((rec['vel'] * np.array([1.0, -1.0])) @ d) > C.MOVING_SPEED_PX_S:
                if np.isnan(threshold):
                    threshold = mag
                continue                                   # still moving -> hold
        mag += C.RAMP_STEP                                 # stalled (or not visible) -> raise

    wall_mag = applied
    wall_pos = np.array(last_pos)
    print(f'  wall ({wall_reason}) at PWM={wall_mag:.4f}, pos=[{wall_pos[0]:.0f} {wall_pos[1]:.0f}]')
    if wall_reason == 'TRACKING_LOST':
        print('  WARNING: robot vanished far from any wall - check the tracker / lighting')

    # ---- leg 2: gradual retreat to 0 (SPACE ignored) ----
    mag = min(applied, C.SWEEP_MAX)
    cam._overlay = [f'{label}: retreating...']
    while mag > 0:
        mag = max(mag - C.RETREAT_STEP_PWM, 0.0)
        rec, _ = _hold_and_log(cam, ard, dirvec, mag, C.STEP_HOLD, C.SAMPLE_DT, t_start, 1)
        if rec is not None:
            records.append(rec)
    currents(ard, [0, 0], 0)

    _, xy_end = _collect(cam, C.RELAX_DURATION, C.SAMPLE_DT)
    end_pos = xy_end[-1] if len(xy_end) else np.full(2, np.nan)
    returned = bool(np.linalg.norm(end_pos - home) < C.RETURN_TOL_PX)
    cam._overlay = []

    return dict(home=home, threshold=threshold, wall_mag=wall_mag, wall_pos=wall_pos,
            wall_reason=wall_reason, returned=returned,
                t=np.array([r['t'] for r in records]),
                mag=np.array([r['mag'] for r in records]),
                pos=np.array([r['pos'] for r in records]).reshape(-1, 2),
                vel=np.array([r['vel'] for r in records]).reshape(-1, 2),
                phase=np.array([r['phase'] for r in records]))

# endregion



_R = 1 / np.sqrt(2)


# 8 directions. Vectors are unit length so a given duty cycle means the same
# total drive for axis and diagonal directions. Image y grows downward, so
# "+Y" is whatever your existing [0, 1] vector does physically.
DIRECTIONS = [
    dict(name='pX',   label='+X',    vec=[1, 0]),
    dict(name='nX',   label='-X',    vec=[-1, 0]),
    dict(name='pY',   label='+Y',    vec=[0, 1]),
    dict(name='nY',   label='-Y',    vec=[0, -1]),
    dict(name='pXpY', label='+X+Y',  vec=[_R, _R]),
    dict(name='pXnY', label='+X-Y',  vec=[_R, -_R]),
    dict(name='nXpY', label='-X+Y',  vec=[-_R, _R]),
    dict(name='nXnY', label='-X-Y',  vec=[-_R, -_R]),
]


def calibrate_direction(cam, ard, D):
    print(f"\n=== Calibrating {D['label']} ===")
    runs = [sweep_direction(cam, ard, D['vec'], f"{D['label']} #{i+1}")
            for i in range(C.N_REPEATS)]
    res = dict(
        home=np.median([r['home'] for r in runs], axis=0),
        wall_pos=np.median([r['wall_pos'] for r in runs], axis=0),
        threshold=float(np.nanmedian([r['threshold'] for r in runs])),
        wall_mag=max(r['wall_mag'] for r in runs),
        wall_reason=runs[-1]['wall_reason'],
        returned=all(r['returned'] for r in runs),
        vec=np.array(D['vec']),
    )
    for k in ('t', 'mag', 'pos', 'vel', 'phase'):
        res[k] = np.concatenate([r[k] for r in runs])
    print(f"  threshold={res['threshold']:.4f}, returned home: {res['returned']}")
    return res

def plot_results(calib):
    """Plot speed-along-direction vs PWM per direction, and the position map.

    Ramp-up samples are drawn in blue and retreat samples in red, so
    hysteresis and stall/restart steps are easy to see.
    """
    try:
        fig, axs = plt.subplots(3, 3, figsize=(13, 10), num='Speed vs PWM')
        for ax, D in zip(axs.flat, DIRECTIONS):
            r = calib[D['name']]
            speed = (r['vel'] * np.array([1.0, -1.0])) @ r['vec']             # px/s along commanded direction
            up, dn = r['phase'] == 0, r['phase'] == 1
            ax.plot(r['mag'][up], speed[up], 'b.-', ms=3, lw=0.5, label='ramp up')
            ax.plot(r['mag'][dn], speed[dn], 'r.-', ms=3, lw=0.5, label='retreat')
            ax.set_title(D['label'])
            ax.set_xlabel('PWM')
            ax.set_ylabel('speed (px/s)')
        axs.flat[0].legend()
        axs.flat[-1].axis('off')
        fig.tight_layout()

        plt.figure('Sweep map', figsize=(7, 6))
        for D in DIRECTIONS:
            r = calib[D['name']]
            up = r['phase'] == 0
            sc = plt.scatter(r['pos'][up, 0], r['pos'][up, 1], c=r['mag'][up], s=8)
        plt.colorbar(sc, label='PWM')
        plt.xlim([0, C.FRAME_SIZE[0]])
        plt.ylim([0, C.FRAME_SIZE[1]])
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.title('Bot position during ramp-up (colour = PWM)')
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.show()
    except Exception as e:
        print(f'Plotting failed: {e}')

def main():
    print('Connecting to Arduino...')
    ard = Arduino(C.ARDUINO_PORT, C.PINS)

    print('Initializing Camera...')
    cam = Camera()
    cam.warmup()
    tracker = setup_tracker(cam, 'Select robot', C.TRACK_LEARN_FRAMES,
                        C.TRACK_MAX_SPEED, C.TRACK_DARK_MARGIN)
    cv2.destroyWindow('Select robot'); cv2.waitKey(1)
    if tracker is None:
        cam.release(); sys.exit()
    set_tracker(tracker)
    cam.start_preview()

    pin_diag = None
    if C.RUN_PIN_DIAG:
        print('=== Per-pin diagnostic ===')
        pin_diag = test_individual_pins(cam, ard, C.TEST_MAGNITUDE, C.TEST_DURATION)
        for row in pin_diag:
            print(row)

    calib = {}
    try:
        for D in DIRECTIONS:
            calib[D['name']] = calibrate_direction(cam, ard, D)
    finally:
        # always switch the coils off, even if something fails mid-sweep
        try:
            currents(ard, [0, 0], 0)
        except Exception:
            pass

    if pin_diag is not None:
        calib['pin_diag'] = pin_diag
    calib['robot_id'] = C.ROBOT_ID
    calib['timestamp'] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

    cam.release()
    save_calibration(calib)
    plot_results(calib)


if __name__ == '__main__':
    main()