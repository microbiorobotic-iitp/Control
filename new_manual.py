import time

import cv2
import numpy as np

import config as C
from hardware import Arduino, Camera
from recording import (DUTY_COLUMNS, Experiment, Recorder, ask_meta, duty_columns, robot_geometry)
from tracker import setup_tracker

DASH = 'Magnetic Coil Control - PRESS ESC TO QUIT'
CAM_WIN = 'Camera'

# ------------------------------------------------------------------ EDIT THESE FREELY
# (key used in meta.json, question shown in the terminal, default answer)
META_QUESTIONS = [
    ('liquid',          'What is the liquid?',            ''),
    ('liquid_depth_mm', 'Depth of the liquid (mm)?',      ''),
    ('float',           'Is Microrobot floating on surface? (True/False)', ''),
]

MIN_STEP = getattr(C, 'MIN_RAMP_STEP', 0.0005)              # smallest duty change (least count)
RAMP_LADDER = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05]   # [ and ] move along this list
TRACK_DURING_RUN = False     # True -> also log robot x,y per frame (costs a little CPU in the loop)
RECORD_FPS = getattr(C, 'RECORD_FPS', 30)                   # nominal fps stored in the video header
RECORD_CODEC = getattr(C, 'RECORD_CODEC', 'MJPG')           # MJPG/.avi = light on CPU, frame-accurate
EXPERIMENT_DIR = getattr(C, 'EXPERIMENT_DIR', 'experiments')
# ------------------------------------------------------------------

# key, coil/pin name, label, dir-pin value, dashboard row (fraction from bottom)
COILS = [
    ('w', 'yplus',  '1. +Y Coil (W)', 0, 0.70),
    ('a', 'xminus', '2. -X Coil (A)', 0, 0.55),
    ('s', 'yminus', '3. -Y Coil (S)', 1, 0.40),   # dir pin is 1 here, carried over from original logic
    ('d', 'xplus',  '4. +X Coil (D)', 0, 0.25),
]
COIL_NAME = {k: name for k, name, *_ in COILS}
SELECT_KEYS = {'1': 'w', '2': 'a', '3': 's', '4': 'd'}

UP_CODES = {2490368, 65362, 0x260000}
DOWN_CODES = {2621440, 65364, 0x280000}

GREEN, RED, GREY, BLACK, BLUE = (0, 153, 0), (0, 0, 153), (102, 102, 102), (0, 0, 0), (153, 51, 0)

COLUMNS = DUTY_COLUMNS + ['selected_coil', 'ramp_step'] + (['x_px', 'y_px', 'track_kind'] if TRACK_DURING_RUN else [])


def quantize(v):
    """Snap to a multiple of MIN_STEP and clip to [0, 1]."""
    return float(min(1.0, max(0.0, round(round(v / MIN_STEP) * MIN_STEP, 6))))


def new_state():
    ladder = sorted(set(round(v, 6) for v in RAMP_LADDER + [C.MANUAL_RAMP_STEP]))
    return dict(run=True, selected='w',
                on={k: False for k in COIL_NAME},
                pwm={k: quantize(C.DEFAULT_PWM) for k in COIL_NAME},
                last_seen={},
                ladder=ladder, ramp_idx=ladder.index(round(C.MANUAL_RAMP_STEP, 6)),
                sent={},              # coil name -> duty actually written to the Arduino
                rec_toggle=False)


def ramp_step(state):
    return state['ladder'][state['ramp_idx']]


def decode_key(code):
    if code == -1:
        return None
    if code in UP_CODES:
        return 'uparrow'
    if code in DOWN_CODES:
        return 'downarrow'
    if code == 27:
        return 'escape'
    if 0 <= code < 256:
        return chr(code).lower()
    return None


def key_down(state, key, now):
    """Updates state. Returns a short text describing what changed (goes to events.csv) or None."""
    if key in COIL_NAME:
        held = now - state['last_seen'].get(key, -np.inf) < C.KEY_HOLD_TIMEOUT
        state['last_seen'][key] = now
        if held:   # prevents flickering toggle if key is held down
            return None
        state['on'][key] = not state['on'][key]
        return f"coil {COIL_NAME[key]} {'ON' if state['on'][key] else 'OFF'}"
    if key in SELECT_KEYS:
        state['selected'] = SELECT_KEYS[key]
        return f"select {COIL_NAME[state['selected']]}"
    if key in ('uparrow', 'downarrow'):
        s = state['selected']
        sign = 1 if key == 'uparrow' else -1
        state['pwm'][s] = quantize(state['pwm'][s] + sign * ramp_step(state))
        return f"pwm {COIL_NAME[s]} set to {state['pwm'][s]:.4f}"
    if key == ']':
        state['ramp_idx'] = min(len(state['ladder']) - 1, state['ramp_idx'] + 1)
        return f"ramp_step = {ramp_step(state):.4f}"
    if key == '[':
        state['ramp_idx'] = max(0, state['ramp_idx'] - 1)
        return f"ramp_step = {ramp_step(state):.4f}"
    if key == 'r':
        state['rec_toggle'] = True
    elif key == 'escape':
        state['run'] = False
    return None


def render_dashboard(state, rec=None, size=(640, 420)):
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, 'MAGNETIC COIL DASHBOARD', (6, int(h * 0.05) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
    cv2.putText(img, '1,2,3,4: Select Coil | UP/DOWN: Adjust PWM | W,A,S,D: Toggle ON/OFF',
                (6, int(h * 0.14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
    cv2.putText(img, '[ / ]: smaller / bigger ramp step | R: start / stop recording',
                (6, int(h * 0.20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
    for key, name, label, _, row in COILS:
        is_on = state['on'][key]
        selected = key == state['selected']
        sel_str = '> [*]' if selected else '  [ ]'
        status = 'ON ' if is_on else 'OFF'
        text = f"{sel_str} {label} | Status: {status} | PWM: {state['pwm'][key]:.4f}"
        cv2.putText(img, text, (int(w * 0.02), int((1 - row) * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    GREEN if is_on else RED, 2 if selected else 1)
    cv2.putText(img, f"Ramp step (UP/DOWN change): {ramp_step(state):.4f}   (min {MIN_STEP})",
                (int(w * 0.02), int(h * 0.86)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLUE, 2)
    if rec is None:
        cv2.putText(img, 'Recording: OFF (press R)', (int(w * 0.02), int(h * 0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1)
    else:
        cv2.putText(img, f"RECORDING {rec.elapsed():6.1f}s  frames={rec.written}  dropped={rec.dropped}  "
                         f"backlog={rec.backlog}", (int(w * 0.02), int(h * 0.95)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return img


def apply_currents(ard, state):
    """Writes to the Arduino only when a duty actually changed (less serial traffic -> faster loop)."""
    for key, name, _, dir_val, _ in COILS:
        duty = state['pwm'][key] if state['on'][key] else 0.0
        if state['sent'].get(name) != duty:
            ard.write_dir(name, dir_val)
            ard.write_pwm(name, duty)
            state['sent'][name] = duty


def stop_all_coils(ard):
    for _, name, *_ in COILS:
        try:
            ard.write_pwm(name, 0)
        except Exception:
            pass


def draw_preview(frame, state, rec, track_res):
    view = frame.copy()
    if track_res is not None:
        c = track_res['center']
        cv2.circle(view, (int(c[0]), int(c[1])), int(track_res['R']), (0, 255, 0), 2)
    if rec is not None:
        cv2.circle(view, (20, 20), 8, (0, 0, 255), -1)
        cv2.putText(view, f'REC {rec.elapsed():.1f}s', (36, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    sent = state['sent']
    cv2.putText(view, f"X: +{sent.get('xplus', 0):.4f} -{sent.get('xminus', 0):.4f}   "
                      f"Y: +{sent.get('yplus', 0):.4f} -{sent.get('yminus', 0):.4f}",
                (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return view


def start_run(exp, state):
    rec = Recorder(exp.new_run_dir(), COLUMNS, fps=RECORD_FPS, codec=RECORD_CODEC,
                   info=dict(initial_pwm={COIL_NAME[k]: v for k, v in state['pwm'].items()},
                             initial_on={COIL_NAME[k]: v for k, v in state['on'].items()},
                             ramp_step=ramp_step(state)))
    rec.event('recording started')
    print(f'[REC] started -> {rec.folder}')
    return rec


def stop_run(rec):
    rec.event('recording stopped')
    info = rec.close()
    print(f"[REC] saved {info['n_frames']} frames, dropped {info['dropped_frames']}, "
          f"{info['duration_s']:.1f}s, measured fps={info['measured_fps']}  -> {rec.folder}")


def main():
    exp_name = input('Experiment name (no spaces): ').strip() or 'manual'
    answers = ask_meta(META_QUESTIONS)

    print('Connecting to Arduino...')
    ard = Arduino(C.ARDUINO_PORT, C.PINS)

    print('Initializing Camera...')
    cam = Camera()
    cam.warmup()

    rec = None
    try:
        # ---- 1. learn the robot's geometry
        cv2.namedWindow(CAM_WIN)
        tracker = setup_tracker(cam, CAM_WIN, C.TRACK_LEARN_FRAMES, C.TRACK_MAX_SPEED, C.TRACK_DARK_MARGIN)
        if tracker is None:
            return
        geometry = robot_geometry(tracker, getattr(C, 'MM_PER_PX', None))
        print('Robot geometry:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in geometry.items()})

        exp = Experiment(
            exp_name, answers, geometry, root=EXPERIMENT_DIR,
            settings=dict(ramp_ladder=RAMP_LADDER, min_ramp_step=MIN_STEP, default_pwm=C.DEFAULT_PWM,
                          coil_dir_values={name: d for _, name, _, d, _ in COILS},
                          pins=C.PINS, camera_index=C.CAMERA_INDEX, frame_size=C.FRAME_SIZE,
                          record_codec=RECORD_CODEC, track_during_run=TRACK_DURING_RUN))
        print(f'Experiment folder: {exp.dir}')

        # ---- 2. manual control loop
        state = new_state()
        cv2.namedWindow(DASH)
        apply_currents(ard, state)            # all coils off, dir pins set
        print('Ready!  R = start/stop recording, ESC = quit.')

        while state['run']:
            if cv2.getWindowProperty(DASH, cv2.WND_PROP_VISIBLE) < 1:
                break

            frame = cam.snapshot()                    # blocks until a NEW frame -> loop runs at camera rate
            t_frame = time.perf_counter()

            track_res, track_row = None, {}
            if TRACK_DURING_RUN:
                track_res = tracker.step(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), now=t_frame)
                if track_res is not None:
                    track_row = dict(x_px=track_res['center'][0], y_px=track_res['center'][1],
                                     track_kind=track_res['kind'])

            # duty in state['sent'] is what was ACTIVE while this frame was captured
            if rec is not None:
                rec.log(frame, t=t_frame, **duty_columns(state['sent']),
                        selected_coil=COIL_NAME[state['selected']], ramp_step=ramp_step(state), **track_row)

            cv2.imshow(CAM_WIN, draw_preview(frame, state, rec, track_res))
            cv2.imshow(DASH, render_dashboard(state, rec))

            key = decode_key(cv2.waitKeyEx(1))
            if key is not None:
                ev = key_down(state, key, t_frame)
                if ev and rec is not None:
                    rec.event(ev)

            if state['rec_toggle']:
                state['rec_toggle'] = False
                if rec is None:
                    rec = start_run(exp, state)
                else:
                    stop_run(rec)
                    rec = None

            apply_currents(ard, state)
    finally:
        print('Stopping coils and cleaning up...')
        stop_all_coils(ard)
        if rec is not None:
            stop_run(rec)
        try:
            exp.update_meta(ended=time.strftime('%Y-%m-%dT%H:%M:%S'), n_runs=exp.n_runs)
        except NameError:
            pass
        cam.release()
        cv2.destroyAllWindows()
        try:
            ard.close()
        except Exception:
            pass
        print('Process Complete.')


if __name__ == '__main__':
    main()