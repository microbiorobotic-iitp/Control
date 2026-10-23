import time

import cv2
import numpy as np

import config as C
from hardware import Arduino, Camera

DASH = 'Magnetic Coil Control - PRESS ESC TO QUIT'
CAM_WIN = 'Camera'

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

GREEN, RED, GREY, BLACK = (0, 153, 0), (0, 0, 153), (102, 102, 102), (0, 0, 0)


def new_state():
    return dict(run=True, selected='w',
                on={k: False for k in COIL_NAME},
                pwm={k: C.DEFAULT_PWM for k in COIL_NAME},
                last_seen={})


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
    if key in COIL_NAME:
        held = now - state['last_seen'].get(key, -np.inf) < C.KEY_HOLD_TIMEOUT
        state['last_seen'][key] = now
        if held:   # prevents flickering toggle if key is held down
            return
        state['on'][key] = not state['on'][key]
    elif key in SELECT_KEYS:
        state['selected'] = SELECT_KEYS[key]
    elif key == 'uparrow':
        s = state['selected']
        state['pwm'][s] = min(1.0, state['pwm'][s] + C.MANUAL_RAMP_STEP)
    elif key == 'downarrow':
        s = state['selected']
        state['pwm'][s] = max(0.0, state['pwm'][s] - C.MANUAL_RAMP_STEP)
    elif key == 'escape':
        state['run'] = False


def render_dashboard(state, size=(600, 400)):
    w, h = size
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, 'MAGNETIC COIL DASHBOARD', (6, int(h * 0.05) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
    cv2.putText(img, '1,2,3,4: Select Coil | UP/DOWN: Adjust PWM | W,A,S,D: Toggle ON/OFF',
                (6, int(h * 0.15) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)
    for key, name, label, _, row in COILS:
        is_on = state['on'][key]
        selected = key == state['selected']
        sel_str = '> [*]' if selected else '  [ ]'
        status = 'ON ' if is_on else 'OFF'
        text = f"{sel_str} {label} | Status: {status} | PWM: {state['pwm'][key]:.4f}"
        cv2.putText(img, text, (int(w * 0.02), int((1 - row) * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    GREEN if is_on else RED, 2 if selected else 1)
    return img


def apply_currents(ard, state):
    for key, name, _, dir_val, _ in COILS:
        ard.write_dir(name, dir_val)
        ard.write_pwm(name, state['pwm'][key] if state['on'][key] else 0)


def stop_all_coils(ard):
    for _, name, *_ in COILS:
        ard.write_pwm(name, 0)


def main():
    print('Connecting to Arduino...')
    ard = Arduino(C.ARDUINO_PORT, C.PINS)

    state = new_state()
    cv2.namedWindow(DASH)
    print('Ready! Click on the figure window to begin.')

    print('Initializing Camera...')
    cam = Camera()

    try:
        while state['run']:
            if cv2.getWindowProperty(DASH, cv2.WND_PROP_VISIBLE) < 1:
                break

            cv2.imshow(CAM_WIN, cam.snapshot())
            cv2.imshow(DASH, render_dashboard(state))
            key = decode_key(cv2.waitKeyEx(20))
            if key is not None:
                key_down(state, key, time.perf_counter())

            apply_currents(ard, state)
    finally:
        print('Stopping coils and cleaning up...')
        stop_all_coils(ard)
        cam.release()
        cv2.destroyAllWindows()
        print('Process Complete.')


if __name__ == '__main__':
    main()