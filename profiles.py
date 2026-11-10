"""Per-fluid profiles. One JSON file per fluid in ./profiles/ holds everything that changes with the liquid:
plant gain G (um/s per unit PWM), friction PWM phi, PID gains for BOTH modes, speeds, ramp times and the
calibration file. All lengths/speeds are in micrometres; apply_scale() converts them to pixels for the
current magnification, so control.py / path.py keep working in pixels unchanged."""
import copy
import json
import os

import config as C

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles')

DEFAULTS = {
    "fluid": "unnamed",
    "calib_file": None,                      # e.g. "calibrations/glycerin_a.pkl" (stored in the run meta)
    "tracker_max_speed_um_s": 1500.0,        # search-window speed for the tracker: >= fastest the robot can go
    "approach": {                            # --- direction matters more than speed
        "G_um_s_per_pwm": 4194.0,            # steady speed per unit PWM above phi   (old K_FALLBACK 900 px/s x 4.66)
        "phi_pwm": 0.0,                      # PWM at which the robot just starts moving (friction/deadband)
        "cruise_um_s": 163.0, "min_speed_um_s": 23.0,
        "lookahead_um": 186.0, "decel_radius_um": 47.0, "roa_um": 47.0, "freeze_radius_um": 93.0,
        "Kp": 2e-3, "Ki": 3e-3, "Kd": 0.0,   # unit-less / per second: unchanged from your KP_V / KI_V / KD_V
        "max_integral_um": 466.0, "m_max_um_s": 80.0,
        "alpha_pos": 0.4, "vel_filter_alpha": 0.5, "line_time_const_s": 1.0,
        "pwm_slew_per_s": 0.3, "max_pwm": 1.0, "min_pwm": 0.0,
        "lost_timeout_s": 0.3, "abort_lost_s": 1.5,
        "reconfigure_hold_scale": 1.0       # R during approach: keep this fraction of the last duty (1 = unchanged, 0 = motors off)
    },
    "action": {                              # --- robot is attached to the cell: speed / strain matters
        "G_um_s_per_pwm": 4194.0, "phi_pwm": 0.0,
        "speed_um_s": 50.0, "distance_um": 200.0,
        "ramp_up_s": 3.0, "ramp_down_s": 3.0,
        "retract": True, "retract_speed_um_s": 50.0, "retract_ramp_up_s": 3.0, "retract_ramp_down_s": 3.0, "dwell_s": 1.0,
        "Kp_pos": 1.5, "Ki_pos": 0.6,        # speed correction (um/s) per um of error, and per um*s (tested in simulation only)
        "max_integral_um": 100.0, "max_lag_um": 40.0, "min_cmd_um_s": 2.0,
        "max_pwm": 0.6, "pwm_slew_per_s": 0.3, "end_tol_um": 10.0, "settle_timeout_s": 5.0,
        "lost_timeout_s": 0.3, "abort_lost_s": 1.5
    }
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def list_profiles():
    if not os.path.isdir(PROFILE_DIR):
        return []
    return [f[:-5] for f in sorted(os.listdir(PROFILE_DIR)) if f.endswith('.json') and not f.startswith('_')]


def load_profile(name):
    path = os.path.join(PROFILE_DIR, f'{name}.json')
    with open(path, encoding='utf-8') as f:
        p = _merge(DEFAULTS, json.load(f))
    p['name'] = name
    return p


def apply_scale(profile, upp):
    """Write the approach-mode settings into config (C) in PIXELS for the current um_per_px.
    Call after loading a profile AND every time the magnification changes."""
    a = profile['approach']
    C.CRUISE_SPEED = a['cruise_um_s'] / upp          # px/s
    C.MIN_SPEED = a['min_speed_um_s'] / upp
    C.LOOKAHEAD = a['lookahead_um'] / upp            # px
    C.DECEL_RADIUS = a['decel_radius_um'] / upp
    C.ROA = a['roa_um'] / upp
    C.FREEZE_RADIUS = a['freeze_radius_um'] / upp
    C.K_FALLBACK = a['G_um_s_per_pwm'] / upp         # px/s per PWM  (used as G by VelocityPID)
    C.MAX_INTEGRAL_V = a['max_integral_um'] / upp
    C.M_MAX = a['m_max_um_s'] / upp
    C.KP_V, C.KI_V, C.KD_V = a['Kp'], a['Ki'], a['Kd']
    C.PHI = a['phi_pwm']
    C.ALPHA_POS, C.VEL_FILTER_ALPHA_INIT, C.LINE_TIME_CONST = a['alpha_pos'], a['vel_filter_alpha'], a['line_time_const_s']
    C.PWM_SLEW, C.MAX_PWM, C.MIN_PWM = a['pwm_slew_per_s'], a['max_pwm'], a['min_pwm']
    C.LOST_TIMEOUT, C.ABORT_LOST_S = a['lost_timeout_s'], a['abort_lost_s']
    C.USE_FEEDFORWARD, C.USE_HEADING_CORRECTION, C.SIMPLE_TEST_PWM = False, False, None   # calib_model ignored for now


def summary(profile, upp):
    a, b = profile['approach'], profile['action']
    return (f"profile '{profile['name']}' ({profile['fluid']}) at {upp:.3f} um/px\n"
            f"  approach: G={a['G_um_s_per_pwm']:.0f} um/s/PWM phi={a['phi_pwm']:.3f} cruise={a['cruise_um_s']:.0f} um/s "
            f"Kp={a['Kp']} Ki={a['Ki']}\n"
            f"  action  : G={b['G_um_s_per_pwm']:.0f} um/s/PWM phi={b['phi_pwm']:.3f} speed={b['speed_um_s']:.0f} um/s "
            f"distance={b['distance_um']:.0f} um ramps={b['ramp_up_s']}/{b['ramp_down_s']} s "
            f"Kp_pos={b['Kp_pos']} Ki_pos={b['Ki_pos']} max_pwm={b['max_pwm']}")