import numpy as np
from calib_model import restoring_drive
import config as C

class VelocityPID:
    """PWM is an integrator of the speed error (no plant gain G). Target speed also ramps up over time."""
    def __init__(self, p, u0=0.0):
        self.p = p
        self.u = float(u0); self.filtered_speed = 0.0; self.in_decel = False; self.t = 0.0

    def step(self, forward_speed, dist_to_waypoint, dist_final, dt, u_max=1.0):
        p = self.p
        self.t += dt
        if not self.in_decel and dist_to_waypoint < C.DECEL_RADIUS:
            self.in_decel = True
        elif self.in_decel and dist_to_waypoint > C.DECEL_RADIUS * 1.15:
            self.in_decel = False

        # speed reference: smooth ramp up, sqrt slow-down near the goal
        target_speed = C.CRUISE_SPEED * min(1.0, self.t / p['speed_ramp_s'])
        alpha = C.VEL_FILTER_ALPHA_INIT
        if self.in_decel:
            alpha = 0.1
            target_speed = min(target_speed,
                               max(C.MIN_SPEED, C.CRUISE_SPEED * np.sqrt(dist_final / C.DECEL_RADIUS)))
        self.filtered_speed = alpha * forward_speed + (1 - alpha) * self.filtered_speed

        error_v = target_speed - self.filtered_speed
        e_n = error_v / C.CRUISE_SPEED                    # normalised error, unitless
        P = p['Kp_u'] * e_n
        pre = self.u + p['Ki_u'] * e_n * dt
        if C.MIN_PWM <= pre <= u_max:
            self.u = pre
        self.u = float(np.clip(self.u, C.MIN_PWM, u_max))
        magnitude = float(np.clip(self.u + P, C.MIN_PWM, u_max))
        return dict(target_speed=target_speed, error_v=error_v, P=P, I=self.u, D=0.0, M=P,
                    base=self.u, magnitude=magnitude, filtered_speed=self.filtered_speed)

def axis_limit(direction, cap=1.0):
    m = max(abs(direction[0]), abs(direction[1]))
    return cap / m if m > 1e-6 else cap

def steering_direction(vector_to_target, last_good_dir):
    steer_dist = np.linalg.norm(vector_to_target)
    if steer_dist > C.FREEZE_RADIUS:
        direction = vector_to_target / steer_dist
        return direction, direction
    return last_good_dir, last_good_dir

def feedforward(bot_pos, calib):
    if not C.USE_FEEDFORWARD or calib is None or calib.get('model') is None:
        return np.zeros(2)
    ff = C.FF_GAIN_SCALE * restoring_drive(calib['model'], bot_pos)
    n = np.linalg.norm(ff)
    return ff * (C.FF_MAX_MAG / n) if n > C.FF_MAX_MAG else ff


def combine_commands(direction, magnitude, ff_cmd):
    cmd_total = direction * magnitude + ff_cmd
    total_mag = np.linalg.norm(cmd_total)
    if total_mag > 0:
        direction = cmd_total / total_mag
        magnitude = max(C.MIN_PWM, min(total_mag, C.MAX_PWM))
    return direction, magnitude
