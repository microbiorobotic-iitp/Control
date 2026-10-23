import numpy as np
from calib_model import restoring_drive
import config as C


class VelocityPID:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = None
        self.filtered_speed = 0.0
        self.in_decel = False
        self.was_in_decel = False

    def step(self, forward_speed, dist_to_waypoint, dist_final, dt, phi=0.0, G=None):
        if not self.in_decel and dist_to_waypoint < C.DECEL_RADIUS:
            self.in_decel = True
        elif self.in_decel and dist_to_waypoint > C.DECEL_RADIUS * 1.15:   # exit buffer
            self.in_decel = False

        target_speed = C.CRUISE_SPEED
        alpha = C.VEL_FILTER_ALPHA_INIT
        if dist_to_waypoint < C.DECEL_RADIUS:
            alpha = 0.1   # near the target, trust the actual velocity less over the laggy estimate
            target_speed = max(C.MIN_SPEED, C.CRUISE_SPEED * np.sqrt(dist_final / C.DECEL_RADIUS))

        self.filtered_speed = alpha * forward_speed + (1 - alpha) * self.filtered_speed

        error_v = target_speed - self.filtered_speed
        self.integral = max(min(self.integral + error_v * dt, C.MAX_INTEGRAL_V), -C.MAX_INTEGRAL_V)
        deriv_v = 0.0 if self.prev_error is None else (error_v - self.prev_error) / dt
        self.prev_error = error_v

        if self.in_decel and not self.was_in_decel:   # transition edge
            self.integral *= 0.3   # soften the carried-over windup
        self.was_in_decel = self.in_decel

        P = C.KP_V * error_v
        I = C.KI_V * self.integral
        D = C.KD_V * deriv_v
        M = float(np.clip(P + I + D, -C.M_MAX, C.M_MAX))        # px/s, NOT PWM

        G = G if (G is not None and G > 1e-6) else C.K_FALLBACK
        base = phi + target_speed / G                            # friction + feedforward speed
        drive = phi + (target_speed + M) / G                     # slide: (V_des + M)/G + phi
        magnitude = max(C.MIN_PWM, min(C.MAX_PWM, drive))
        return dict(target_speed=target_speed, error_v=error_v, P=P, I=I, D=D, M=M,
                    base=base, magnitude=magnitude, filtered_speed=self.filtered_speed)


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
