"""Action mode: robot is attached to the cell and is pulled along a fixed direction with a controlled speed.

Instead of regulating velocity (noisy at 30 fps), the controller tracks a REFERENCE POSITION along the pull
direction that moves with a trapezoid speed profile (ramp up -> hold -> ramp down -> [dwell -> same back]):

    e      = s_ref(t) - s_measured                    (um along the pull direction)
    v_cmd  = v_ref(t) + Kp*e + Ki*integral(e)         (feed-forward speed + PI correction, um/s)
    pwm    = phi + |v_cmd| / G                        (same plant model as the approach mode)

Safety: pwm is capped (max_pwm) and slew-limited; the integrator is clamped; and if the robot lags the
reference by more than max_lag_um (cell resisting / robot stuck) the reference clock is PAUSED, so nothing winds up.
"""
import numpy as np


def _trap(D, v, t_up, t_down):
    v = abs(float(v))
    if D < 0.5 * v * (t_up + t_down):                 # too short to reach full speed: triangle profile
        v = 2.0 * D / (t_up + t_down)
    t_hold = (D - 0.5 * v * (t_up + t_down)) / v if v > 0 else 0.0
    return [(t_up, 0.0, v), (t_hold, v, v), (t_down, v, 0.0)]


class Reference:
    def __init__(self, p, dt=0.005):
        self.dt = dt
        D = float(p['distance_um'])
        segs = list(zip(['ramp_up', 'hold', 'ramp_down'], _trap(D, p['speed_um_s'], p['ramp_up_s'], p['ramp_down_s'])))
        segs = [(n, d, a, b) for n, (d, a, b) in segs]
        if p['retract']:
            segs.append(('dwell', float(p['dwell_s']), 0.0, 0.0))
            back = _trap(D, p['retract_speed_um_s'], p['retract_ramp_up_s'], p['retract_ramp_down_s'])
            segs += [(n, d, -a, -b) for n, (d, a, b) in zip(['back_up', 'back_hold', 'back_down'], back)]
        v_parts, names = [], []
        for n, d, a, b in segs:
            k = int(round(d / dt))
            if k > 0:
                v_parts.append(np.linspace(a, b, k, endpoint=False))
                names += [n] * k
        self.v = np.concatenate(v_parts) if v_parts else np.zeros(1)
        self.names = names or ['hold']
        self.s = np.cumsum(self.v) * dt
        self.T = len(self.v) * dt

    def at(self, t):
        i = min(max(int(t / self.dt), 0), len(self.v) - 1)
        return float(self.s[i]), float(self.v[i]), self.names[i]


class ActionController:
    def __init__(self, p, upp):
        self.p, self.upp = p, upp
        self.ref = Reference(p)
        self.t_ref = 0.0
        self.t_total = 0.0
        self.integral = 0.0
        self.prev_pwm = 0.0
        self.done = False

    def step(self, s_meas_um, dt):
        """s_meas_um: displacement from the start point along the pull direction (um). Returns a dict."""
        p = self.p
        s_ref, v_ref, phase = self.ref.at(self.t_ref)
        e = s_ref - s_meas_um
        lagging = abs(e) > p['max_lag_um']
        if not lagging:
            self.t_ref = min(self.t_ref + dt, self.ref.T)          # reference clock only runs while we keep up
        self.t_total += dt
        lim = p['max_integral_um']
        self.integral = float(np.clip(self.integral + e * dt, -lim, lim))
        P, I = p['Kp_pos'] * e, p['Ki_pos'] * self.integral
        v_cmd = v_ref + P + I
        sign = 1.0 if v_cmd >= 0 else -1.0
        speed = abs(v_cmd)
        pwm = 0.0 if speed < p['min_cmd_um_s'] else p['phi_pwm'] + speed / p['G_um_s_per_pwm']
        pwm = min(max(pwm, 0.0), p['max_pwm'], self.prev_pwm + p['pwm_slew_per_s'] * dt)   # cap + ramp up, never jump
        self.prev_pwm = pwm
        finished = self.t_ref >= self.ref.T
        if finished and (abs(e) <= p['end_tol_um'] or self.t_total > self.ref.T + p['settle_timeout_s']):
            self.done = True
        return dict(sign=sign, pwm=pwm, v_cmd=v_cmd, v_ref=v_ref, s_ref=s_ref, e=e, P=P, I=I,
                    phase=phase if not lagging else phase + '(lag)', lagging=lagging, done=self.done)

    def hold(self):
        """Call instead of step() while the robot is not seen: motors off, reference clock stays paused."""
        self.prev_pwm = 0.0
