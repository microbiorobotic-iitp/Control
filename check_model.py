"""
Run:  python check_model.py
Shows, per calibrated direction, how much PWM the bot NEEDS (restoring force R(s) + friction f)
versus the most PWM your controller settings can actually deliver, and where it would stall.
"""
import numpy as np

import config as C
from storage import load_calibration

calib = load_calibration()
model = calib['model']
S = [0, 20, 40, 60, 80, 100, 150, 200, 300]

print(f"MAX_PWM={C.MAX_PWM}  FF: gain {C.FF_GAIN_SCALE}, cap {C.FF_MAX_MAG}  "
      f"TRIM_MAX={C.TRIM_MAX}  MAX_EXCESS_PWM={C.MAX_EXCESS_PWM}  target={C.CRUISE_SPEED}px/s")
worst = 0.0
for name, x in model['rays'].items():
    k = x['k']
    excess = min(C.CRUISE_SPEED / k, C.MAX_EXCESS_PWM) if np.isfinite(k) and k > 0 else 0.0
    print(f"\n{name}: friction f={x['f']:.4f}  k={k:.0f} px/s per PWM  "
          f"bias={np.degrees(x['bias']):.0f}deg  calibrated out to s={x['s'].max():.0f}px")
    print("   s(px)    R(s)   needed(R+f)   max deliverable   ok?")
    stall = None
    for s in S:
        if s > x['s'].max():
            break
        R = float(np.interp(s, x['s'], x['R']))
        needed = R + x['f']
        ff = min(C.FF_GAIN_SCALE * R, C.FF_MAX_MAG)
        avail = min(C.MAX_PWM, ff + x['f'] + excess + C.TRIM_MAX)   # along the ray, best case
        ok = avail >= needed
        if not ok and stall is None:
            stall = s
        worst = max(worst, needed)
        print(f"  {s:5d}  {R:7.4f}   {needed:9.4f}   {avail:13.4f}      {'ok' if ok else 'STALLS'}")
    print(f"   -> {'can reach the whole calibrated range' if stall is None else f'stalls around s ~ {stall} px from home'}")

print(f"\nLargest PWM needed anywhere: {worst:.4f}.  MAX_PWM should be at least ~{1.2 * worst:.3f}.")