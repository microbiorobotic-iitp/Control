"""Tuning helpers that read the run folders written by navigate.py.

  python tune_tools.py report <run_dir> [<run_dir> ...]   health + quality metrics of one or more runs
  python tune_tools.py fit    <run_dir> <run_dir> ...     fit G and phi:  speed = G * (pwm - phi)
                              (use APPROACH runs made with Kp = Ki = 0 at 3-5 different cruise speeds)
"""
import csv
import json
import os
import sys

import numpy as np


def load(run):
    info = json.load(open(os.path.join(run, 'run_info.json')))
    rows = list(csv.DictReader(open(os.path.join(run, 'log.csv'))))
    col = lambda k: np.array([float(r[k]) if r.get(k, '') != '' else np.nan for r in rows])
    return info, rows, col


def pct(a, q):
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if a.size else float('nan')


def health(info, rows, col):
    kind = [r['kind'] for r in rows]
    n = len(rows)
    print(f"  frames {n}, dropped {info['dropped_frames']}, fps {info['measured_fps']:.1f}, "
          f"lost {100 * kind.count('lost') / n:.1f}%, partial {100 * kind.count('partial') / n:.1f}%, "
          f"tracker p95 {pct(col('tracker_ms'), 95):.2f} ms, loop median {np.nanmedian(col('loop_ms')):.1f} ms")


def report(run):
    info, rows, col = load(run)
    print(f"\n== {run}  mode={info['mode']}  fluid={info.get('fluid')}  {info['um_per_px']:.3f} um/px")
    health(info, rows, col)
    t = col('t_s')
    ph = np.array([r['phase'] for r in rows])
    if info['mode'] == 'approach':
        tgt, sp = col('target_speed_um_s'), col('speed_along_um_s')
        cruise = (tgt >= 0.98 * np.nanmax(tgt)) & (t > t[0] + 1.5)
        xt = np.abs(col('cross_track_um'))
        print(f"  cross-track um   : median {np.nanmedian(xt):.1f}  p95 {pct(xt, 95):.1f}  max {np.nanmax(xt):.1f}")
        if cruise.any():
            err = (np.nanmean(sp[cruise]) - np.nanmean(tgt[cruise])) / np.nanmean(tgt[cruise]) * 100
            print(f"  cruise speed um/s: mean {np.nanmean(sp[cruise]):.1f} (target {np.nanmean(tgt[cruise]):.1f}, "
                  f"{err:+.1f}%)  std {np.nanstd(sp[cruise]):.1f}   pwm {np.nanmedian(col('pwm')[cruise]):.4f}")
            print(f"  speed-loop term M: std {np.nanstd(col('M')[cruise]):.1f} um/s (noise passed to the motors)")
        print(f"  time {t[-1] - t[0]:.1f} s   final distance to goal {col('dist_goal_um')[-1]:.0f} um")
    else:
        e, sm, sr = col('e_pos_um'), col('s_meas_um'), col('s_ref_um')
        lag = np.mean([p.endswith('(lag)') for p in ph]) * 100
        print(f"  position error um: RMS {np.sqrt(np.nanmean(e ** 2)):.1f}  max {np.nanmax(np.abs(e)):.1f}   "
              f"time in 'lag' (reference paused) {lag:.1f}%")
        print(f"  s_max: ref {np.nanmax(sr):.0f}  measured {np.nanmax(sm):.0f} um   final position {sm[-1]:.1f} um   pwm max {np.nanmax(col('pwm')):.4f}")
        for name in ('hold', 'back_hold'):
            m = np.array([p.startswith(name) for p in ph])
            if m.any():
                v, vr = col('speed_along_um_s')[m], col('target_speed_um_s')[m]
                print(f"  {name:<9} speed um/s: mean {np.nanmean(v):.1f} (ref {np.nanmean(vr):.1f})  std {np.nanstd(v):.1f}")


def fit(runs):
    pts = []
    for run in runs:
        info, rows, col = load(run)
        if info['mode'] != 'approach':
            print(f'skip {run}: not an approach run')
            continue
        t, tgt = col('t_s'), col('target_speed_um_s')
        m = (tgt >= 0.98 * np.nanmax(tgt)) & (t > t[0] + 1.5) & np.array([r['kind'] == 'full' for r in rows])
        if m.sum() < 10:
            print(f'skip {run}: cruise phase too short')
            continue
        pts.append((float(np.nanmedian(col('pwm')[m])), float(np.nanmedian(col('speed_along_um_s')[m]))))
        print(f'  {run}: pwm {pts[-1][0]:.4f} -> speed {pts[-1][1]:.1f} um/s')
    if len(pts) < 2:
        print('need at least 2 usable runs at different cruise speeds')
        return
    pwm, v = np.array(pts).T
    G, b = np.polyfit(pwm, v, 1)
    phi = -b / G
    res = v - (G * pwm + b)
    print(f"\nfit: speed = G*(pwm - phi)   G = {G:.0f} um/s per PWM   phi = {phi:.4f}   max residual {np.abs(res).max():.1f} um/s")
    print('put these in the profile: "G_um_s_per_pwm": %.0f, "phi_pwm": %.4f' % (G, phi))


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('report', 'fit'):
        sys.exit(__doc__)
    if sys.argv[1] == 'fit':
        fit(sys.argv[2:])
    else:
        for r in sys.argv[2:]:
            report(r)
