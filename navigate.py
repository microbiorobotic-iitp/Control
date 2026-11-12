"""
navigate.py - approach / action modes with the FastTracker, per-fluid profiles and frame-synchronised recording.

Flow
----
  start -> pick fluid profile -> pick robot / mask / magnification -> click goals -> MENU
  MENU keys:  P approach (go to the goals)   A action (pull the cell)   G new goals
              R re-pick robot / magnification / mask (approach side only)   F reload the profile JSON   Q quit
  During an APPROACH run:  R = stop motors and reconfigure (magnification, robot, mask, goals), then continue
                           Q = stop and go back to the menu
  During an ACTION run:    only Q (stop). Magnification / tracker keys are deliberately disabled.

Recording (one folder per run, see recording.py): video.avi is the RAW camera frame (nothing drawn on it);
log.csv has one row per video frame (row i <-> frame i) with position, velocity, controller terms and duties.
Overlays are drawn only on the live window.
"""
import os
import time
import traceback
from collections import deque
from dataclasses import asdict
from datetime import datetime

import cv2
import numpy as np

import config as C
import display
import profiles
from action import ActionController
from display import draw_scene, select_goals
from hardware import Arduino, Camera, currents
from path import advance_segment, carrot_point, segment_geometry
from control import VelocityPID, steering_direction, axis_limit
from recording import DUTY_COLUMNS, Experiment, Recorder, duties_from_vector, duty_columns
from storage import save_log
from tracker_config import Cfg, Scale
from tracker_setup import robot_geometry, setup_tracker
from vision import get_tracker, set_tracker, to_gray, track

WIN = 'Microrobot Control'
HOLD = dict(direction=np.zeros(2), magnitude=0.0)    # last duty sent in approach mode (kept during a reconfigure)
LOG_KEYS = ['t', 'x', 'y', 'target_speed', 'raw_speed', 'filtered_speed', 'error_v',
            'P_term', 'I_term', 'D_term', 'magnitude', 'cross_track', 'seg_idx', 'ff_x', 'ff_y', 'base', 'pwm_sent']

# one row per video frame in log.csv ('frame', 't_s', 't_wall' are added by the Recorder)
COLUMNS = (['mode', 'phase', 'kind', 'um_per_px', 'x_px', 'y_px', 'x_um', 'y_um', 'R_um',
            'vx_um_s', 'vy_um_s', 'speed_um_s', 'vrawx_um_s', 'vrawy_um_s', 'drift',
            'target_speed_um_s', 'speed_along_um_s', 'filtered_speed_um_s', 'error_v', 'P', 'I', 'D', 'M', 'base',
            'pwm', 'dir_x', 'dir_y', 'cross_track_um', 'dist_goal_um', 'seg',
            's_ref_um', 's_meas_um', 'e_pos_um', 'tracker_ms', 'loop_ms'] + DUTY_COLUMNS)


# ----------------------------------------------------------------- small helpers
def grab(cam):
    """-> raw frame, its timestamp, tracker result (or None), tracker time in ms."""
    frame, t = cam.snapshot_t()
    t0 = time.perf_counter()
    res = track(to_gray(frame), t)
    return frame, t, res, (time.perf_counter() - t0) * 1000.0


def base_row(res, mode, upp, tracker_ms, loop_ms, phase=''):
    row = dict(mode=mode, phase=phase, um_per_px=upp, tracker_ms=round(tracker_ms, 3), loop_ms=round(loop_ms, 2))
    if res is None:
        row['kind'] = 'lost'
        return row
    c, cu, v, vr = res['center'], res['center_um'], res['vel_um_s'], res['vel_raw_um_s']
    row.update(kind=res['kind'], x_px=c[0], y_px=c[1], x_um=cu[0], y_um=cu[1], R_um=res['R_um'],
               vx_um_s=v[0], vy_um_s=v[1], speed_um_s=res['speed_um_s'], vrawx_um_s=vr[0], vrawy_um_s=vr[1],
               drift=res['drift'])
    return row


def motors_off(ard):
    try:
        currents(ard, [0, 0], 0)
    except Exception:
        traceback.print_exc()


def start_run(exp, mode, profile, scale, extra):
    """Make the run folder + recorder; also save a reference frame and the mask for later analysis."""
    run_dir = exp.new_run_dir()
    trk = get_tracker()
    info = dict(mode=mode, fluid=profile['fluid'], profile=profile['name'], um_per_px=scale.um_per_px,
                magnification=scale.key, magnification_verified=scale.verified,
                robot_diameter_um=2 * trk.R * scale.um_per_px, settings=profile[mode], **extra)
    rec = Recorder(run_dir, COLUMNS, fps=C.RECORD_FPS, codec=getattr(C, 'RECORD_CODEC', 'MJPG'), info=info)
    m = trk.exclusion_total()
    if m is not None:
        cv2.imwrite(os.path.join(run_dir, 'mask.png'), m)
    return run_dir, rec


def wait_for_robot(cam, timeout=2.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        frame, t, res, ms = grab(cam)
        if res is not None:
            return frame, t, res, ms
    return None, None, None, None


# ----------------------------------------------------------------- APPROACH mode (your PID, in px at the current mag)
def run_approach(cam, ard, exp, profile, scale, raw_goals, win, resume=False):
    """Returns (log, path_pts, status) with status in 'reached' | 'aborted' | 'reconfigure'.
    resume=True: continue after a reconfigure WITHOUT dropping the duty to zero (PWM slew starts from the held value)."""
    upp = scale.um_per_px
    trk = get_tracker()
    frame, t, res, _ = wait_for_robot(cam)
    if res is None:
        print('Robot not found - cannot start')
        return {}, np.empty((0, 2)), 'aborted'
    bot_pos, bot_r = res['center'].copy(), res['R']

    path_pts = np.vstack([bot_pos, raw_goals])
    num_seg = len(path_pts) - 1
    final_goal = raw_goals[-1]
    virtual_start = bot_pos.copy()
    seg_idx = 0
    prev_bot_pos = bot_pos.copy()
    last_good_dir = np.zeros(2)
    log = {k: [] for k in LOG_KEYS}
    status = 'aborted'
    prev_mag = HOLD['magnitude'] if resume else 0.0
    pid = VelocityPID(profile['approach'], u0=prev_mag) 
    trail = deque(maxlen=300)
    last_print = 0.0

    run_dir, rec = start_run(exp, 'approach', profile, scale, dict(goals_px=raw_goals, start_px=bot_pos))
    cv2.imwrite(os.path.join(run_dir, 'start_frame.png'), frame)
    rec.event('approach start')
    t_start = t_prev = t_loop = time.perf_counter()
    trk.reset_clock()

    while True:
        try:
            frame, t, res, trk_ms = grab(cam)
            loop_ms = (time.perf_counter() - t_loop) * 1000.0
            t_loop = time.perf_counter()
            key = -1

            if res is None:
                # robot not confirmed -> do NOT run control on a stale position
                lost_for = trk.time_lost()
                rec.log(frame, t=t, **base_row(None, 'approach', upp, trk_ms, loop_ms, 'lost'),
                        **duty_columns(duties_from_vector([0, 0], 0)))
                if lost_for > C.ABORT_LOST_S:
                    print('Robot lost - aborting run')
                    rec.event('robot lost - abort')
                    break
                if lost_for > C.LOST_TIMEOUT:
                    currents(ard, [0, 0], 0)
                    prev_mag = 0.0
                    HOLD['magnitude'] = 0.0
                    pid = VelocityPID(profile['approach'], u0=prev_mag)              # drop the old integral state
                t_prev = time.perf_counter()         # avoid a huge dt when it comes back
                prev_bot_pos = bot_pos.copy()
                if C.VISUALIZE:
                    cv2.imshow(win, draw_scene(frame, bot_pos, bot_r, raw_goals, dict(lost=True), trk.last_contour))
                key = cv2.waitKey(1) & 0xFF
            else:
                bot_pos = C.ALPHA_POS * res['center'] + (1 - C.ALPHA_POS) * bot_pos
                bot_r = res['R']
                now = time.perf_counter()
                dt = now - t_prev
                t_prev = now
                if dt <= 0:
                    dt = 0.05

                dist_final = np.linalg.norm(final_goal - bot_pos)
                if dist_final <= C.ROA:
                    rec.log(frame, t=t, **base_row(res, 'approach', upp, trk_ms, loop_ms, 'goal'),
                            dist_goal_um=dist_final * upp, **duty_columns(duties_from_vector([0, 0], 0)))
                    rec.event('goal reached')
                    status = 'reached'
                    break

                seg_idx, advanced = advance_segment(bot_pos, path_pts, seg_idx, num_seg)
                if advanced:
                    virtual_start = bot_pos.copy()
                virtual_start = virtual_start + (dt / C.LINE_TIME_CONST) * (bot_pos - virtual_start)
                seg_start = virtual_start
                seg_end = path_pts[seg_idx + 1]
                u_seg, seg_len, cross_track_error, proj = segment_geometry(bot_pos, seg_start, seg_end)
                dist_to_waypoint = np.linalg.norm(seg_end - bot_pos)

                carrot = carrot_point(path_pts, seg_idx, num_seg, seg_start, seg_end, u_seg, seg_len, proj)
                vector_to_target = carrot - bot_pos
                vector_to_target[1] = -vector_to_target[1]           # image y is down, coil y is up

                raw_velocity = (bot_pos - prev_bot_pos) / dt
                prev_bot_pos = bot_pos.copy()
                forward_speed = np.dot(raw_velocity, u_seg)

                direction, last_good_dir = steering_direction(vector_to_target, last_good_dir)
                u_max = axis_limit(direction, C.MAX_PWM)
                pid_out = pid.step(forward_speed, dist_to_waypoint, dist_final, dt, u_max)
                slew = profile['approach']['pwm_slew_per_s']
                magnitude = min(pid_out['magnitude'], u_max, prev_mag + slew * dt)
                prev_mag = magnitude
                currents(ard, direction, magnitude)
                HOLD['direction'], HOLD['magnitude'] = np.array(direction, float), float(magnitude)

                if time.perf_counter() - last_print > 0.25:
                    last_print = time.perf_counter()
                    ang = np.degrees(np.arctan2(direction[1], direction[0]))
                    goal_ang = np.degrees(np.arctan2(vector_to_target[1], vector_to_target[0]))
                    print(f"t={now - t_start:5.2f} dt={dt * 1000:3.0f}ms pos=({bot_pos[0]:5.0f},{bot_pos[1]:5.0f}) "
                          f"speed={forward_speed * upp:6.1f}um/s pwm={magnitude:.4f} cmd_ang={ang:6.1f} "
                          f"goal_ang={goal_ang:6.1f} {trk.last_kind}")

                vals = [now - t_start, bot_pos[0], bot_pos[1], pid_out['target_speed'], forward_speed,
                        pid_out['filtered_speed'], pid_out['error_v'], pid_out['P'], pid_out['I'], pid_out['D'],
                        magnitude, cross_track_error, seg_idx + 1, 0.0, 0.0, pid_out['base'], magnitude]
                for k, v in zip(LOG_KEYS, vals):
                    log[k].append(v)

                row = base_row(res, 'approach', upp, trk_ms, loop_ms, 'approach')
                row.update(target_speed_um_s=pid_out['target_speed'] * upp, speed_along_um_s=forward_speed * upp,
           filtered_speed_um_s=pid_out['filtered_speed'] * upp, error_v=pid_out['error_v'] * upp,
           P=pid_out['P'], I=pid_out['I'], D=pid_out['D'], M=pid_out['M'], base=pid_out['base'],
           pwm=magnitude, dir_x=direction[0], dir_y=direction[1],
                           cross_track_um=cross_track_error * upp, dist_goal_um=dist_final * upp, seg=seg_idx + 1)
                row.update(duty_columns(duties_from_vector(direction, magnitude)))
                rec.log(frame, t=t, **row)                          # RAW frame: nothing is drawn on it

                trail.append(bot_pos.copy())
                if C.VISUALIZE:
                    frozen = np.linalg.norm(vector_to_target) <= C.FREEZE_RADIUS
                    debug = dict(
                        path_pts=path_pts, trail=list(trail), seg_start=seg_start, seg_end=seg_end,
                        final_goal=final_goal, decel_centre=seg_end, carrot=carrot, frozen=frozen,
                        img_dir=np.array([direction[0], -direction[1]]),
                        hud=[f"APPROACH seg {seg_idx + 1}/{num_seg}  goal {dist_final * upp:.0f}um  xtrack {cross_track_error * upp:+.0f}um",
                             f"speed {forward_speed * upp:.0f} (filt {pid_out['filtered_speed'] * upp:.0f}) / target {pid_out['target_speed'] * upp:.0f} um/s",
                             f"PWM {magnitude:.4f} (base {pid_out['base']:.4f})  {scale.key}x {upp:.2f}um/px   [R] reconfigure  [Q] stop",
                             f"{'FROZEN dir' if frozen else 'steering'}  {trk.last_kind}  rec backlog {rec.backlog}"])
                    cv2.imshow(win, draw_scene(frame, bot_pos, bot_r, raw_goals, debug, trk.last_contour))
                key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                rec.event('stopped by user')
                break
            if key == ord('r'):                       # approach mode only: stop and reconfigure
                rec.event('reconfigure requested')
                status = 'reconfigure'
                break
        except Exception:
            traceback.print_exc()
            rec.event('exception')
            break

    if status == 'reconfigure':
        # do NOT zero the duty: the meniscus would pull the robot away while magnification / robot are re-selected
        HOLD['magnitude'] *= profile['approach']['reconfigure_hold_scale']
        currents(ard, HOLD['direction'], HOLD['magnitude'])
        rec.event(f"holding duty {HOLD['magnitude']:.4f} during reconfigure")
        print(f"[hold] keeping the last duty ({HOLD['magnitude']:.4f}) while you reconfigure")
    else:
        motors_off(ard)
        HOLD['magnitude'] = 0.0
    summary = rec.close()
    print(f"[recording] {run_dir}: {summary['n_frames']} frames, dropped {summary['dropped_frames']}, "
          f"{summary['measured_fps']} fps")
    return {k: np.array(v) for k, v in log.items()}, path_pts, status


# ----------------------------------------------------------------- ACTION mode (pull the cell along a fixed direction)
def select_direction(win, cam):
    """Click ONE point: the cell is pulled from the robot towards it. ENTER confirms, Q cancels.
    Returns unit vector in IMAGE coordinates (y down) or None."""
    pt = {'p': None}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pt['p'] = np.array([x, y], float)

    cv2.setMouseCallback(win, on_mouse)
    u = None
    while True:
        frame, t, res, _ = grab(cam)
        img = frame.copy()
        if res is not None:
            cv2.polylines(img, [res['contour']], True, (0, 255, 0), 1)
            if pt['p'] is not None:
                cv2.arrowedLine(img, tuple(int(v) for v in res['center']), tuple(int(v) for v in pt['p']),
                                (0, 255, 255), 2, tipLength=0.05)
        cv2.putText(img, 'ACTION: click the pull direction, ENTER to confirm, Q to cancel', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k in (13, 10) and pt['p'] is not None and res is not None:
            d = pt['p'] - res['center']
            if np.linalg.norm(d) > 3:
                u = d / np.linalg.norm(d)
                break
    cv2.setMouseCallback(win, lambda *a: None)
    return u


def run_action(cam, ard, exp, profile, scale, win):
    """Locked mode: no magnification / tracker changes. Returns status 'done' | 'aborted' | 'cancelled'."""
    upp = scale.um_per_px
    p = profile['action']
    trk = get_tracker()
    u_img = select_direction(win, cam)
    if u_img is None:
        return 'cancelled'
    u_coil = np.array([u_img[0], -u_img[1]])              # coil axes: y is up

    u_max = p['max_pwm'] / max(abs(u_coil[0]), abs(u_coil[1]))

    # start point = average of a few detections (less noise than a single frame)
    pts, frame = [], None
    t0 = time.perf_counter()
    while len(pts) < 8 and time.perf_counter() - t0 < 3.0:
        frame, t, res, _ = grab(cam)
        if res is not None:
            pts.append(res['center'])
    if len(pts) < 8:
        print('Robot not found - cannot start action')
        return 'aborted'
    pos0 = np.mean(pts, axis=0)

    ctl = ActionController(p, upp)
    run_dir, rec = start_run(exp, 'action', profile, scale,
                             dict(pull_dir_img=u_img, start_px=pos0, planned_duration_s=ctl.ref.T))
    cv2.imwrite(os.path.join(run_dir, 'start_frame.png'), frame)
    rec.event('action start')
    print(f"Action: {p['distance_um']:.0f} um at {p['speed_um_s']:.0f} um/s, planned {ctl.ref.T:.1f} s. Q = stop.")
    status = 'aborted'
    t_prev = t_loop = time.perf_counter()
    trk.reset_clock()
    last_print = 0.0
    trail = deque(maxlen=600)

    while True:
        try:
            frame, t, res, trk_ms = grab(cam)
            loop_ms = (time.perf_counter() - t_loop) * 1000.0
            t_loop = time.perf_counter()
            now = time.perf_counter()
            dt = max(now - t_prev, 1e-3)
            t_prev = now
            hud = []

            if res is None:
                lost_for = trk.time_lost()
                ctl.hold()
                row = base_row(None, 'action', upp, trk_ms, loop_ms, 'lost')
                row.update(duty_columns(duties_from_vector([0, 0], 0)))
                rec.log(frame, t=t, **row)
                if lost_for > C.LOST_TIMEOUT:
                    currents(ard, [0, 0], 0)
                if lost_for > p['abort_lost_s']:
                    print('Robot lost - aborting action')
                    rec.event('robot lost - abort')
                    break
                hud = ['ROBOT NOT FOUND - motors off, reference paused']
            else:
                s_meas = float(np.dot(res['center'] - pos0, u_img)) * upp           # um along the pull direction
                out = ctl.step(s_meas, dt, u_max) 
                direction = out['sign'] * u_coil
                currents(ard, direction, out['pwm'])

                v_along = float(np.dot(res['vel_um_s'], u_img))
                row = base_row(res, 'action', upp, trk_ms, loop_ms, out['phase'])
                row.update(target_speed_um_s=out['v_ref'], speed_along_um_s=v_along, filtered_speed_um_s=v_along,
                           error_v=out['v_ref'] - v_along, P=out['P'], I=out['I'], M=out['P'] + out['I'], pwm=out['pwm'],
                           dir_x=direction[0], dir_y=direction[1], s_ref_um=out['s_ref'], s_meas_um=s_meas,
                           e_pos_um=out['e'])
                row.update(duty_columns(duties_from_vector(direction, out['pwm'])))
                rec.log(frame, t=t, **row)                                          # RAW frame
                trail.append(res['center'].copy())

                if time.perf_counter() - last_print > 0.25:
                    last_print = time.perf_counter()
                    print(f"t={ctl.t_total:5.2f} {out['phase']:<14} s_ref={out['s_ref']:7.1f} s={s_meas:7.1f} um "
                          f"e={out['e']:6.1f} v={v_along:6.1f} um/s pwm={out['pwm']:.4f}")
                hud = [f"ACTION {out['phase']}   t {ctl.t_ref:.1f}/{ctl.ref.T:.1f} s   [Q] stop (tracker keys locked)",
                       f"s_ref {out['s_ref']:.0f}  s {s_meas:.0f}  err {out['e']:+.1f} um   speed {v_along:.0f} (ref {out['v_ref']:.0f}) um/s",
                       f"PWM {out['pwm']:.4f}  P {out['P']:+.3f} I {out['I']:+.3f} (PWM)   rec backlog {rec.backlog}"]
                if out['done']:
                    rec.event('action done')
                    status = 'done'
                    break

            if C.VISUALIZE:
                img = frame.copy()
                if res is not None:
                    cv2.polylines(img, [res['contour']], True, (0, 255, 0), 1)
                    if len(trail) > 1:
                        cv2.polylines(img, [np.round(np.array(trail)).astype(np.int32).reshape(-1, 1, 2)], False,
                                      (180, 0, 180), 1, cv2.LINE_AA)
                    ref_pt = pos0 + u_img * out['s_ref'] / upp
                    cv2.line(img, tuple(int(v) for v in pos0), tuple(int(v) for v in pos0 + u_img * p['distance_um'] / upp),
                             (255, 150, 0), 1, cv2.LINE_AA)
                    cv2.circle(img, tuple(int(v) for v in ref_pt), 5, (255, 255, 0), 2)     # where the robot SHOULD be
                for i, s in enumerate(hud):
                    cv2.putText(img, s, (10, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow(win, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):        # the ONLY key in action mode
                rec.event('stopped by user')
                break
        except Exception:
            traceback.print_exc()
            rec.event('exception')
            break

    motors_off(ard)
    summary = rec.close()
    print(f"[recording] {run_dir}: {summary['n_frames']} frames, dropped {summary['dropped_frames']}, "
          f"{summary['measured_fps']} fps")
    return status


# ----------------------------------------------------------------- menu + session
MENU_LINES = ["MENU   [P] approach   [A] action   [G] new goals   [R] robot / magnification / mask   "
              "[F] reload profile   [Q] quit"]


def menu(cam, win, goals, scale, profile, note=''):
    while True:
        frame, t, res, _ = grab(cam)
        trk = get_tracker()
        pos = res['center'] if res is not None else trk.pos
        hud = MENU_LINES + [f"{scale.key}x = {scale.um_per_px:.3f} um/px | fluid {profile['fluid']} | "
                            f"{'robot OK' if res is not None else 'ROBOT NOT FOUND'}", note]
        cv2.imshow(win, draw_scene(frame, pos, trk.R, goals if goals is not None and len(goals) else None,
                                   dict(hud=hud, lost=res is None), trk.last_contour))
        k = cv2.waitKey(1) & 0xFF
        if k != 255 and chr(k).lower() in 'pagrfq':
            return chr(k).lower()


def reconfigure(cam, scale, tcfg, profile, ard=None, hold=False):
    """Magnification / robot / mask. Returns the tracker (or None if the user quit the set-up).
    hold=True: keep re-sending the held duty every frame while the user works (duty never drops to 0)."""
    tick = (lambda: currents(ard, HOLD['direction'], HOLD['magnitude'])) if hold else None
    tracker = setup_tracker(cam, WIN, scale, tcfg, get_tracker(), tick=tick)
    if tracker is not None:
        set_tracker(tracker)
        profiles.apply_scale(profile, scale.um_per_px)
        display.set_scale(scale.um_per_px)
    return tracker


def ask_goals(cam):
    frame, t, res, _ = wait_for_robot(cam)
    trk = get_tracker()
    pos = res['center'] if res is not None else trk.pos
    return select_goals(WIN, frame if frame is not None else cam.snapshot(), pos, trk.R, trk.last_contour)


def main():
    exp_name = input('Experiment name (no spaces): ').strip()
    names = profiles.list_profiles()
    print('Fluid profiles:', names)
    fluid = input(f'Fluid profile [{names[0] if names else "?"}]: ').strip() or names[0]
    profile = profiles.load_profile(fluid)
    scale = Scale(key=(input('Magnification key [default from magnification.json]: ').strip() or None))
    scale.check_resolution(*C.FRAME_SIZE)

    print('Connecting to Arduino...')
    ard = Arduino(C.ARDUINO_PORT, C.PINS)
    print('Initializing Camera...')
    cam = Camera()
    cam.warmup()
    cv2.namedWindow(WIN)

    tcfg = Cfg()
    tcfg.max_speed_um_s = profile['tracker_max_speed_um_s']
    tracker = None
    try:
        tracker = setup_tracker(cam, WIN, scale, tcfg)
        if tracker is None:
            return
        set_tracker(tracker)
        profiles.apply_scale(profile, scale.um_per_px)
        display.set_scale(scale.um_per_px)
        print(profiles.summary(profile, scale.um_per_px))

        exp = Experiment(exp_name, answers=dict(fluid=fluid), geometry=robot_geometry(tracker),
                         settings=dict(profile=profile, tracker=asdict(tcfg), magnification=scale.key,
                                       um_per_px=scale.um_per_px),
                         root=getattr(C, 'EXPERIMENT_DIR', 'experiments'))
        goals = ask_goals(cam)
        note = ''
        choice = 'p' if len(goals) else None
        resume = False
        while True:
            if choice is None:
                choice = menu(cam, WIN, goals, scale, profile, note)
            note = ''
            if choice == 'q':
                break
            elif choice == 'f':
                profile = profiles.load_profile(fluid)
                tcfg.max_speed_um_s = profile['tracker_max_speed_um_s']
                profiles.apply_scale(profile, scale.um_per_px)
                exp.update_meta(profile=profile)
                print('Reloaded profile.\n' + profiles.summary(profile, scale.um_per_px))
                note = 'profile reloaded'
            elif choice == 'g':
                goals = ask_goals(cam)
            elif choice == 'r':
                if reconfigure(cam, scale, tcfg, profile) is None:
                    break
                exp.update_meta(robot_geometry=robot_geometry(get_tracker()), magnification=scale.key,
                                um_per_px=scale.um_per_px)
                goals = ask_goals(cam)                        # old goals are meaningless after a zoom / move
            elif choice == 'p':
                if goals is None or len(goals) == 0:
                    goals = ask_goals(cam)
                    if len(goals) == 0:
                        choice = None
                        continue
                log, path_pts, status = run_approach(cam, ard, exp, profile, scale, goals, WIN, resume=resume)
                resume = False
                if len(log.get('t', [])):
                    save_log(log, dict(profile=profile, um_per_px=scale.um_per_px, magnification=scale.key,
                                       raw_goals=goals, path_pts=path_pts, status=status,
                                       timestamp=datetime.now().strftime('%d-%b-%Y %H:%M:%S')), exp_name)
                if status == 'reconfigure':                   # allowed in approach mode: reconfigure, then go on
                    if reconfigure(cam, scale, tcfg, profile, ard, hold=True) is None:
                        break                                 # (finally: motors off)
                    exp.update_meta(robot_geometry=robot_geometry(get_tracker()), magnification=scale.key,
                                    um_per_px=scale.um_per_px)
                    goals = ask_goals(cam)
                    if len(goals):
                        choice, resume = 'p', True            # continue from the held duty
                    else:
                        motors_off(ard)                       # nothing to continue: do not keep pushing
                        HOLD['magnitude'] = 0.0
                        choice = None
                    continue
                note = f'approach {status}'
            elif choice == 'a':
                status = run_action(cam, ard, exp, profile, scale, WIN)
                note = f'action {status}'
            choice = None
    finally:
        print('Shutting down safely...')
        motors_off(ard)
        try:
            cam.release()
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()