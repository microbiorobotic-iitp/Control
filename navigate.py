import time
import traceback
from datetime import datetime

import cv2
import numpy as np

import config as C
from control import VelocityPID, combine_commands, feedforward, steering_direction
from calib_model import heading_correction, local_params
from display import draw_scene, select_goals
from hardware import Arduino, Camera, currents
from path import advance_segment, carrot_point, segment_geometry
from storage import load_calibration, save_log
from vision import find_bots, to_gray, track_bot, set_tracker, get_tracker
from tracker import setup_tracker
from collections import deque     

WIN = 'Microrobot Control'
LOG_KEYS = ['t', 'x', 'y', 'target_speed', 'raw_speed', 'filtered_speed', 'error_v',
            'P_term', 'I_term', 'D_term', 'magnitude', 'cross_track', 'seg_idx', 'ff_x', 'ff_y','base', 'pwm_sent']


def run_navigation(cam, ard, calib, bot_pos, bot_r, raw_goals):
    path_pts = np.vstack([bot_pos, raw_goals])
    num_seg = len(path_pts) - 1
    final_goal = raw_goals[-1]
    virtual_start = bot_pos.copy()
    seg_idx = 0
    prev_bot_pos = bot_pos.copy()
    last_good_dir = np.zeros(2)
    pid = VelocityPID()
    log = {k: [] for k in LOG_KEYS}
    aborted = False

    prev_mag = 0.0

    model = calib['model'] if calib else None
    t_start = t_prev = time.perf_counter()
    get_tracker().reset_clock()
    trail = deque(maxlen=300)
    last_print = 0.0
    while not aborted:
        try:
            frame = cam.snapshot()
            centres, radii = track_bot(to_gray(frame))
            if len(centres):
                bot_pos = C.ALPHA_POS * centres[0] + (1 - C.ALPHA_POS) * bot_pos
                bot_r = radii[0]
            else:
                # robot not confirmed -> do NOT run control on a stale position
                if get_tracker().time_lost() > C.ABORT_LOST_S:
                    print('Robot lost - aborting run')
                    aborted = True
                    break
                if get_tracker().time_lost() > C.LOST_TIMEOUT:
                    currents(ard, [0, 0], 0)
                    prev_mag = 0.0
                    pid = VelocityPID()          # drop the old integral state
                t_prev = time.perf_counter()     # avoid a huge dt when it comes back
                prev_bot_pos = bot_pos.copy()
                if C.VISUALIZE:
                    cv2.imshow(WIN, draw_scene(frame, bot_pos, bot_r, raw_goals))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    aborted = True
                continue

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt <= 0:
                dt = 0.05

            dist_final = np.linalg.norm(final_goal - bot_pos)
            if dist_final <= C.ROA:
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
            vector_to_target[1] = -vector_to_target[1]

            raw_velocity = (bot_pos - prev_bot_pos) / dt
            prev_bot_pos = bot_pos.copy()
            forward_speed = np.dot(raw_velocity, u_seg)

            direction, last_good_dir = steering_direction(vector_to_target, last_good_dir)
            if model is not None and C.USE_HEADING_CORRECTION:
                direction = heading_correction(model, direction)
            lp = local_params(model, bot_pos) if model else dict(G=None, phi=0.0)

            if C.SIMPLE_TEST_PWM:
                pid_out = dict(target_speed=0.0, error_v=0.0, P=0.0, I=0.0, D=0.0, base=0.0,
                            magnitude=0.0, filtered_speed=0.0)
                ff_cmd = np.zeros(2)
                magnitude = C.SIMPLE_TEST_PWM          # direction stays the pure steering direction
            else:
                pid_out = pid.step(forward_speed, dist_to_waypoint, dist_final, dt, lp['phi'], lp['G'])
                ff_cmd = feedforward(bot_pos, calib)
                direction, magnitude = combine_commands(direction, pid_out['magnitude'], ff_cmd)

            magnitude = min(magnitude, C.MAX_PWM, prev_mag + C.PWM_SLEW * dt)   # ramp up, never jump
            pwm_sent = magnitude
            prev_mag = magnitude
            currents(ard, direction, magnitude)

            if time.perf_counter() - last_print > 0.25:
                last_print = time.perf_counter()
                ang = np.degrees(np.arctan2(direction[1], direction[0]))
                goal_ang = np.degrees(np.arctan2(vector_to_target[1], vector_to_target[0]))
                print(f"t={now - t_start:5.2f} dt={dt*1000:3.0f}ms pos=({bot_pos[0]:5.0f},{bot_pos[1]:5.0f}) "
                    f"speed={forward_speed:6.1f} pwm={magnitude:.4f} cmd_ang={ang:6.1f} goal_ang={goal_ang:6.1f} "
                    f"{get_tracker().last_kind}")

            vals = [time.perf_counter() - t_start, bot_pos[0], bot_pos[1], pid_out['target_speed'],
                                forward_speed, pid_out['filtered_speed'], pid_out['error_v'],
                                pid_out['P'], pid_out['I'], pid_out['D'], magnitude,
                                cross_track_error, seg_idx + 1, ff_cmd[0], ff_cmd[1], pid_out['base'], pwm_sent]

            for k, v in zip(LOG_KEYS, vals):
                log[k].append(v)

            trail.append(bot_pos.copy())
            if C.VISUALIZE:
                trk = get_tracker()
                frozen = np.linalg.norm(vector_to_target) <= C.FREEZE_RADIUS
                debug = dict(
                    path_pts=path_pts, trail=list(trail), seg_start=seg_start, seg_end=seg_end,
                    final_goal=final_goal, decel_centre=seg_end, carrot=carrot, frozen=frozen,
                    img_dir=np.array([direction[0], -direction[1]]),
                    hud=[f"seg {seg_idx+1}/{num_seg}  goal {dist_final:.0f}px  xtrack {cross_track_error:+.0f}px",
                        f"speed {forward_speed:.0f} (filt {pid_out['filtered_speed']:.0f}) / target {pid_out['target_speed']:.0f} px/s",
                        f"PWM sent {magnitude:.4f} = base {pid_out['base']:.4f} + trim + ff {np.linalg.norm(ff_cmd):.4f}",
                        f"{'FROZEN dir' if frozen else 'steering'}  {trk.last_kind}"])
                cv2.imshow(WIN, draw_scene(frame, bot_pos, bot_r, raw_goals, debug, trk.last_contour))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                aborted = True

        except Exception as e:
            traceback.print_exc()
            break

    return {k: np.array(v) for k, v in log.items()}, path_pts, aborted


def main():
    exp_name = input('Experiment name (no spaces): ')

    print('Connecting to Arduino...')
    ard = Arduino(C.ARDUINO_PORT, C.PINS)

    print('Initializing Camera...')
    cam = Camera()
    cam.warmup()

    cv2.namedWindow(WIN)
    tracker = setup_tracker(cam, WIN, C.TRACK_LEARN_FRAMES, C.TRACK_MAX_SPEED, C.TRACK_DARK_MARGIN)
    if tracker is None:
        cam.release(); cv2.destroyAllWindows(); return
    set_tracker(tracker)

    frame = cam.snapshot()
    bot_pos = tracker.pos.copy()
    bot_r = tracker.R
    raw_goals = select_goals(WIN, frame, bot_pos, bot_r, tracker.last_contour)

    if len(raw_goals) == 0:
        print('No goals selected. Exiting...')
        cam.release()
        cv2.destroyAllWindows()
        return

    calib = load_calibration()
    log, path_pts, aborted = run_navigation(cam, ard, calib, bot_pos, bot_r, raw_goals)

    print('Shutting down safely...')
    try:
        currents(ard, [0, 0], 0)
    except Exception:
        pass
    try:
        cam.release()
        cv2.destroyAllWindows()
    except Exception:
        pass
    print('Goal reached. Stopping robot...')

    meta = dict(Kp_v=C.KP_V, Ki_v=C.KI_V, Kd_v=C.KD_V, cruise_speed=C.CRUISE_SPEED,
                lookahead=C.LOOKAHEAD, decel_radius=C.DECEL_RADIUS, roa=C.ROA,
                raw_goals=raw_goals, path_pts=path_pts,
                timestamp=datetime.now().strftime('%d-%b-%Y %H:%M:%S'), aborted=aborted)
    save_log(log, meta, exp_name)


if __name__ == '__main__':
    main()