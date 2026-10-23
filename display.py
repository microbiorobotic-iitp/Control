import cv2
import numpy as np

import config as C

RED, BLUE, GREEN = (0, 0, 255), (255, 0, 0), (0, 255, 0)
MAGENTA, YELLOW, CYAN, BLACK = (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 0, 0)
EDGE_COL  = (0, 255, 0)      # same green as the outline shown when you confirm the robot
PATH_COL  = (255, 150, 0)    # planned path (BGR light blue)
SEG_COL   = (0, 165, 255)    # active approach line (orange)
TRAIL_COL = (180, 0, 180)    # where the bot actually went (purple)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def dashed_circle(img, centre, radius, color, step=20):
    for a in range(0, 360, step * 2):
        cv2.ellipse(img, _pt(centre), (int(radius), int(radius)), 0, a, a + step, color, 1)


def _poly(pts):
    return np.round(np.asarray(pts, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
 
 
def draw_scene(frame, bot_pos, bot_r, goals=None, debug=None, contour=None):
    img = frame.copy()
    cv2.line(img, (540, 430), (582, 430), BLACK, 3)
    cv2.putText(img, '200 microns', (505, 455), FONT, 0.6, BLACK, 1)
    d = debug or {}
 
    # planned path through all goals + where the bot has actually been
    pp = d.get('path_pts')
    if pp is not None and len(pp) > 1:
        cv2.polylines(img, [_poly(pp)], False, PATH_COL, 1, cv2.LINE_AA)
    tr = d.get('trail')
    if tr is not None and len(tr) > 1:
        cv2.polylines(img, [_poly(tr)], False, TRAIL_COL, 1, cv2.LINE_AA)
 
    # active approach line (the line the cross-track error is measured against)
    if 'seg_start' in d and 'seg_end' in d:
        cv2.line(img, _pt(d['seg_start']), _pt(d['seg_end']), SEG_COL, 2, cv2.LINE_AA)
 
    # robot: full edge (fallback to a circle only if no contour is available)
    if d.get('lost'):
        cv2.putText(img, 'ROBOT NOT FOUND', (10, 20), FONT, 0.7, RED, 2)
    elif contour is not None:
        cv2.drawContours(img, [contour], -1, EDGE_COL, 1)
        cv2.circle(img, _pt(bot_pos), 2, EDGE_COL, -1)
    else:
        cv2.circle(img, _pt(bot_pos), int(bot_r), RED, 2)
 
    if goals is not None:
        for g in goals:
            cv2.drawMarker(img, _pt(g), BLUE, cv2.MARKER_STAR, 14, 2)
 
    if 'carrot' in d:
        cv2.line(img, _pt(bot_pos), _pt(d['carrot']), CYAN, 1, cv2.LINE_AA)   # bot -> carrot
        cv2.circle(img, _pt(d['carrot']), 6, CYAN, 2)
    if 'img_dir' in d:
        end = bot_pos + 40 * d['img_dir']
        cv2.arrowedLine(img, _pt(bot_pos), _pt(end), YELLOW, 2, tipLength=0.3)
 
    # rings: decel (around the current waypoint), freeze + stop (around the final goal)
    if 'decel_centre' in d:
        dashed_circle(img, d['decel_centre'], C.DECEL_RADIUS, GREEN)
    fg = d.get('final_goal')
    if fg is not None:
        dashed_circle(img, fg, C.ROA, RED)
        if d.get('frozen'):
            cv2.circle(img, _pt(fg), int(C.FREEZE_RADIUS), MAGENTA, 2)      # solid = steering frozen
        else:
            dashed_circle(img, fg, C.FREEZE_RADIUS, MAGENTA)
 
    for i, s in enumerate(d.get('hud', [])):
        cv2.putText(img, s, (10, 20 + 18 * i), FONT, 0.5, BLACK, 1, cv2.LINE_AA)
    return img

def select_goals(win, frame, bot_pos, bot_r, contour=None):
    """Click targets sequentially; press ENTER when done."""
    goals = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            goals.append(np.array([x, y], dtype=float))

    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = draw_scene(frame, bot_pos, bot_r, goals, contour=contour)
        cv2.putText(img, 'Click targets sequentially. Press ENTER when done.', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 1)
        cv2.imshow(win, img)
        if cv2.waitKey(30) & 0xFF in (13, 10):
            break
    cv2.setMouseCallback(win, lambda *a: None)
    return np.array(goals).reshape(-1, 2)