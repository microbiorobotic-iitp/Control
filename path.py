import numpy as np

import config as C


def advance_segment(bot_pos, path_pts, seg_idx, num_seg):
    if seg_idx + 1 < num_seg and np.linalg.norm(bot_pos - path_pts[seg_idx + 1]) <= C.ROA:
        return seg_idx + 1, True
    return seg_idx, False


def segment_geometry(bot_pos, seg_start, seg_end):
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec)
    u_seg = seg_vec / seg_len
    line_normal = np.array([-u_seg[1], u_seg[0]])
    cross_track_error = np.dot(bot_pos - seg_start, line_normal)
    proj = np.dot(bot_pos - seg_start, u_seg)
    proj = max(0.0, min(seg_len, proj))
    return u_seg, seg_len, cross_track_error, proj


def carrot_point(path_pts, seg_idx, num_seg, seg_start, seg_end, u_seg, seg_len, proj):
    ahead = proj + C.LOOKAHEAD
    if ahead >= seg_len and seg_idx + 1 < num_seg:
        remaining = ahead - seg_len
        next_vec = path_pts[seg_idx + 2] - seg_end
        next_u = next_vec / np.linalg.norm(next_vec)
        return seg_end + remaining * next_u
    return seg_start + min(ahead, seg_len) * u_seg