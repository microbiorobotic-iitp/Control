"""Tiny stage timer. timer.lap('name') records the time since the previous lap/start."""
import time
from collections import defaultdict

import numpy as np

_ns = time.perf_counter_ns


class NullTimer:
    def start(self): pass
    def lap(self, name): pass
    def add(self, name, ns): pass


class Timer:
    ORDER = ["grab", "gray", "drift", "blur", "threshold", "contours", "select",
             "centroid", "update", "TRACKER_TOTAL", "log", "draw", "show", "FRAME_TOTAL"]

    def __init__(self):
        self.t = _ns()
        self.data = defaultdict(list)

    def start(self):
        self.t = _ns()

    def lap(self, name):
        n = _ns()
        self.data[name].append(n - self.t)
        self.t = n

    def add(self, name, ns):
        self.data[name].append(ns)

    def summary(self, skip=5):
        rows = [f"{'stage':<14}{'calls':>7}{'mean ms':>9}{'p50':>8}{'p95':>8}{'max':>8}"]
        names = [n for n in self.ORDER if n in self.data] + \
                [n for n in self.data if n not in self.ORDER]
        for n in names:
            v = np.array(self.data[n][skip:] or self.data[n], float) / 1e6
            rows.append(f"{n:<14}{len(v):>7}{v.mean():>9.3f}{np.percentile(v, 50):>8.3f}"
                        f"{np.percentile(v, 95):>8.3f}{v.max():>8.3f}")
        if "FRAME_TOTAL" in self.data:
            v = np.array(self.data["FRAME_TOTAL"][skip:] or self.data["FRAME_TOTAL"], float) / 1e9
            rows.append(f"=> gray-conversion + tracking alone could sustain {1 / v.mean():.0f} fps (FRAME_TOTAL = gray + tracker)")
        return "\n".join(rows)
