"""All tunables in one place. Lengths are in micrometres, speeds in um/s, time in s."""
import json
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Cfg:
    # ---- physics / motion
    max_speed_um_s: float = 5000.0     # fastest the robot can possibly go (sets search window)
    # ---- segmentation
    blur_k: int = 3                    # Gaussian blur on the search window (odd; 0 = off)
    thr_frac: float = 0.8              # threshold = this fraction of the way from local bg to robot darkness
    min_margin: float = 4.0            # threshold is always >= this many grey levels below local bg
    bg_sigma_diams: float = 4.0        # background smoothness in robot diameters (set-up only)
    bg_passes: int = 3                 # set-up: bg / auto-exclusion refinement passes
    auto_exclude: bool = True          # set-up: every OTHER dark blob in the learn frame is static -> exclude
    auto_exclude_grow_px: int = 4
    # ---- candidate acceptance (only an area sanity gate, no shape comparison)
    area_min_frac: float = 0.30
    area_max_frac: float = 4.0         # >1 so motion-blur streaks are accepted
    partial_min_frac: float = 0.08     # min visible area when the robot is cut by the image border
    # ---- search window
    win_R_factor: float = 3.0
    full_search_after: int = 5         # frames lost -> search the whole frame
    # ---- velocity filter (alpha-beta)
    alpha: float = 0.85
    beta: float = 0.50
    # ---- misc
    drift_comp: bool = True            # compensate global brightness drift (exposure not truly locked)
    boundary_points: int = 12
    shift_check_every: int = 30        # frames between workspace-shift checks (0 = off)
    shift_warn_px: float = 2.0


class Scale:
    """um-per-pixel for the current zoom setting, from magnification.json."""

    def __init__(self, path=None, key=None):
        path = path or os.path.join(HERE, "magnification.json")
        with open(path) as f:
            d = json.load(f)
        self.table = d["magnifications"]
        self.keys = sorted(self.table, key=float)
        self.capture_resolution = d.get("capture_resolution")
        self.key = key if key is not None else d.get("default", self.keys[0])
        if self.key not in self.table:
            raise KeyError(f"magnification {self.key!r} not in {self.keys}")

    @property
    def um_per_px(self):
        return float(self.table[self.key]["um_per_px"])

    @property
    def verified(self):
        return bool(self.table[self.key].get("verified", False))

    def step(self, d):
        i = self.keys.index(self.key) + d
        self.key = self.keys[max(0, min(len(self.keys) - 1, i))]
        return self.key

    def check_resolution(self, w, h):
        r = self.capture_resolution
        if r and (int(r[0]), int(r[1])) != (w, h):
            print(f"[warn] frame is {w}x{h} but magnification.json was calibrated at {r[0]}x{r[1]}: "
                  f"um/px is WRONG unless you rescale it")
