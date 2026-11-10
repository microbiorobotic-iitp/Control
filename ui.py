"""Interactive helpers (OpenCV windows): mask editor, camera tuning panel."""
import cv2
import numpy as np


def edit_mask(win, frame, mask=None):
    """Paint the areas the robot can NEVER be in (walls, dark corners, outside the workspace).
    Mouse (brush mode): left-drag = draw, right-drag = the opposite (erase).   E = swap so LEFT-drag erases
    (use this if your right mouse button / touchpad does not register).   [ / ] = brush size
    P = polygon mode: left-click vertices, then  K = keep INSIDE (exclude outside)  or  X = exclude inside;  ESC = cancel polygon
    Z = undo   C = clear   ENTER = accept   ESC (no polygon) = cancel.   Returns uint8 mask (255 = excluded)."""
    H, W = frame.shape[:2]
    orig = np.zeros((H, W), np.uint8) if mask is None else mask.copy()
    mask = orig.copy()
    st = dict(brush=14, paint=None, last=None, mode="brush", poly=[], mouse=(0, 0), undo=[], erase=False)

    def push():
        st["undo"].append(mask.copy())
        del st["undo"][:-25]

    def on_mouse(ev, x, y, flags, _):
        st["mouse"] = (x, y)
        if st["mode"] == "poly":
            if ev == cv2.EVENT_LBUTTONDOWN:
                st["poly"].append((x, y))
            return
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            push()
            left = ev == cv2.EVENT_LBUTTONDOWN
            st["paint"] = 255 if (left != st["erase"]) else 0      # E swaps what the left button does
            st["last"] = (x, y)
        if ev == cv2.EVENT_MOUSEMOVE and st["paint"] is not None:
            cv2.line(mask, st["last"], (x, y), int(st["paint"]), st["brush"] * 2)
            st["last"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN or ev == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(mask, (x, y), st["brush"], int(st["paint"]), -1)
        if ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            st["paint"] = None

    cv2.setMouseCallback(win, on_mouse)
    red = np.zeros_like(frame)
    red[:] = (0, 0, 255)
    while True:
        view = cv2.addWeighted(frame, 0.5, red, 0.5, 0)
        view = np.where(mask[..., None] > 0, view, frame)
        if st["mode"] == "brush":
            cv2.circle(view, st["mouse"], st["brush"], (0, 255, 0) if st["erase"] else (0, 255, 255), 1)
            msg = (f"MASK [{'LEFT=ERASE' if st['erase'] else 'LEFT=DRAW'}]  E swap draw/erase  [ ] size  "
                   f"P polygon  Z undo  C clear  ENTER accept  ESC cancel")
        else:
            if st["poly"]:
                cv2.polylines(view, [np.array(st["poly"], np.int32)], False, (0, 255, 0), 2)
                cv2.line(view, st["poly"][-1], st["mouse"], (0, 255, 0), 1)
            msg = "POLYGON  click vertices | K keep inside | X exclude inside | ESC back"
        cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(15) & 0xFF
        if k in (13, 10):
            break
        if k == 27:
            if st["mode"] == "poly":
                st["mode"], st["poly"] = "brush", []
            else:
                mask = orig
                break
        elif k == ord("e"):
            st["erase"] = not st["erase"]
        elif k == ord("["):
            st["brush"] = max(2, st["brush"] - 3)
        elif k == ord("]"):
            st["brush"] += 3
        elif k == ord("z") and st["undo"]:
            mask[:] = st["undo"].pop()
        elif k == ord("c"):
            push()
            mask[:] = 0
        elif k == ord("p"):
            st["mode"], st["poly"] = "poly", []
        elif k in (ord("k"), ord("x")) and st["mode"] == "poly" and len(st["poly"]) >= 3:
            push()
            poly = np.zeros((H, W), np.uint8)
            cv2.fillPoly(poly, [np.array(st["poly"], np.int32)], 255)
            mask[:] = np.maximum(mask, cv2.bitwise_not(poly) if k == ord("k") else poly)
            st["mode"], st["poly"] = "brush", []
    cv2.setMouseCallback(win, lambda *a: None)
    return mask


def tune_camera(win, cam, get_frame):
    """Live camera panel. Trackbars write straight to the camera. Saturated pixels are painted red.
    E = lock exposure (keeps the value the camera has now, verifies it) | A = auto exposure back on
    D = driver dialog (Windows) | ENTER = done."""
    cp = cv2
    ctl = "camera controls"
    cv2.namedWindow(ctl, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctl, 420, 200)
    st = dict(msg="")

    def mk(name, prop, lo, hi, to_cam=lambda v: v, from_cam=lambda v: v):
        cur = from_cam(cam.get(prop))
        cv2.createTrackbar(name, ctl, int(np.clip(cur, lo, hi)), hi, lambda v: cam.set(prop, to_cam(v)))
    mk("exposure(-log2 s)", cp.CAP_PROP_EXPOSURE, 0, 13, lambda v: -float(v), lambda v: -v)
    mk("gain", cp.CAP_PROP_GAIN, 0, 255)
    mk("brightness", cp.CAP_PROP_BRIGHTNESS, 0, 255)
    mk("contrast", cp.CAP_PROP_CONTRAST, 0, 255)
    while True:
        f = get_frame()
        if f is None:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        view = f.copy()
        sat = g >= 250
        view[sat] = (0, 0, 255)
        hist = cv2.calcHist([g], [0], None, [64], [0, 256]).ravel()
        hist = (hist / hist.max() * 60).astype(int)
        for i, h in enumerate(hist):
            cv2.line(view, (10 + i * 3, 120), (10 + i * 3, 120 - h), (255, 255, 255), 2)
        cv2.putText(view, f"mean={g.mean():.0f}  saturated={100 * sat.mean():.2f}%  {st['msg']}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(view, "E lock exposure | A auto | D driver dialog | ENTER done", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow(win, view)
        k = cv2.waitKey(1) & 0xFF
        if k in (13, 10, 27):
            break
        if k == ord("e"):
            ok, st["msg"] = cam.lock_exposure()
            print("[camera]", st["msg"])
        elif k == ord("a"):
            cam.auto_exposure_on()
            st["msg"] = "auto exposure ON"
        elif k == ord("d"):
            cam.open_driver_dialog()
    cv2.destroyWindow(ctl)