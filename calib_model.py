import warnings

import numpy as np

import config as C

FLIP = np.array([1.0, -1.0])      # image (y down) <-> control frame (y up); its own inverse
NAMES = ['pX', 'nX', 'pY', 'nY', 'pXpY', 'pXnY', 'nXpY', 'nXnY']


def _binned(s, m, edges, fn):
    idx = np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 2)
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(out)):
        sel = idx == b
        if sel.any():
            out[b] = fn(m[sel])
    return out


def _fill(c, y):
    ok = np.isfinite(y)
    return np.interp(c, c[ok], y[ok]) if ok.any() else None


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _fit_ray(name, r, centre):
    d = np.asarray(r['vec'], float)
    s = ((r['pos'] - centre) * FLIP) @ d            # progress along d from rest centre
    up, dn = r['phase'] == 0, r['phase'] == 1
    if up.sum() < 3 or dn.sum() < 3:
        return None
    net = (np.asarray(r['wall_pos'], float) - centre) * FLIP
    if net @ d <= 0:
        warnings.warn(f'{name}: bot moved against the commanded direction - ray skipped '
                      f'(check y-flip / wiring).')
        return None

    edges = np.linspace(s.min(), s.max() + 1e-9, C.MODEL_BINS + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    # up leg: highest PWM used while in a bin ~ breakaway; down leg: lowest PWM held ~ return level
    m_up = _fill(c, _binned(s[up], r['mag'][up], edges, np.max))
    m_dn = _fill(c, _binned(s[dn], r['mag'][dn], edges, np.min))
    if m_up is None or m_dn is None:
        return None

    # near home the bot stops because restoring->0, not because PWM hit the return level
    # (m_dn is censored at 0), so there use the typical friction from uncensored bins.
    cens = m_dn <= 0.5 * C.RETREAT_STEP_PWM
    f_raw = np.clip(0.5 * (m_up - m_dn), 0, None)
    f_med = np.median(f_raw[~cens]) if (~cens).any() else 0.5 * m_up.min()
    R = np.where(cens, np.clip(m_up - f_med, 0, None), 0.5 * (m_up + m_dn))
    f = np.where(cens, f_med, f_raw)

    v = (r['vel'] * FLIP) @ d
    idx = np.clip(np.digitize(s, edges) - 1, 0, C.MODEL_BINS - 1)
    G = np.full(C.MODEL_BINS, np.nan)
    for b in range(C.MODEL_BINS):
        sel = (idx == b) & (v > C.MOVING_SPEED_PX_S)          # only moving samples
        if sel.sum() >= 3 and np.ptp(r['mag'][sel]) >= 2 * C.RAMP_STEP:
            G[b] = np.polyfit(r['mag'][sel], v[sel], 1)[0]     # (v(u2)-v(u1))/(u2-u1)
    G = _fill(c, G)          # interpolate empty bins; may be None -> handle in build_model

    # speed gain: speed ~ k * (PWM - hold level), hold level = R+f going up, R-f going back
    Rs, fs = np.interp(s, c, R), np.interp(s, c, f)
    e = r['mag'] - np.where(up, Rs + fs, Rs - fs)
    v = ((r['vel'] * FLIP) @ d)
    keep = np.abs(e) > C.RAMP_STEP

    theta = float(np.arctan2(d[1], d[0]))
    pos_ok = c > 0
    return dict(s=c, m_up=m_up, m_dn=m_dn, R=R[pos_ok], f=f[pos_ok], G=G[pos_ok], theta=theta,
                bias=float(np.clip(_wrap(np.arctan2(net[1], net[0]) - theta), -0.5, 0.5)),
                e=e[keep], v=v[keep],
                P=centre + np.outer(c[pos_ok], d) * FLIP,    # image px
                V=np.outer(R[pos_ok], d))                    # control-frame PWM vectors


def build_model(calib):
    homes = np.array([calib[n]['home'] for n in NAMES], float)
    centre = np.nanmedian(homes, axis=0)           # where the bot rests with no current

    rays = {}
    for n in NAMES:
        fit = _fit_ray(n, calib[n], centre)
        if fit is not None:
            rays[n] = fit
    if len(rays) < 4:
        return None

    E = np.concatenate([x['e'] for x in rays.values()])
    V = np.concatenate([x['v'] for x in rays.values()])
    k_pool = float(V @ E / (E @ E)) if E.size > 5 else np.nan
    if not np.isfinite(k_pool):
        print('WARNING: speed gain not identifiable from calibration data, using K_FALLBACK')
        for x in rays.values():
            x['k'] = C.K_FALLBACK
    else:
        for x in rays.values():
            k = float(x['v'] @ x['e'] / (x['e'] @ x['e'])) if x['e'].size > 5 else np.nan
            x['k'] = k_pool if not np.isfinite(k) else float(np.clip(k, 0.5 * k_pool, 2 * k_pool))

    order = sorted(rays.values(), key=lambda x: x['theta'])

    Pts = np.vstack([x['P'] for x in order])                       # image px
    rho = np.concatenate([x['R'] for x in order])
    phi = np.concatenate([x['f'] for x in order])
    G   = np.concatenate([x['G'] for x in order])
    G   = np.where(np.isfinite(G), G, np.nanmedian(G))             # fallback for empty bins
    # add the home point: rho = 0
    Pts = np.vstack([centre[None], Pts]); rho = np.r_[0, rho]
    phi = np.r_[np.median(phi), phi];     G = np.r_[np.median(G), G]
    return dict(centre=centre, rays=rays, pts=Pts, rho=rho, phi=phi, G=G,
        theta=np.array([x['theta'] for x in order]),
        f=np.array([x['f'] for x in order]),
        k=np.array([x['k'] for x in order]),
        bias=np.array([x['bias'] for x in order]),
        P=np.vstack([centre[None]] + [x['P'] for x in order]),
        V=np.vstack([np.zeros((1, 2))] + [x['V'] for x in order]),
    )

def _idw(model, pos_img, key, eps=1e-6):
    d2 = np.sum((model['pts'] - np.asarray(pos_img, float))**2, axis=1)
    w = 1.0 / (d2 + eps)
    w /= w.sum()                                  # sum w_i = 1
    return float(w @ model[key])

def restoring_drive(model, pos_img):
    v = (np.asarray(pos_img, float) - model['centre']) * FLIP
    r = np.linalg.norm(v)
    return np.zeros(2) if r < 1e-6 else _idw(model, pos_img, 'rho') * v / r

def motion_drive(model, pos_img, u, v_target):
    if np.linalg.norm(u) < 1e-9:
        return 0.0
    phi, G = _idw(model, pos_img, 'phi'), _idw(model, pos_img, 'G')
    excess = v_target / G if G > 0 else 0.0
    return phi + min(excess, C.MAX_EXCESS_PWM)

def local_params(model, pos_img):
    """Local G(p), phi(p), rho(p) by inverse-distance² weighting."""
    G = _idw(model, pos_img, 'G')
    return dict(G=G if G > 1e-6 else C.K_FALLBACK,
                phi=_idw(model, pos_img, 'phi'),
                rho=_idw(model, pos_img, 'rho'))

def _ang(model, key, theta):
    return float(np.interp(theta, model['theta'], model[key], period=2 * np.pi))


def restoring_drive(model, pos_img):
    """Control-frame command that cancels the meniscus pull: radial, pointing away from home."""
    v = (np.asarray(pos_img, float) - model['centre']) * FLIP   # outward direction, control frame
    r = np.linalg.norm(v)
    if r < 1e-6:
        return np.zeros(2)
    return np.interp(r, model['r_grid'], model['R_grid']) * v / r


def motion_drive(model, u, v_target):
    """PWM along u needed to overcome friction and move at v_target (px/s)."""
    if np.linalg.norm(u) < 1e-9:
        return 0.0
    th = np.arctan2(u[1], u[0])
    f, k = _ang(model, 'f', th), _ang(model, 'k', th)
    excess = v_target / k if np.isfinite(k) and k > 0 else 0.0
    return f + min(excess, C.MAX_EXCESS_PWM)


def heading_correction(model, u):
    """Rotate command so the bot's actual heading matches the wanted one."""
    n = np.linalg.norm(u)
    if n < 1e-9:
        return u
    a = np.arctan2(u[1], u[0]) - C.HEADING_CORR_GAIN * _ang(model, 'bias', np.arctan2(u[1], u[0]))
    return np.array([np.cos(a), np.sin(a)])


def plot_model(model):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(13, 9), num='Calibration model')
    for ax, (n, x) in zip(axs.flat, model['rays'].items()):
        ax.plot(x['s'], x['m_up'], 'b.-', label='breakaway R+f')
        ax.plot(x['s'], x['m_dn'], 'r.-', label='return R-f')
        ax.plot(x['s'], x['R'], 'k-', lw=2, label='restoring R')
        ax.set_title(f"{n}  f={np.median(x['f']):.3f}  k={x['k']:.0f}  bias={np.degrees(x['bias']):.0f}°")
        ax.set_xlabel('s (px)'); ax.set_ylabel('PWM')
    axs.flat[0].legend()
    fig.tight_layout(); plt.show()