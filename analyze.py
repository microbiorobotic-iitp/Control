import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np

from storage import load_pickle
import config as C


def pick_files():
    root = tk.Tk()
    root.withdraw()
    return list(filedialog.askopenfilenames(title='Select log file(s)', initialdir=C.LOG_DIR,
                                            filetypes=[('Log files', 'log_*.pkl')]))


def load_experiments(paths):
    data, labels = [], []
    for p in paths:
        S = load_pickle(p)
        data.append(S)
        m = S['meta']
        name = os.path.splitext(os.path.basename(p))[0]
        labels.append(f"{name} (Kp={m['Kp_v']:.4g} Ki={m['Ki_v']:.4g} Kd={m['Kd_v']:.4g})")
    return data, labels


def plot_trajectory(data, labels, colors):
    plt.figure('Trajectory')
    for d, lab, c in zip(data, labels, colors):
        plt.plot(d['log']['x'], d['log']['y'], '-', color=c, linewidth=1.5, label=lab)
    pp = data[0]['meta']['path_pts']
    plt.plot(pp[:, 0], pp[:, 1], 'k--o', linewidth=1, label='Intended path')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.xlabel('x (px)'); plt.ylabel('y (px)'); plt.title('Bot Trajectory vs Intended Path')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); plt.grid(True)
    plt.tight_layout()


def plot_speed(data, labels, colors):
    fig, (ax1, ax2) = plt.subplots(2, 1, num='Speed Tracking')
    for d, lab, c in zip(data, labels, colors):
        L = d['log']
        ax1.plot(L['t'], L['filtered_speed'], '-', color=c, linewidth=1.3, label=lab)
        ax2.plot(L['t'], L['error_v'], '-', color=c, linewidth=1.3, label=lab)
    ax1.plot(data[0]['log']['t'], data[0]['log']['target_speed'], 'k--', linewidth=1.2, label='Target speed')
    ax1.set_xlabel('time (s)'); ax1.set_ylabel('speed (px/s)'); ax1.set_title('Filtered Speed vs Target')
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlabel('time (s)'); ax2.set_ylabel('speed error (px/s)'); ax2.set_title('Velocity Error')
    for ax in (ax1, ax2):
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)); ax.grid(True)
    fig.tight_layout()


def plot_pid_breakdown(data, labels):
    plt.figure('PID Term Breakdown')
    L = data[0]['log']
    plt.plot(L['t'], L['P_term'], 'r', label='P term')
    plt.plot(L['t'], L['I_term'], 'g', label='I term')
    plt.plot(L['t'], L['D_term'], 'b', label='D term')
    plt.plot(L['t'], L['magnitude'], 'k', label='Output magnitude')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('time (s)'); plt.ylabel('contribution'); plt.title('PID Breakdown: ' + labels[0]); plt.grid(True)
    plt.tight_layout()


def plot_series(name, key, data, labels, colors, ylabel, title, zero_line=False):
    plt.figure(name)
    for d, lab, c in zip(data, labels, colors):
        L = d['log']
        plt.plot(L['t'], L[key], '-', color=c, linewidth=1.3, label=lab)
    if zero_line:
        plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('time (s)'); plt.ylabel(ylabel); plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)); plt.grid(True)
    plt.tight_layout()


def print_summary(data, paths):
    print(f"\n{'Experiment':<30} {'TotalTime':>10} {'RMS_CTE':>10} {'RMS_VelErr':>10} {'PathEff':>10}")
    for d, p in zip(data, paths):
        L, m = d['log'], d['meta']
        total_time = L['t'][-1]
        rms_cte = np.sqrt(np.mean(L['cross_track'] ** 2))
        rms_verr = np.sqrt(np.mean(L['error_v'] ** 2))
        path_len = np.sum(np.sqrt(np.diff(L['x']) ** 2 + np.diff(L['y']) ** 2))
        straight_len = np.linalg.norm(m['raw_goals'][-1] - m['path_pts'][0])
        path_eff = straight_len / path_len
        print(f"{os.path.basename(p):<30} {total_time:>10.2f} {rms_cte:>10.2f} {rms_verr:>10.2f} {path_eff:>10.3f}")


def main():
    paths = pick_files()
    if not paths:
        print('No files selected.')
        return
    data, labels = load_experiments(paths)
    colors = [plt.cm.tab10(i % 10) for i in range(len(paths))]

    plot_trajectory(data, labels, colors)
    plot_speed(data, labels, colors)
    plot_pid_breakdown(data, labels)
    plot_series('Control Effort', 'magnitude', data, labels, colors, 'magnitude (PWM)', 'Control Output')
    plot_series('Cross-Track Error', 'cross_track', data, labels, colors, 'perpendicular deviation (px)',
                'Cross-Track Error (0 = perfectly straight)', zero_line=True)
    print_summary(data, paths)
    plt.show()


if __name__ == '__main__':
    main()