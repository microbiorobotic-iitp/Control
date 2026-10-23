"""
Magnetic field of a thick, finite-length coil (bobbin winding R1 -> R2, length L)
for square and hexagonal wire packing.

 - On-axis field: exact closed form (Fabry formula for a thick solenoid)
 - Radial profile at mid-plane (off-axis): numerical sum of loops using
   elliptic integrals (no elementary closed form exists)
"""
import numpy as np
import matplotlib.pyplot as plt
# from scipy.special import ellipk, ellipe

# ======================= GLOBAL PARAMETERS (edit here) =======================
MU0 = 4e-7 * np.pi          # T*m/A
R1 = 0.005                  # inner radius of winding [m]
R2 = 0.015                  # outer radius of winding [m]
L = 0.038                   # winding length along axis [m]
D_WIRE = 0.000559             # wire OUTER diameter incl. enamel [m]
CURRENT = 2.0               # current per turn [A]
actualN = 790      # Actual number of turns

N_AXIAL_POINTS = 400        # points for the axial plot
AXIAL_RANGE = 0.015          # plot x from -AXIAL_RANGE to +AXIAL_RANGE [m]
N_RADIAL_POINTS = 60        # points for the radial plot
GRID_NR, GRID_NZ = 40, 40   # loop grid used for the numerical (off-axis) sum
# =============================================================================



def turns(packing):
    area = (R2 - R1) * L
    if packing == "square":
        return area / D_WIRE**2
    if packing == "hex":
        return area / (np.sqrt(3) / 2 * D_WIRE**2)

    if packing == "actual":
            return actualN
    raise ValueError("packing must be 'square' or 'hex'")


def B_axis(x, N, I=CURRENT):
    """Exact on-axis field of a thick solenoid (Fabry); x = distance from centre."""
    x = np.asarray(x, dtype=float)
    zp, zm = x + L / 2, x - L / 2
    termp = zp * np.log((R2 + np.sqrt(R2**2 + zp**2)) / (R1 + np.sqrt(R1**2 + zp**2)))
    termm = zm * np.log((R2 + np.sqrt(R2**2 + zm**2)) / (R1 + np.sqrt(R1**2 + zm**2)))
    return MU0 * N * I / (2 * L * (R2 - R1)) * (termp - termm)


def main():
    packings = {"square": "tab:blue", "hex": "tab:red", "actual":"tab:green"}
    N = {p: turns(p) for p in packings}
    for p in packings:
        print(f"{p:7s}: N = {N[p]:.0f} turns,  B(centre) = {B_axis(0.0, N[p])*1e3:.3f} mT")

    x = np.linspace(-AXIAL_RANGE, AXIAL_RANGE, N_AXIAL_POINTS)

    fig, ax = plt.subplots(figsize=(7, 5))
    for p, c in packings.items():
        ax.plot(x * 1e3, B_axis(x, N[p]) * 1e3, c, label=f"{p} (N={N[p]:.0f})")
    ax.axvspan(-L / 2 * 1e3, L / 2 * 1e3, color="gray", alpha=0.15, label="winding length")
    ax.set(xlabel="Axial distance from centre x [mm]", ylabel="B [mT]",
           title="On-axis field of thick coil")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("thick_coil_axis.png", dpi=150)
    plt.show()

# ================= ADD THESE GLOBALS (next to the other globals) =================
MEAS_DUTY    = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
MEAS_FIELD   = [10,20,30,41,52,63,73,85,96,107,117,128,139,149,157,163,173,182,188,188]
MEAS_CURRENT = [0.05,0.07,0.1,0.14,0.2,0.27,0.3,0.5,0.5,0.6,0.8,0.9,1,1.1,1.3,1.5,1.6,1.9,2.1,2.21]
FIELD_UNIT_TO_TESLA = 1e-3   # measured field unit: 1e-3 for mT, 1e-4 for gauss, 1e-6 for uT
# ==================================================================================


# ============== ADD THIS FUNCTION (anywhere above main) ==============
def compare_with_measurement(x_mm=0.0):
    """Compare measured field at x_mm (on axis) with the analytical coil model."""
    I = np.array(MEAS_CURRENT, dtype=float)
    B_meas = np.array(MEAS_FIELD, dtype=float) * FIELD_UNIT_TO_TESLA
    duty = np.array(MEAS_DUTY, dtype=float)
    x = x_mm * 1e-3

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    I_line = np.linspace(0, I.max(), 100)
    ax[0].plot(I, B_meas * 1e3, "ko", label="measured")

    print(f"{'packing':8s} {'N':>6s} {'k_model [mT/A]':>15s} {'mean meas/model':>16s} {'meas/model @max I':>18s}")
    for p, c in {"square": "tab:blue", "hex": "tab:red"}.items():
        N = turns(p)
        k = B_axis(x, N, I=1.0)                 # field per amp (model is linear in I)
        ax[0].plot(I_line, k * I_line * 1e3, c, label=f"analytical {p} (N={N:.0f})")
        ax[1].plot(duty, B_meas / (k * I), c + "", marker="s", ls="-", label=f"measured / {p}")
        print(f"{p:8s} {N:6.0f} {k*1e3:15.2f} {np.mean((B_meas/(k*I))[I > 0.2]):16.2f} {B_meas[-1]/(k*I[-1]):18.2f}")

    ax[0].set(xlabel="Current [A]", ylabel="B at x = %.1f mm [mT]" % x_mm,
              title="Measured vs analytical")
    ax[1].axhline(1, color="gray", ls="--")
    ax[1].set(xlabel="Duty cycle", ylabel="B_measured / B_analytical",
              title="Ratio (1 = perfect agreement)")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()
# 

if __name__ == "__main__":
    # main()
    compare_with_measurement()