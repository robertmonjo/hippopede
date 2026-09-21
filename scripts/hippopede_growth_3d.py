"""Generate hippopede_growth_3d.png.

Visualises the nested hippopede hypersurfaces for t = 0.5, 1, 2, 3.
For fixed t and angle chi, the radius is rho(t, chi) = sqrt(sin^2(chi) + 4*t^2*cos^2(chi)).
The surface is a body of revolution around the u-axis:
  u      = rho(t, chi) * cos(chi)
  R_perp = rho(t, chi) * sin(chi)  (distance from u-axis)
Shown in 3D as (u, R_perp * cos(phi), R_perp * sin(phi)).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
OUT = FIGURES / "hippopede_growth_3d.png"

T_VALUES = [0.5, 1.0, 2.0, 3.0]
# Grayscale + alpha: outer surfaces are pale and translucent so the
# inner (darker, more opaque) surfaces remain visible inside them.
COLORS = ["#1a1a1a", "#555555", "#999999", "#cccccc"]
ALPHAS = [0.80, 0.60, 0.40, 0.25]

CHI = np.linspace(0, np.pi, 300)
PHI = np.linspace(0, 2 * np.pi, 120)
CHI_GRID, PHI_GRID = np.meshgrid(CHI, PHI)


def hippopede_surface(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.sqrt(np.sin(CHI_GRID) ** 2 + 4 * t ** 2 * np.cos(CHI_GRID) ** 2)
    u = rho * np.cos(CHI_GRID)
    r_perp = rho * np.sin(CHI_GRID)
    y_ax = r_perp * np.cos(PHI_GRID)
    z_ax = r_perp * np.sin(PHI_GRID)
    return u, y_ax, z_ax


fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

for t_val, color, alpha in zip(T_VALUES, COLORS, ALPHAS):
    u, y_ax, z_ax = hippopede_surface(t_val)
    ax.plot_surface(u, y_ax, z_ax, color=color, alpha=alpha, linewidth=0, antialiased=True)
    ax.plot([], [], [], color=color, lw=3, label=f"$t={t_val}$")

ax.set_xlabel("$u$", labelpad=8)
ax.set_ylabel(r"$\sqrt{x^2+y^2}$", labelpad=8)
ax.set_zlabel("$z$", labelpad=8)
ax.legend(loc="upper left", fontsize=9)
ax.view_init(elev=20, azim=-60)

# Equal physical scale on all three axes: expand shorter axes to the max range.
x_lims = ax.get_xlim3d()
y_lims = ax.get_ylim3d()
z_lims = ax.get_zlim3d()
ranges = [x_lims[1] - x_lims[0], y_lims[1] - y_lims[0], z_lims[1] - z_lims[0]]
max_range = max(ranges)
x_mid = sum(x_lims) / 2
y_mid = sum(y_lims) / 2
z_mid = sum(z_lims) / 2
ax.set_xlim3d(x_mid - max_range / 2, x_mid + max_range / 2)
ax.set_ylim3d(y_mid - max_range / 2, y_mid + max_range / 2)
ax.set_zlim3d(z_mid - max_range / 2, z_mid + max_range / 2)
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
