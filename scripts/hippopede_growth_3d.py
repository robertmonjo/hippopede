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
COLORS = ["#4477AA", "#66CCEE", "#CCBB44", "#EE6677"]

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

for t_val, color in zip(T_VALUES, COLORS):
    u, y_ax, z_ax = hippopede_surface(t_val)
    ax.plot_surface(u, y_ax, z_ax, color=color, alpha=0.35, linewidth=0, antialiased=True)
    ax.plot([], [], [], color=color, lw=3, label=f"$t={t_val}$")

ax.set_xlabel("$u$", labelpad=8)
ax.set_ylabel(r"$\sqrt{x^2+y^2}$", labelpad=8)
ax.set_zlabel("$z$", labelpad=8)
ax.set_title("Hippopede hypersurfaces", pad=12)
ax.legend(loc="upper left", fontsize=9)
ax.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
