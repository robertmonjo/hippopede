"""Generate hippopede_growth_3d.png.

Visualises the nested hippopede hypersurfaces for t = 0.5, 1, 2, 3.
For fixed t and angle chi, the radius is rho(t, chi) = sqrt(sin^2(chi) + 4*t^2*cos^2(chi)).
The surface is a body of revolution around the u-axis:
  u      = rho(t, chi) * cos(chi)
  R_perp = rho(t, chi) * sin(chi)  (distance from u-axis)
Shown in 3D as (u, R_perp * cos(phi), R_perp * sin(phi)).
"""

from __future__ import annotations

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
OUT = FIGURES / "hippopede_growth_3d.png"

# Inner surfaces: light + opaque; outer surfaces: dark + transparent.
# Drawn from outermost to innermost so the painter's algorithm shows
# the inner (bright) shells through the semi-transparent outer (dark) ones.
T_VALUES   = [3.0,     2.0,     1.0,     0.5    ]
COLORS     = ["#1a1a1a", "#666666", "#aaaaaa", "#e8e8e8"]
ALPHAS     = [0.20,    0.40,    0.65,    0.90   ]

CHI = np.linspace(0, np.pi, 300)
PHI = np.linspace(0, 2 * np.pi, 120)
CHI_GRID, PHI_GRID = np.meshgrid(CHI, PHI)

# Light direction: azimuth 225°, altitude 45° (upper-left rear)
_az  = math.radians(225)
_alt = math.radians(45)
LIGHT = np.array([math.cos(_alt) * math.cos(_az),
                  math.cos(_alt) * math.sin(_az),
                  math.sin(_alt)])
AMBIENT  = 0.30   # fraction of ambient light (prevents totally black facets)
DIFFUSE  = 0.70   # fraction of diffuse light


def hippopede_surface(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho    = np.sqrt(np.sin(CHI_GRID) ** 2 + 4 * t ** 2 * np.cos(CHI_GRID) ** 2)
    u      = rho * np.cos(CHI_GRID)
    r_perp = rho * np.sin(CHI_GRID)
    y_ax   = r_perp * np.cos(PHI_GRID)
    z_ax   = r_perp * np.sin(PHI_GRID)
    return u, y_ax, z_ax


def surface_normals(u, y, z) -> np.ndarray:
    """Per-vertex outward unit normals via finite differences of the parametric grid."""
    # Tangent along chi (axis=1) and phi (axis=0)
    tu = np.stack([np.gradient(u, axis=1),
                   np.gradient(y, axis=1),
                   np.gradient(z, axis=1)], axis=-1)
    tv = np.stack([np.gradient(u, axis=0),
                   np.gradient(y, axis=0),
                   np.gradient(z, axis=0)], axis=-1)
    # Normal = tu × tv
    n = np.cross(tu, tv)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    mag = np.where(mag < 1e-12, 1.0, mag)
    return n / mag


def shaded_facecolors(u, y, z, base_color) -> np.ndarray:
    """RGBA array with Lambertian shading from the surface normals."""
    rgb    = np.array(mcolors.to_rgb(base_color))
    normals = surface_normals(u, y, z)
    # Lambertian: intensity = ambient + diffuse * max(0, dot(n, L))
    dot    = np.einsum("...i,i->...", normals, LIGHT)
    intensity = AMBIENT + DIFFUSE * np.clip(dot, 0, 1)
    fc = rgb[np.newaxis, np.newaxis, :] * intensity[:, :, np.newaxis]
    return np.clip(fc, 0, 1)


fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection="3d")

for t_val, color, alpha in zip(T_VALUES, COLORS, ALPHAS):
    u, y_ax, z_ax = hippopede_surface(t_val)
    fc = shaded_facecolors(u, y_ax, z_ax, color)
    ax.plot_surface(u, y_ax, z_ax, facecolors=fc, alpha=alpha,
                    linewidth=0, antialiased=True)

# Legend in increasing-t order
for t_val, color in sorted(zip(T_VALUES, COLORS)):
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
ranges    = [x_lims[1] - x_lims[0], y_lims[1] - y_lims[0], z_lims[1] - z_lims[0]]
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
