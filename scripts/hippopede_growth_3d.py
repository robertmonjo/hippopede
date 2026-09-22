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

# Innermost surface: darkest + most opaque.
# Outermost surface: lightest + most transparent (so inner surfaces remain visible).
# Drawn from outermost to innermost so the opaque dark cores appear on top.
T_VALUES   = [3.0,     2.0,     1.0,     0.5    ]  # draw order: outer first
COLORS     = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
ALPHAS     = [0.18,    0.40,    0.65,    0.88   ]

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

# Legend in increasing-t order (T_VALUES is decreasing for draw order)
for t_val, color in sorted(zip(T_VALUES, COLORS)):
    ax.plot([], [], [], color=color, lw=3, label=f"$t={t_val}$")

ax.set_xlabel("$u$", labelpad=8)
ax.set_ylabel(r"$\sqrt{x^2+y^2}$", labelpad=8)
ax.set_zlabel("")   # placeholder; actual label placed as 3D text below
ax.view_init(elev=20, azim=-60)

# Tight proportional limits: t=3 surface spans u∈[-6,6] but y/z∈[-3,3].
# set_box_aspect matches the physical scale to these data ranges (no distortion).
pad = 0.4
ax.set_xlim3d(-6 - pad, 6 + pad)
ax.set_ylim3d(-3 - pad, 3 + pad)
ax.set_zlim3d(-3 - pad, 3 + pad)
ax.set_box_aspect([12.8, 6.8, 6.8])

ax.legend(loc="lower left", fontsize=9)
# z-axis label: tight bbox clips set_zlabel, so place it in 2D axes coords
ax.text2D(0.93, 0.72, "$z$", transform=ax.transAxes, fontsize=12, ha="left", va="bottom")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.3)
plt.close()
print(f"Saved {OUT}")
