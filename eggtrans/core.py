"""Egg geometry and public API. Photon transport is performed by PMCX (MCX).

No Monte Carlo scattering, absorption, Fresnel or propagation algorithms are
implemented in this project; `eggtrans.mcx_engine` builds MCX inputs and bins
its *detected* photons into camera pixels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal
import numpy as np

Mode = Literal['Transverse line', 'Spot', 'Longitudinal line', 'Broad backlight']
MODE_NAMES: tuple[Mode, ...] = ('Transverse line', 'Spot', 'Longitudinal line', 'Broad backlight')
MATERIAL_NAMES = ('Air', 'Shell', 'Albumen', 'Yolk', 'Blood spot')
# [absorption 1/mm, scattering 1/mm, g, n] at 800 nm.
# Literature anchors: albumen & yolk mua, mus, g ONLY; other numbers illustrative.
MATERIALS_800 = np.array([
    [0.000, 0.000, 0.00, 1.000],
    [0.040, 2.000, 0.85, 1.530],
    [0.006, 0.069, 0.90, 1.350],
    [0.014, 5.350, 0.92, 1.400],
    [0.700, 1.000, 0.90, 1.350],
], dtype=np.float64)

@dataclass
class Geometry:
    egg_width_mm: float = 44.0
    egg_length_mm: float = 59.0
    egg_height_mm: float = 44.0
    shell_thickness_mm: float = 0.35
    yolk_radius_x_mm: float = 14.0
    yolk_radius_y_mm: float = 16.0
    yolk_radius_z_mm: float = 14.0
    yolk_offset_x_mm: float = -1.5
    yolk_offset_y_mm: float = 0.0
    yolk_offset_z_mm: float = -1.0
    include_yolk: bool = True
    blood_x_mm: float = 6.0
    blood_y_mm: float = 13.0
    blood_z_mm: float = 9.0
    blood_radius_mm: float = 2.3
    include_blood: bool = True

@dataclass
class Optics:
    mode: Mode = 'Transverse line'
    line_length_mm: float = 48.0
    line_width_mm: float = 2.0
    spot_diameter_mm: float = 6.0
    source_offset_y_mm: float = 0.0
    source_gap_mm: float = 2.0
    camera_plane_gap_mm: float = 3.0
    # MCX native planar/disk focusing: 0 = parallel, negative = divergent,
    # positive = convergent. One *isotropic* focal length, NOT independent x/y.
    beam_focal_length_mm: float = 0.0
    camera_half_angle_deg: float = 35.0
    camera_line_width_mm: float = 4.0
    photons_per_line: int = 100_000
    scan_lines: int = 24
    camera_pixels: int = 64
    voxel_mm: float = 0.5
    seed: int = 42
    gpu_id: int = 1


def validate(g: Geometry, o: Optics, materials: np.ndarray) -> None:
    m = np.asarray(materials)
    if m.shape != (5, 4) or not np.isfinite(m).all():
        raise ValueError('Material properties must be a finite 5×4 [mua,mus,g,n] array.')
    if np.any(m[:, :2] < 0) or np.any(np.abs(m[:, 2]) > 1) or np.any(m[:, 3] < 1):
        raise ValueError('Require mua,mus >=0, |g|<=1, n>=1.')
    radii = np.array([g.egg_width_mm, g.egg_length_mm, g.egg_height_mm]) / 2
    if g.shell_thickness_mm <= 0 or np.min(radii) <= g.shell_thickness_mm:
        raise ValueError('Egg must be larger than the shell thickness.')
    yr = np.array([g.yolk_radius_x_mm, g.yolk_radius_y_mm, g.yolk_radius_z_mm])
    yc = np.array([g.yolk_offset_x_mm, g.yolk_offset_y_mm, g.yolk_offset_z_mm])
    if np.any(yr <= 0) or np.any((np.abs(yc)+yr) >= (radii-g.shell_thickness_mm)):
        raise ValueError('Yolk ellipsoid must fit inside the shell. Adjust size/position.')
    if g.blood_radius_mm <= 0:
        raise ValueError('Blood radius must be positive.')
    if g.include_blood:
        bp = np.array([g.blood_x_mm,g.blood_y_mm,g.blood_z_mm])
        if np.sum((bp/(radii-g.shell_thickness_mm))**2) > 0.84:
            raise ValueError('Blood inclusion must be inside the shell.')
    if o.mode not in MODE_NAMES:
        raise ValueError(f'Unknown source mode: {o.mode}')
    if min(o.line_length_mm,o.line_width_mm,o.spot_diameter_mm,o.source_gap_mm,
           o.camera_plane_gap_mm,o.camera_line_width_mm,o.voxel_mm) <= 0:
        raise ValueError('Widths, distances and voxel size must be positive.')
    if not (0 < o.camera_half_angle_deg < 90):
        raise ValueError('Camera half-angle must be 0–90° (exclusive).')
    if min(o.scan_lines,o.camera_pixels,o.photons_per_line) < 1 or o.gpu_id < 1:
        raise ValueError('Packet count, scan size and GPU ID must be positive.')
    if o.voxel_mm > min(g.shell_thickness_mm*10, 3):
        # Warn through UI; do not claim subvoxel physical shell accuracy.
        pass


def projected_masks(g: Geometry, x: np.ndarray, y: np.ndarray) -> dict[str,np.ndarray]:
    xx,yy=np.meshgrid(x,y)
    return {
        'egg': (xx/(g.egg_width_mm/2))**2 + (yy/(g.egg_length_mm/2))**2 <= 1,
        'yolk': g.include_yolk & (((xx-g.yolk_offset_x_mm)/g.yolk_radius_x_mm)**2 +
                                   ((yy-g.yolk_offset_y_mm)/g.yolk_radius_y_mm)**2 <= 1),
        'blood': g.include_blood & (((xx-g.blood_x_mm)**2+(yy-g.blood_y_mm)**2) <= g.blood_radius_mm**2)
    }


def simulate(geometry: Geometry | None = None, optics: Optics | None = None,
             materials: np.ndarray | None = None) -> dict:
    """Run PMCX for each scan position. Requires installed PMCX + NVIDIA GPU."""
    from .mcx_engine import simulate_mcx
    return simulate_mcx(geometry or Geometry(), optics or Optics(),
                        np.asarray(MATERIALS_800 if materials is None else materials))
