"""Import user-measured wavelength-dependent optical coefficients."""
from __future__ import annotations
import csv
import io
import numpy as np
from .core import MATERIAL_NAMES, MATERIALS_800

COLUMNS = ('wavelength_nm', 'material', 'mu_a_mm_1', 'mu_s_mm_1', 'g', 'n')


def load_spectra_csv(contents: str | bytes) -> dict[str, np.ndarray]:
    """Read measurement table, rejecting incomplete or duplicate wavelength entries.

    One record per wavelength/material. Example material names: 'Shell', 'Yolk'.
    Air is assigned analytically; all four other materials are required.
    """
    if isinstance(contents, bytes):
        contents = contents.decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(contents))
    if not set(COLUMNS).issubset(set(reader.fieldnames or [])):
        raise ValueError('Required columns: ' + ', '.join(COLUMNS))
    by_material: dict[str, list[tuple[float, float, float, float, float]]] = {
        name: [] for name in MATERIAL_NAMES[1:]
    }
    for row in reader:
        name = row['material'].strip()
        if name not in by_material:
            raise ValueError(f'Unknown material {name!r}; expected {list(by_material)}')
        nums = tuple(float(row[k]) for k in COLUMNS if k != 'material')
        wavelength, mu_a, mu_s, g, n = nums
        if not (400 <= wavelength <= 1700 and mu_a >= 0 and mu_s >= 0 and abs(g) < 1 and n >= 1 and all(np.isfinite(nums))):
            raise ValueError(f'Invalid coefficients for {name} at {wavelength}nm')
        by_material[name].append(nums)
    out = {}
    for name, rows in by_material.items():
        if not rows:
            raise ValueError(f'Missing records for {name}')
        values = np.asarray(sorted(rows), dtype=np.float64)
        if np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError(f'Duplicate or invalid wavelength for {name}')
        out[name] = values
    return out


def at_wavelength(table: dict[str, np.ndarray], wavelength_nm: float) -> np.ndarray:
    """Linearly interpolate measured optical properties; NEVER extrapolate."""
    out = np.array(MATERIALS_800, copy=True)
    for i, name in enumerate(MATERIAL_NAMES[1:], 1):
        data = table[name]
        if wavelength_nm < data[0,0] or wavelength_nm > data[-1,0]:
            raise ValueError(f'{name}: {wavelength_nm:g}nm outside [{data[0,0]:g}, {data[-1,0]:g}]nm; extrapolation forbidden')
        for j in range(4):
            out[i,j]=np.interp(wavelength_nm,data[:,0],data[:,j+1])
    return out