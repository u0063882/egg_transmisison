"""Research prototype for spatial egg transmission using Monte Carlo packets."""
from .core import Geometry, Optics, MATERIALS_800, MODE_NAMES, MATERIAL_NAMES, simulate, projected_masks
__all__ = ['Geometry', 'Optics', 'MATERIALS_800', 'MODE_NAMES', 'MATERIAL_NAMES', 'simulate', 'projected_masks']
__version__ = '0.1.0'