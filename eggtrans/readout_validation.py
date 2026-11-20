"""Readout equivalence check using the same MCX photon exits.

This checks detector discretization only. It is not a comparison between an
actual moving pushbroom experiment and a stationary 2D acquisition: that
requires equivalent illumination, motion, exposure, and detector throughput.
All optical transport is delegated to PMCX/MCX; no local ray-tracing kernel.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np

from .core import Geometry, Optics, MATERIALS_800, validate
from .mcx_engine import (
    backend_status, build_egg_volume, mcx_config, top_face_photons,
    camera_bin, camera_bin_area, photon_history_mask,
)


def compare_fixed_scene_readouts(
    geometry: Geometry,
    optics: Optics,
    materials: np.ndarray | None = None,
    backend=None,
) -> dict:
    """Simulate one fixed scene, then read its exits as an area image and rows.

    Uses one native MCX photon transport run and the *same recorded photons*
    for both readouts. The virtual line detector uses the area camera pixel
    pitch as its slit width, making the two integrations mathematically
    equivalent up to floating-point bin-edge conventions.

    This is a diagnostic of readout geometry, not an independent proof of
    photon-transport accuracy or a moving-scanner versus fixed-camera test.
    """
    mat = np.asarray(MATERIALS_800 if materials is None else materials)
    o = replace(optics, camera_mode='Area')
    validate(geometry, o, mat)
    if backend is None:
        try:
            import pmcx as backend
        except (ImportError, OSError) as exc:
            raise RuntimeError('PMCX must be installed to run this check.') from exc
        ok, msg = backend_status()
        if not ok:
            raise RuntimeError(msg)

    domain = build_egg_volume(geometry, o, mat)
    x_edges = np.linspace(-geometry.egg_width_mm * .56,
                          geometry.egg_width_mm * .56, o.camera_pixels + 1)
    y_edges = np.linspace(-geometry.egg_length_mm * .54,
                          geometry.egg_length_mm * .54, o.area_rows + 1)
    x_mm = .5 * (x_edges[:-1] + x_edges[1:])
    y_mm = .5 * (y_edges[:-1] + y_edges[1:])
    cfg = mcx_config(domain, o, 0., max(1, int(o.seed)))
    output = backend.mcxlab(cfg)
    if not isinstance(output, dict) or 'detp' not in output:
        raise RuntimeError('PMCX did not return detected photons.')
    p,v,w=top_face_photons(output['detp'],domain.props,backend,unitinmm=domain.grid_mm)
    keep=photon_history_mask(output['detp'],len(p),o,domain.grid_mm,
                             n_materials=domain.props.shape[0]-1)
    p,v,w=p[keep],v[keep],w[keep]
    area, n_area = camera_bin_area(p, v, w, domain, o, y_edges, x_edges)

    # Match line slit width to the area pixel row pitch. This deliberately
    # differs from the user-selected pushbroom slit, which may be much wider.
    row_pitch = float(y_edges[1] - y_edges[0])
    line_optics = replace(o, camera_line_width_mm=row_pitch)
    rows = [camera_bin(p, v, w, domain, line_optics, float(y), x_edges)[0]
            for y in y_mm]
    scan_reconstruction = np.asarray(rows)

    scale = float(o.photons_per_line)
    area = area / scale
    scan_reconstruction = scan_reconstruction / scale
    diff = scan_reconstruction - area
    return {
        'area_image': area,
        'line_readout_image': scan_reconstruction,
        'difference': diff,
        'x_mm': x_mm,
        'y_mm': y_mm,
        'diagnostics': {
            'engine': 'MCX via PMCX',
            'comparison': 'same photons, fixed illumination and egg',
            'actual_egg_motion': False,
            'n_mcx_runs': 1,
            'photons_launched': int(o.photons_per_line),
            'exit_photons': len(p),
            'accepted_area_photons': n_area,
            'matched_slit_width_mm': row_pitch,
            'max_abs_difference': float(np.max(np.abs(diff))),
            'relative_l1_difference': float(np.sum(np.abs(diff)) / max(np.sum(area), 1e-15)),
        },
    }
