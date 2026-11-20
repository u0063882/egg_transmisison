"""Controlled MCX area-versus-virtual-line comparison under a fixed light field.

This compares camera sampling *in a shared stationary optical scene*. The
actual moving-egg pushbroom camera is a different experiment when its source
is narrow. MCX/PMCX performs all photon transport; no optical physics here.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import numpy as np

from .core import Geometry, Optics, MATERIALS_800
from .mcx_engine import (backend_status, build_egg_volume, camera_bin,
                         camera_bin_area, mcx_config, top_face_photons, photon_history_mask)


def compare_matched_acquisitions(
    geometry: Geometry, optics: Optics, materials: np.ndarray | None = None,
    *, rows: int = 12, backend=None,
) -> dict:
    """One area MCX exposure vs repeated narrow camera rows, same broad source.

    The broad backlight is fixed *relative to the egg* throughout. Area and
    sequential virtual-line readouts use identical launch geometry, detector
    acceptance, and y pixel pitch, so per-launched-photon images should agree
    statistically. This does NOT reproduce the lab-fixed moving-egg source
    scenario when a narrow spot/line follows the scanning camera slit.
    """
    mat = np.asarray(MATERIALS_800 if materials is None else materials)
    if rows < 2:
        raise ValueError('At least two comparison rows are needed.')
    broad_width = max(float(geometry.egg_width_mm), float(geometry.egg_length_mm)) + 16.
    o = replace(optics, mode='Broad backlight', camera_mode='Area',
                line_length_mm=broad_width, area_rows=int(rows))
    if backend is None:
        try:
            import pmcx as backend
        except (ImportError, OSError) as exc:
            raise RuntimeError('PMCX is required to run the matched comparison.') from exc
        ready, message = backend_status()
        if not ready:
            raise RuntimeError(message)
    domain = build_egg_volume(geometry, o, mat)
    x_edges = np.linspace(-geometry.egg_width_mm*.56,
                          geometry.egg_width_mm*.56, o.camera_pixels+1)
    y_edges = np.linspace(-geometry.egg_length_mm*.54,
                          geometry.egg_length_mm*.54, rows+1)
    x = (x_edges[1:]+x_edges[:-1])*0.5
    y = (y_edges[1:]+y_edges[:-1])*0.5
    pitch=float(y_edges[1]-y_edges[0])
    lo=replace(o,camera_line_width_mm=pitch)

    def exits(seed):
        # Fixed source at y = offset; never move the source for this controlled test.
        cfg=mcx_config(domain,o,0.,seed)
        result=backend.mcxlab(cfg)
        if not isinstance(result,dict) or 'detp' not in result:
            raise RuntimeError('MCX returned no detp; verify exit-face detection.')
        p,v,w=top_face_photons(result['detp'],domain.props,backend,unitinmm=domain.grid_mm)
        keep=photon_history_mask(result['detp'],len(p),o,domain.grid_mm,
                                 n_materials=domain.props.shape[0]-1)
        return p[keep],v[keep],w[keep]

    p,v,w = exits(max(1,int(o.seed)))
    area, n_area = camera_bin_area(p,v,w,domain,o,y_edges,x_edges)
    image_rows = np.zeros_like(area)
    counts=[]
    for j, pos in enumerate(y):
        p,v,w=exits(max(1,(int(o.seed)+j+1)%2147483647))
        row,n = camera_bin(p,v,w,domain,lo,float(pos),x_edges)
        image_rows[j,:]=row
        counts.append(n)
    area /= float(o.photons_per_line)
    image_rows /= float(o.photons_per_line)
    difference=image_rows-area
    return {
        'area_image':area,'virtual_pushbroom_image':image_rows,
        'difference':difference,'x_mm':x,'y_mm':y,
        'diagnostics':{
            'simulation':'fixed broad illumination, sequential virtual-line detector',
            'transport':'MCX / PMCX, no optical transport code in Python',
            'actual_moving_egg_scanner':False,
            'source_mode':o.mode,'source_size_mm':broad_width,
            'number_of_runs':int(rows+1),
            'photons_per_mcx_run':int(o.photons_per_line),
            'total_photons_launched':int((rows+1)*o.photons_per_line),
            'accepted_area_photons':int(n_area),
            'accepted_line_photons':int(sum(counts)),
            'row_pitch_mm':pitch,
            'area_weighted_signal':float(area.sum()),
            'line_weighted_signal':float(image_rows.sum()),
            'integrated_signal_relative_difference':float(
                abs(image_rows.sum()-area.sum())/max(area.sum(),1e-15)),
            'relative_l1_image_discrepancy':float(
                np.sum(np.abs(difference))/max(np.sum(area),1e-15)),
            'domain_plan':asdict(domain.plan),
        },
    }
