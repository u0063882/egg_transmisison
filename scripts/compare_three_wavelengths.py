"""Reproducible, explicitly *hypothetical* three-band illumination sensitivity study.

The 800 nm optical coefficients for albumen and yolk are the only literature-
anchored material values in the current simulator. The 580/1200 nm settings
are labeled absorption-only sensitivity scenarios, NOT measured BOP spectra.

Run from repo root:
    python scripts/compare_three_wavelengths.py --photons 24000 --lines 112 --pixels 128

A constant photon-packet budget is used *per scan line and per geometry*.
Absolute flux cannot be compared to a calibrated illumination system until
irradiance, camera throughput, shell, blood properties, and spectral data are
measured. The launcher approximates a stationary camera / source scanning a
moving egg through equivalent coordinates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggtrans.core import Geometry, MATERIALS_800, Optics, projected_masks, simulate

# This table is a HYPOTHESIS, not measured wavelength-dependent BOP data.
# We change absorption ONLY, so changes can be attributed unambiguously to
# the specified assumptions instead of simultaneously changing scattering,
# anisotropy, refractive indices and setup. Coefficients at 800 nm for shell
# and blood remain illustrative and so are all values derived from them.
ABSORPTION_MULTIPLIERS = {
    580: {'Shell': 1.25, 'Albumen': 1.0, 'Yolk': 1.8, 'Blood spot': 4.0},
    800: {'Shell': 1.0, 'Albumen': 1.0, 'Yolk': 1.0, 'Blood spot': 1.0},
    1200: {'Shell': 1.20, 'Albumen': 5.0, 'Yolk': 4.0, 'Blood spot': 0.90},
}

# Keep the same *physical* line width when isolating divergence.
BASE = Optics(
    mode='Transverse line', line_length_mm=48.0, line_width_mm=2.0,
    spot_diameter_mm=6.0, source_offset_y_mm=0.0,
    divergence_x_deg=8.0, divergence_y_deg=3.0,
    camera_half_angle_deg=40.0, camera_line_width_mm=4.0,
    max_scattering_events=900, seed=711,
)
CONFIGS = (
    ('line', 'Transverse line', 'Transverse line', {}),
    ('collimated', 'Same line, 1° divergence', 'Transverse line', {
        'divergence_x_deg': 1.0, 'divergence_y_deg': 1.0,
    }),
    ('spot', '6 mm spot', 'Spot', {}),
    ('offset', 'Line offset +5 mm', 'Transverse line', {'source_offset_y_mm': 5.0}),
    ('broad', '72 mm backlight', 'Broad backlight', {'line_length_mm': 72.0}),
)


def material_scenario(wavelength_nm: int) -> np.ndarray:
    """Absorption-only what-if scenario based on the repo's 800 nm baseline."""
    if wavelength_nm not in ABSORPTION_MULTIPLIERS:
        raise ValueError('Only deliberately documented scenarios are available')
    m = np.array(MATERIALS_800, copy=True)
    for index, name in enumerate(('Shell', 'Albumen', 'Yolk', 'Blood spot'), 1):
        m[index, 0] *= ABSORPTION_MULTIPLIERS[wavelength_nm][name]
    return m


def _save_png(fig, dest: Path, dpi: int = 130) -> None:
    fig.savefig(dest, dpi=dpi, facecolor='white')
    plt.close(fig)


def run(output: Path, photons: int, lines: int, pixels: int, seeds: tuple[int,...] = (711,)) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if not (photons > 0 and lines > 1 and pixels > 1):
        raise ValueError('Invalid scan dimensions')
    geometry = Geometry()
    results = {}
    rows = []
    for wl in ABSORPTION_MULTIPLIERS:
        mat = material_scenario(wl)
        results[wl] = {}
        for code, label, mode, overrides in CONFIGS:
            # One seed for the standard comparison. Optional multi-seed
            # average can be enabled for convergence analyses.
            images = []
            diagnostics = []
            for seed in seeds:
                opts = replace(BASE, mode=mode, photons_per_line=photons,
                    scan_lines=lines, camera_pixels=pixels, seed=seed, **overrides)
                tic = time.perf_counter()
                result = simulate(geometry, opts, mat)
                images.append(result['image'])
                diagnostics.append(result['diagnostics'])
                print(f'{wl:4d} nm | {code:11s} | seed={seed} '
                      f'| {time.perf_counter()-tic:5.1f} s', flush=True)
            im = np.mean(images, axis=0)
            dd = {key: float(np.mean([z[key] for z in diagnostics])) for key in diagnostics[0]}
            total = dd['launched_photons']
            info = {
                'wavelength_nm': wl, 'config': code, 'label': label,
                'optics': asdict(opts), 'detected_rays': int(round(dd['detected_ray_count'])),
                'relative_detected_flux_pct': 100*dd['weighted_camera_signal']/total,
                'zero_volume_scatter_share_pct': 100*dd['weighted_ballistic_signal']/max(dd['weighted_camera_signal'],1.e-30),
                'event_limit_termination_pct': 100*dd['max_event_terminated']/total,
            }
            rows.append(info)
            results[wl][code] = im
            x, y = result['x_mm'], result['scan_y_mm']
            masks = projected_masks(geometry, x, y)
            np.savez_compressed(output/f'{code}_{wl}nm.npz',
                image=im.astype(np.float32), x_mm=x, y_mm=y,
                ground_truth_yolk=masks['yolk'], ground_truth_blood=masks['blood'],
                materials=mat, optics_json=json.dumps(asdict(opts)))
    # Clarity first: use per-panel *relative* scale to compare shapes, never
    # imply intensity equality across configurations or wavelengths.
    fig, axs = plt.subplots(3, len(CONFIGS)-1, figsize=(13.4, 11.1),
                            sharex=True,sharey=True, layout='constrained')
    fig.suptitle('Egg transmission — 580 / 800 / 1200 nm | hypothetical absorption scenarios',
                 fontsize=15,weight='bold')
    ex = [x[0],x[-1],y[0],y[-1]]
    for ri,wl in enumerate(ABSORPTION_MULTIPLIERS):
        for ci, (code,label,_,_) in enumerate(CONFIGS[:-1]):
            ax = axs[ri,ci]
            data = gaussian_filter(results[wl][code], 1.05)
            peak = max(float(np.quantile(data, .995)), 1.e-25)
            # Common asinh transfer function *after* per-panel scaling.
            normed = np.arcsinh(data/(0.035*peak))/np.arcsinh(1/0.035)
            ax.imshow(normed,origin='lower',extent=ex,interpolation='nearest',
                      aspect='equal',cmap='inferno',vmin=0,vmax=1)
            ax.contour(x,y,masks['yolk'],levels=[.5],colors='cyan',
                       linewidths=.9,linestyles='--')
            ax.contour(x,y,masks['blood'],levels=[.5],colors='#6ee7b7',
                       linewidths=1,linestyles=':')
            if ri == 0:
                ax.set_title(label, fontsize=11,weight='bold')
            if ci == 0:
                ax.set_ylabel(f'{wl} nm\ny (mm)',fontsize=11)
            if ri == 2:
                ax.set_xlabel('x (mm)')
    _save_png(fig,output/'comparison_patterns.png',155)

    # Signal levels show how much packet-weight survives, but NOT calibrated
    # irradiance and NOT trustworthy relative camera performance across bands.
    fig,ax = plt.subplots(figsize=(11.5,4.1),layout='constrained')
    loc = np.arange(len(CONFIGS))
    width = .24
    for j,wl in enumerate(ABSORPTION_MULTIPLIERS):
        rr=[next(r for r in rows if r['wavelength_nm']==wl and r['config']==c[0]) for c in CONFIGS]
        ax.bar(loc+(j-1)*width,[r['relative_detected_flux_pct'] for r in rr],width,
               label=f'{wl} nm')
    ax.set_xticks(loc,[c[1] for c in CONFIGS],rotation=13,ha='right')
    ax.set_ylabel('Weighted detected flux / launched photon packets (%)')
    ax.legend(frameon=False,ncol=3)
    ax.set_title('Within-model relative throughput (not irradiance-matched)',fontsize=12)
    ax.grid(axis='y',alpha=.18)
    _save_png(fig,output/'flux_comparison.png',165)

    summary={
        'status':'unvalidated; illustrative absorption scenarios at 580 and 1200 nm',
        'model':'custom Numba Monte Carlo (not independently benchmarked)',
        'photons_per_line':photons, 'scan_lines':lines,'camera_pixels':pixels,
        'seeds':list(seeds), 'baseline_materials_800':MATERIALS_800.tolist(),
        'absorption_multipliers':ABSORPTION_MULTIPLIERS,
        'scenario_materials':{str(w):material_scenario(w).tolist() for w in ABSORPTION_MULTIPLIERS},
        'geometry':asdict(geometry), 'runs':rows,
    }
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--photons',type=int,default=24000)
    p.add_argument('--lines',type=int,default=112)
    p.add_argument('--pixels',type=int,default=128)
    p.add_argument('--output',type=Path,default=Path('reports/three_wavelengths'))
    p.add_argument('--seeds',type=int,nargs='+',default=[711])
    args=p.parse_args()
    run(args.output,args.photons,args.lines,args.pixels,tuple(args.seeds))
    print('Saved',args.output,flush=True)


if __name__ == '__main__':
    main()