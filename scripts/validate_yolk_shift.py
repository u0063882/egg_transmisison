"""Falsification controls for egg-transmission MC: remove and displace the yolk.

Run from repository root:
    python scripts/validate_yolk_shift.py --photons 16000 --lines 112 --pixels 128 --seeds 881 882

At 800 nm only. Two light sources; a *single* common raw display scale per
source, no ground-truth overlays on camera images. Also plots signed
no-yolk minus yolk images, documenting lost/gained signal and MC uncertainty.
The shell/blood settings are illustrative and the solver is unbenchmarked.
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggtrans.core import Geometry, Optics, MATERIALS_800, projected_masks, simulate

SCENARIOS = {
    'none': ('No yolk', {'include_yolk': False}),
    'center': ('Centred (0, 0)', {'yolk_offset_x_mm': 0, 'yolk_offset_y_mm': 0}),
    'left': ('Shifted (−5, −7)', {'yolk_offset_x_mm': -5, 'yolk_offset_y_mm': -7}),
    'right': ('Shifted (+5, −7)', {'yolk_offset_x_mm': 5, 'yolk_offset_y_mm': -7}),
}
MODES = {
    'line': ('Transverse line', 'Transverse line'),
    'spot': ('6 mm spot', 'Spot'),
}
BASE_GEOMETRY = Geometry(include_blood=False, yolk_offset_x_mm=0.0, yolk_offset_y_mm=0.0, yolk_offset_z_mm=-1.0)
BASE_OPTICS = Optics(photons_per_line=16000, scan_lines=112, camera_pixels=128,
                     camera_half_angle_deg=40., camera_line_width_mm=4.,
                     line_length_mm=48.,line_width_mm=2., spot_diameter_mm=6.,
                     divergence_x_deg=8., divergence_y_deg=3., max_scattering_events=900)


def _centroid(delta: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> dict:
    # Conceptual negative-transmission signal, NOT a segmentation or a true boundary.
    smooth = gaussian_filter(delta, 1.25)
    positive = np.maximum(smooth, 0)
    total = float(positive.sum())
    if total <= 1e-30:
        return {'x_mm': None, 'y_mm': None, 'integrated_loss': total}
    return {'x_mm': float(np.sum(positive * xx[None, :]) / total),
            'y_mm': float(np.sum(positive * yy[:, None]) / total),
            'integrated_loss': total}


def run(output: Path, photons: int = 16000, lines: int = 112, pixels: int = 128,
        seeds: tuple[int, ...] = (881, 882)) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if photons < 1 or lines < 2 or pixels < 2 or len(seeds) < 1:
        raise ValueError('Invalid experiment dimensions / seeds')
    imgs = {}
    meta = {'wavelength_nm': 800, 'optical_parameters': MATERIALS_800.tolist(),
            'reference': 'published coefficients only for albumen/yolk at 800 nm; shell hypothetical',
            'geometry': asdict(BASE_GEOMETRY), 'optics': asdict(BASE_OPTICS),
            'photons_per_line': photons, 'scan_lines': lines, 'camera_pixels': pixels,
            'seeds': list(seeds), 'scenarios': {}, 'metrics': {}, 'sources': MODES}
    for source,(display_name,mode) in MODES.items():
        imgs[source] = {}
        for label, (name, gprops) in SCENARIOS.items():
            g = replace(BASE_GEOMETRY, **gprops)
            o = replace(BASE_OPTICS, mode=mode,photons_per_line=photons,
                        scan_lines=lines,camera_pixels=pixels)
            independent = []
            diagnostics = []
            for seed in seeds:
                tic = time.perf_counter()
                run_result = simulate(g, replace(o, seed=seed), MATERIALS_800)
                independent.append(run_result['image'])
                diagnostics.append(run_result['diagnostics'])
                print(f'{source:5s} | {label:6s} | seed {seed} | {time.perf_counter()-tic:.1f}s', flush=True)
            mean = np.mean(independent,axis=0)
            imgs[source][label] = mean
            x,y=run_result['x_mm'],run_result['scan_y_mm']
            np.savez_compressed(output / f'{source}_{label}.npz',
                image_mean=mean.astype('float32'),
                image_per_seed=np.array(independent,dtype='float32'),
                x_mm=x,y_mm=y, materials=MATERIALS_800,
                geometry_json=json.dumps(asdict(g)), optics_json=json.dumps(asdict(o)),
                seeds=np.array(seeds))
            meta['scenarios'][f'{source}/{label}'] = {
                'geometry': asdict(g),
                'total_flux_mean':float(mean.sum()),
                'detected_ray_count_per_seed': [d['detected_ray_count'] for d in diagnostics],
                'event_limit_terminations_per_seed': [d['max_event_terminated'] for d in diagnostics],
            }
        ref=imgs[source]['none']
        for label in ('center','left','right'):
            delta = ref - imgs[source][label]
            meta['metrics'][f'{source}/{label}'] = {
                'positive_effect_centroid':_centroid(delta,x,y),
                'signed_integrated_change':float(delta.sum()),
                'fraction_flux_drop_vs_no_yolk':float(delta.sum()/ref.sum()),
            }
        # Mirror symmetry: compare the *changes caused by* left and right yolks.
        dl=gaussian_filter(ref - imgs[source]['left'],1.5)
        dr=gaussian_filter(ref - imgs[source]['right'],1.5)[:, ::-1]
        meta['metrics'][f'{source}/mirror_test'] = {
            'corr_reflected_differential':float(np.corrcoef(dl.ravel(),dr.ravel())[0,1])}

    # Camera output: no contours, no arbitrary individual normalization.
    fig, axes=plt.subplots(2,4,figsize=(14.2,7.6),layout='constrained',sharex=True,sharey=True)
    fig.suptitle('800 nm | Raw Monte Carlo camera images (NO geometry overlays)', fontsize=15,weight='bold')
    extent=[x[0],x[-1],y[0],y[-1]]
    for ri,(source,(pretty,_)) in enumerate(MODES.items()):
        cap=max(float(np.quantile(img,0.995)) for img in imgs[source].values())
        for ci,(label,(heading,_)) in enumerate(SCENARIOS.items()):
            ax=axes[ri,ci]
            image=imgs[source][label]
            im=ax.imshow(image,cmap='magma',origin='lower',extent=extent,
                         interpolation='nearest',vmin=0,vmax=cap,aspect='equal')
            if ri==0: ax.set_title(heading,fontsize=12)
            if ci==0: ax.set_ylabel(pretty+'\nTranslation y (mm)')
            if ri==1: ax.set_xlabel('Transverse x (mm)')
        fig.colorbar(im,ax=axes[ri,:],shrink=.64,pad=.012, label='Relative detected flux / launched packet')
    fig.savefig(output/'camera_raw_no_overlays.png',dpi=175,facecolor='white')
    plt.close(fig)

    # Subtract no-yolk case to isolate the effect of yolk optical coefficients.
    fig, axes=plt.subplots(2,3,figsize=(12.0,7.6),layout='constrained',sharex=True,sharey=True)
    fig.suptitle('800 nm | Counterfactual effect = camera(no yolk) − camera(with yolk)',fontsize=14,weight='bold')
    for ri,(source,(pretty,_)) in enumerate(MODES.items()):
        differences={label:gaussian_filter(imgs[source]['none']-imgs[source][label],1.1)
                     for label in ('center','left','right')}
        bound=max(float(np.quantile(np.abs(v),.992)) for v in differences.values())
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.,vmax=bound)
        for ci,label in enumerate(('center','left','right')):
            ax=axes[ri,ci]
            im=ax.imshow(differences[label],cmap='RdBu_r',norm=norm,origin='lower',extent=extent,
                         interpolation='nearest',aspect='equal')
            if ri==0:ax.set_title(SCENARIOS[label][0],fontsize=12)
            if ci==0:ax.set_ylabel(pretty+'\nTranslation y (mm)')
            if ri==1:ax.set_xlabel('Transverse x (mm)')
        fig.colorbar(im,ax=axes[ri,:],shrink=.7,pad=.012, label='Signed relative flux change')
    fig.savefig(output/'yolk_effect_no_overlays.png',dpi=175,facecolor='white')
    plt.close(fig)

    # Geometry intentionally shown separately; never blended with camera photons.
    fig,axes=plt.subplots(1,4,figsize=(13,3.8),layout='constrained')
    for ax,(name,(heading,gprops)) in zip(axes,SCENARIOS.items()):
        g=replace(BASE_GEOMETRY,**gprops)
        masks=projected_masks(g,x,y)
        ax.imshow(np.where(masks['egg'],0.35,0)+np.where(masks['yolk'],0.65,0),
                  extent=extent,origin='lower',vmin=0,vmax=1,cmap='cividis')
        ax.set_title(heading);ax.set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    fig.suptitle('Geometry ground truth ONLY — not a simulated camera image')
    fig.savefig(output/'geometry_truth_only.png',dpi=175,facecolor='white')
    plt.close(fig)

    (output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--photons',type=int,default=16000)
    ap.add_argument('--lines',type=int,default=112)
    ap.add_argument('--pixels',type=int,default=128)
    ap.add_argument('--seeds',type=int,nargs='+',default=[881,882])
    ap.add_argument('--output',type=Path,default=Path('reports/yolk_displacement_800nm'))
    a=ap.parse_args()
    run(a.output,a.photons,a.lines,a.pixels,tuple(a.seeds))
