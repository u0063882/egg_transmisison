"""Batch CLI to run a scan without Streamlit.

python -m eggtrans.cli --photons 12000 --lines 64 --mode "Transverse line" --output results/demo
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import Geometry, Optics, MATERIALS_800, MODE_NAMES, simulate, projected_masks
from .spectra import load_spectra_csv, at_wavelength


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,help='JSON with geometry, optics, materials and wavelength_nm')
    p.add_argument('--mode',choices=MODE_NAMES,default=None)
    p.add_argument('--photons',type=int,default=None)
    p.add_argument('--lines',type=int,default=None)
    p.add_argument('--pixels',type=int,default=None)
    p.add_argument('--wavelength',type=float,default=800.)
    p.add_argument('--spectra-csv',type=Path,help='Measured optical coefficients CSV')
    p.add_argument('--output',type=Path,default=Path('results/egg_scan'))
    args=p.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    g=Geometry(**cfg.get('geometry',{}))
    o=Optics(**cfg.get('optics',{}))
    if args.mode: o.mode=args.mode
    if args.photons: o.photons_per_line=args.photons
    if args.lines: o.scan_lines=args.lines
    if args.pixels: o.camera_pixels=args.pixels
    wl=cfg.get('wavelength_nm',args.wavelength)
    mat=np.asarray(cfg.get('materials',MATERIALS_800),dtype=float)
    if args.spectra_csv:
        mat=at_wavelength(load_spectra_csv(args.spectra_csv.read_text(encoding='utf-8')),wl)
    res=simulate(g,o,mat)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    metadata={'geometry':asdict(g),'optics':asdict(o),'materials':mat.tolist(),
              'wavelength_nm':wl,'diagnostics':res['diagnostics']}
    masks=projected_masks(g,res['x_mm'],res['scan_y_mm'])
    np.savez_compressed(str(args.output)+'.npz',image=res['image'],x_mm=res['x_mm'],scan_y_mm=res['scan_y_mm'],
                        yolk_truth=masks['yolk'],blood_truth=masks['blood'],egg_truth=masks['egg'],metadata_json=json.dumps(metadata))
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    extent=(res['x_mm'][0],res['x_mm'][-1],res['scan_y_mm'][0],res['scan_y_mm'][-1])
    axes[0].imshow(np.arcsinh(res['image']/max(np.max(res['image'])*.03,1.e-12)),origin='lower',extent=extent,cmap='inferno',aspect='equal')
    axes[0].set_title('Simulated transmitted light (asinh visibility)')
    truth=masks['egg'].astype(int)+masks['yolk'].astype(int)+masks['blood'].astype(int)*g.include_blood
    axes[1].imshow(truth,origin='lower',extent=extent,aspect='equal',cmap='viridis',vmin=0,vmax=3)
    axes[1].set_title('True projected materials (not observable)')
    for ax in axes:
        ax.set_xlabel('Across egg x (mm)');ax.set_ylabel('Translation y (mm)')
    fig.savefig(str(args.output)+'.png',dpi=180)
    fig.savefig(str(args.output)+'.svg')
    plt.close(fig)
    (args.output.parent/(args.output.name+'.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Created {args.output}.npz / .png / .svg / .json')
    print(json.dumps(res['diagnostics'],indent=2))

if __name__=='__main__': main()