"""Save inspectable PMCX scan inputs; uses exactly the same source builder as UI.

This is *not* a second simulation engine. The saved `.npz` contains the MCX
voxel volume; each JSON file is a Python PMCX configuration with `vol`
replaced by a pointer to the .npz volume (not MCX executable JSON format).

Run: python -m eggtrans.mcx_export --mode Spot --scan-lines 5
"""
from __future__ import annotations
import argparse,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from .core import Geometry,Optics,MATERIALS_800,MODE_NAMES
from .mcx_engine import build_egg_volume,mcx_config


def export(g:Geometry|None=None,o:Optics|None=None,materials:np.ndarray|None=None,
           output:str|Path='results/mcx_inputs',voxel_mm:float|None=None,
           scan_lines:int|None=None)->list[Path]:
    g=g or Geometry();o=o or Optics()
    if voxel_mm is not None:o=replace(o,voxel_mm=voxel_mm)
    if scan_lines is not None:o=replace(o,scan_lines=scan_lines)
    m=np.asarray(MATERIALS_800 if materials is None else materials,dtype=float)
    d=build_egg_volume(g,o,m)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'mcx_domain.npz',vol=d.vol,prop=d.props,origin_mm=np.array(d.origin_mm),voxel_mm=d.grid_mm)
    y=np.linspace(-g.egg_length_mm*.54,g.egg_length_mm*.54,o.scan_lines)
    paths=[]
    for i,sy in enumerate(y):
        c=mcx_config(d,o,float(sy),o.seed+i)
        del c['vol']
        c['volume_npz']='mcx_domain.npz';c['source_scan_y_mm']=float(sy)
        path=out/f'pmcx_scan_{i:03d}.json'
        path.write_text(json.dumps(c,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x))
        paths.append(path)
    (out/'metadata.json').write_text(json.dumps({'geometry':asdict(g),'optics':asdict(o),
        'note':'PMCX diagnostic config, not directly consumable by mcx -f; simulation is through pmcx.mcxlab.'},indent=2))
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=MODE_NAMES,default='Spot')
    p.add_argument('--voxel-mm',type=float,default=1)
    p.add_argument('--photons',type=int,default=100000)
    p.add_argument('--scan-lines',type=int,default=5)
    p.add_argument('--output',type=Path,default=Path('results/mcx_inputs'))
    a=p.parse_args()
    paths=export(o=Optics(mode=a.mode,voxel_mm=a.voxel_mm,photons_per_line=a.photons,
                           scan_lines=a.scan_lines),output=a.output)
    print(f'Saved {len(paths)} PMCX configuration(s) and voxel volume in {a.output}')

if __name__=='__main__':main()
