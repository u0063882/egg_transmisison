"""Run padding convergence using established PMCX/MCX transport.

Dry-run estimates sizes without MCX or GPU; real runs execute MCX.
Examples:
 python scripts/run_domain_convergence.py --dry-run --mode 'Transverse line' --voxel .35
 python scripts/run_domain_convergence.py --camera Area --padding 6 10 14 --photons 300000
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eggtrans.core import Geometry, Optics, MATERIALS_800, simulate
from eggtrans.mcx_engine import domain_plan


def run(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera',choices=('Area','Pushbroom'),default='Area')
    parser.add_argument('--mode',choices=('Transverse line','Spot','Longitudinal line','Broad backlight'),default='Transverse line')
    parser.add_argument('--padding',type=float,nargs='+',default=[6.,10.,14.,20.])
    parser.add_argument('--voxel',type=float,default=.5)
    parser.add_argument('--line-length',type=float,default=60.)
    parser.add_argument('--photons',type=int,default=300000)
    parser.add_argument('--lines',type=int,default=12)
    parser.add_argument('--pixels',type=int,default=48)
    parser.add_argument('--area-rows',type=int,default=48)
    parser.add_argument('--host-budget-mib',type=float,default=1024.)
    parser.add_argument('--output',type=Path,default=Path('reports/padding_convergence.json'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if any(v<=0 for v in args.padding):parser.error('All padding values must be positive.')
    g=Geometry()
    o=Optics(camera_mode=args.camera,mode=args.mode,voxel_mm=args.voxel,
             line_length_mm=args.line_length,photons_per_line=args.photons,
             scan_lines=args.lines,camera_pixels=args.pixels,area_rows=args.area_rows,
             domain_build_budget_mib=args.host_budget_mib)
    plans=[]
    images=[]
    for pad in sorted(set(args.padding)):
        optics=replace(o,domain_padding_mm=pad)
        plan=domain_plan(g,optics,MATERIALS_800,enforce_budget=False)
        record={'padding_mm':pad,'domain':asdict(plan),
                'budget_exceeded':plan.host_build_estimate_mib>args.host_budget_mib}
        if not args.dry_run:
            result=simulate(g,optics,MATERIALS_800)
            images.append(result['image'])
            record['mcx_diagnostics']=result['diagnostics']
            record['total_camera_signal']=float(result['image'].sum())
        plans.append(record)
    if not args.dry_run:
        reference=images[-1]
        denom=max(float(np.sum(np.abs(reference))),1e-15)
        for record,image in zip(plans,images):
            record['relative_l1_vs_largest_box']=float(np.sum(np.abs(image-reference))/denom)
            record['relative_flux_difference_vs_largest_box']=float(abs(image.sum()-reference.sum())/max(reference.sum(),1e-15))
    output={'dry_run':args.dry_run,'comparison':'padding at fixed voxel size',
            'notes':'Actual MCX GPU results are required for convergence. Photon noise and side-exit losses also affect the differences.',
            'results':plans}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,default=str),encoding='utf-8')
    print(json.dumps(output,indent=2,default=str))
    return output


if __name__=='__main__':run()
