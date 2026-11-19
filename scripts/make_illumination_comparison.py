"""Reproduce the high-resolution 800 nm illumination comparison.

Run: python scripts/make_illumination_comparison.py --photons 45000 --lines 112 --pixels 128
The source/receiver and shell/blood parameters remain *illustrative*.
"""
from __future__ import annotations
import argparse, json, time, sys
from dataclasses import asdict, replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from eggtrans.core import Geometry, Optics, MATERIALS_800, simulate, projected_masks

BASE=Optics(line_length_mm=48,line_width_mm=2,spot_diameter_mm=6,
           divergence_x_deg=8,divergence_y_deg=3,camera_half_angle_deg=40,
           camera_line_width_mm=4,max_scattering_events=900,seed=711)
CASES=[
 ('A','Transverse line','Transverse line',{}),
 ('B','Spot','Spot',{}),
 ('C','Longitudinal line','Longitudinal line',{}),
 ('D','Broad backlight','Broad backlight',{'line_length_mm':110}),
 ('E','1 mm collimated line','Transverse line',{'line_width_mm':1,'divergence_x_deg':1,'divergence_y_deg':1}),
 ('F','Line offset +5 mm','Transverse line',{'source_offset_y_mm':5}),
 ('G','Wide stripe 10 mm','Transverse line',{'line_width_mm':10}),
 ('H','Camera acceptance 10°','Transverse line',{'camera_half_angle_deg':10}),
]
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--photons',type=int,default=45000)
 p.add_argument('--lines',type=int,default=112)
 p.add_argument('--pixels',type=int,default=128)
 p.add_argument('--output',type=Path,default=Path('reports/illumination_800nm'))
 args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
 g=Geometry(); results={}; summary=[]
 for code,label,mode,deltas in CASES:
  opt=replace(BASE,mode=mode,photons_per_line=args.photons,scan_lines=args.lines,camera_pixels=args.pixels,**deltas)
  print('Running',code,label,flush=True); t=time.perf_counter()
  r=simulate(g,opt,MATERIALS_800); di=r['diagnostics']; n=di['launched_photons']
  vals={'case':code,'name':label,'mode':mode,'optics':asdict(opt),'elapsed_sec':round(time.perf_counter()-t,1),
    'detected_rays':di['detected_ray_count'],
    'relative_flux_pct':round(100*di['weighted_camera_signal']/n,5),
    'zero_scatter_share_pct':round(100*di['weighted_ballistic_signal']/max(di['weighted_camera_signal'],1e-20),3),
    'terminated_at_event_limit_pct':round(100*di['max_event_terminated']/n,3)}
  print(vals,flush=True);results[code]=r;summary.append(vals)
  masks=projected_masks(g,r['x_mm'],r['scan_y_mm'])
  np.savez_compressed(args.output/f'case_{code}.npz',image=r['image'],x_mm=r['x_mm'],y_mm=r['scan_y_mm'],
    egg=masks['egg'],yolk=masks['yolk'],blood=masks['blood'],
    model_config_json=json.dumps({'geometry':asdict(g),'optics':asdict(opt), 'material_matrix':MATERIALS_800.tolist()}))
 xs=results['A']['x_mm'];ys=results['A']['scan_y_mm'];extent=(xs[0],xs[-1],ys[0],ys[-1]);masks=projected_masks(g,xs,ys)
 images=[results[c]['image'] for c,_,_,_ in CASES]
 cap=max(np.quantile(im,.998) for im in images)
 for style in ['shared_linear','relative_visibility']:
  fig,axes=plt.subplots(2,4,figsize=(14.3,8.0),sharex=True,sharey=True,layout='constrained')
  for ax,(code,label,_,_) in zip(axes.flat,CASES):
   im=results[code]['image']
   if style=='relative_visibility':
    sm=gaussian_filter(im,1.25); m=max(float(np.quantile(sm,.995)),1e-24)
    vis=np.arcsinh(sm/(.03*m)); vtop=float(np.arcsinh(1/.03))
   else:vis=im;vtop=cap
   h=ax.imshow(vis,origin='lower',extent=extent,aspect='equal',cmap='inferno',vmin=0,vmax=vtop,interpolation='none')
   if style=='relative_visibility':
    ax.contour(xs,ys,masks['yolk'],levels=[.5],colors=['cyan'],linestyles='--',linewidths=.8)
    ax.contour(xs,ys,masks['blood'],levels=[.5],colors=['lime'],linestyles=':',linewidths=.8)
   ax.set_title(f'{code}. {label}',fontsize=11);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
  if style=='shared_linear':
   fig.colorbar(h,ax=axes.ravel().tolist(),shrink=.75,pad=.01,label='Detected weight / launched packet')
  fig.suptitle(('Shared linear flux scale' if style=='shared_linear' else 'Per-image normalized visibility; smoothed σ=1.25 pixels')+
               f' | 800 nm, {args.lines}×{args.pixels} pixels, {args.photons:,} packets/line',fontsize=13)
  fig.savefig(args.output/f'comparison_{style}.png',dpi=145);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12.5,4.4),layout='constrained')
 for code,label,_,_ in CASES:
  sm=gaussian_filter(results[code]['image'],1.25)
  horizontal=sm[len(ys)//2,:]; vertical=sm[:,len(xs)//2]
  axes[0].plot(xs,horizontal/max(float(horizontal.max()),1e-25),label=f'{code} {label}')
  axes[1].plot(ys,vertical/max(float(vertical.max()),1e-25),label=f'{code} {label}')
 for yy in [g.yolk_offset_x_mm-g.yolk_radius_x_mm,g.yolk_offset_x_mm+g.yolk_radius_x_mm]:axes[0].axvline(yy,ls=':',c='.4',lw=1)
 for yy in [g.yolk_offset_y_mm-g.yolk_radius_y_mm,g.yolk_offset_y_mm+g.yolk_radius_y_mm]:axes[1].axvline(yy,ls=':',c='.4',lw=1)
 axes[0].set_title('Mid-scan across egg (x)');axes[1].set_title('Centre-pixel along movement (y)')
 for ax in axes:ax.set_ylim(0,1.05);ax.set_ylabel('Normalized intensity');ax.grid(alpha=.15)
 axes[0].set_xlabel('x (mm)');axes[1].set_xlabel('y (mm)');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.21),ncol=2,fontsize=7)
 fig.savefig(args.output/'profiles.png',dpi=145,bbox_inches='tight');plt.close(fig)
 (args.output/'summary.json').write_text(json.dumps({'wavelength_nm':800,'source':'custom Numba MC prototype',
  'material_matrix_800':MATERIALS_800.tolist(),'geometry':asdict(g),'parameters':{'photons':args.photons,'lines':args.lines,'pixels':args.pixels},
  'results':summary},indent=2)+'\n',encoding='utf-8')
 print('DONE',args.output,flush=True)
if __name__=='__main__':main()