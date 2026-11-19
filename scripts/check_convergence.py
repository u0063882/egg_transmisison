"""Independent-seed check of 800 nm baseline versus angularly collimated line.

Run AFTER scripts/compare_three_wavelengths.py creates the 800 nm NPZ files.
Use exactly the parameters in the published 2026-11-19 note.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import numpy as np
from dataclasses import replace
from scipy.ndimage import gaussian_filter
from scripts.compare_three_wavelengths import BASE,material_scenario
from eggtrans.core import Geometry,simulate
D=Path('reports/three_wavelengths')
res={}
for code,pars in [('line',{}),('collimated',{'divergence_x_deg':1.,'divergence_y_deg':1.})]:
 b=simulate(Geometry(),replace(BASE,photons_per_line=18000,scan_lines=112,camera_pixels=128,seed=712,**pars),material_scenario(800))
 np.savez_compressed(D/f'{code}_800nm_seed712.npz',image=b['image'])
 a=np.load(D/f'{code}_800nm.npz')['image'].astype(float)
 x,y=gaussian_filter(a,2).ravel(),gaussian_filter(b['image'],2).ravel()
 res[code]={'weighted_flux_pct_seed712':100*b['diagnostics']['weighted_camera_signal']/b['diagnostics']['launched_photons'],
             'image_correlation_after_2px_smoothing':float(np.corrcoef(x,y)[0,1]),
             'different_seed_image_mean_abs_relative_to_mean':float(np.mean(abs(x-y))/np.mean(x))}
 print(code,res[code],flush=True)
(D/'convergence_seed712.json').write_text(json.dumps(res,indent=2)+'\n')