"""Export voxel-labelled egg and single-view MCX JSON configurations.

For an independently developed and validated Monte Carlo engine, see
https://github.com/fangq/mcx. This exporter is not a camera reconstructor.
It uses fully collimated planar source geometries; spectral coefficients
and pupil acceptance must be handled by the research user.

CLI: python -m eggtrans.mcx_export --output results/mcx_case --voxel-mm 0.5
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import Geometry, Optics, MATERIALS_800, MODE_NAMES, validate


def voxelize(g: Geometry, o: Optics, voxel_mm: float = 0.5, margin_mm: float = 6.0, scan_y_range_mm: float = 0.0):
    """Generate x,y,z-indexed MCX uint8 labels and grid coordinate metadata.

    Labels match optical-property matrix rows: 0 air, 1 shell,
    2 albumen, 3 yolk, 4 artificial blood. MCX expects *Fortran*
    ordering for the raw byte file.
    """
    if not 0.1 <= voxel_mm <= 2.0 or margin_mm < 2:
        raise ValueError('Voxel size must be 0.1–2 mm and margin >= 2 mm.')
    source_y_span = (o.line_width_mm if o.mode == "Transverse line" else
                     o.spot_diameter_mm if o.mode == "Spot" else
                     o.line_length_mm if o.mode == "Longitudinal line" else
                     max(o.line_length_mm, g.egg_width_mm, g.egg_length_mm))
    size=np.array([
        max(g.egg_width_mm,o.line_length_mm,o.spot_diameter_mm)+2*margin_mm,
        max(g.egg_length_mm,source_y_span+2*(abs(o.source_offset_y_mm)+scan_y_range_mm))+2*margin_mm,
        g.egg_height_mm+2*(margin_mm+o.source_gap_mm),
    ],dtype=float)
    dims=np.ceil(size/voxel_mm).astype(int)
    extent=dims*voxel_mm
    x=((np.arange(dims[0])+.5)*voxel_mm-extent[0]/2)[:,None,None]
    y=((np.arange(dims[1])+.5)*voxel_mm-extent[1]/2)[None,:,None]
    z=((np.arange(dims[2])+.5)*voxel_mm-extent[2]/2)[None,None,:]
    outer=(x/(g.egg_width_mm/2))**2+(y/(g.egg_length_mm/2))**2+(z/(g.egg_height_mm/2))**2<=1
    inner=(x/(g.egg_width_mm/2-g.shell_thickness_mm))**2+\
          (y/(g.egg_length_mm/2-g.shell_thickness_mm))**2+\
          (z/(g.egg_height_mm/2-g.shell_thickness_mm))**2<=1
    labels=np.zeros(tuple(dims),dtype=np.uint8)
    labels[outer]=1
    labels[inner]=2
    yolk=((x-g.yolk_offset_x_mm)/g.yolk_radius_x_mm)**2+\
         ((y-g.yolk_offset_y_mm)/g.yolk_radius_y_mm)**2+\
         ((z-g.yolk_offset_z_mm)/g.yolk_radius_z_mm)**2<=1
    labels[yolk]=3
    if g.include_blood:
        blood=(x-g.blood_x_mm)**2+(y-g.blood_y_mm)**2+(z-g.blood_z_mm)**2 <=g.blood_radius_mm**2
        labels[blood]=4
    return labels,extent


def mcx_case(g: Geometry, o: Optics, materials: np.ndarray, voxel_mm: float,
             extent_mm: np.ndarray, volume_filename: str, source_y_mm: float=0.0) -> dict:
    """MCX configuration at one scan coordinate. Assumes collimated launch.

    MCX's fluence output is not a camera image. Use exit-position/exit-direction
    records plus receiver geometry to generate a synthetic line detector.
    """
    tovoxel=lambda x,y,z: list((np.array([x,y,z],dtype=float)+extent_mm/2)/voxel_mm)
    source_z=-g.egg_height_mm/2-o.source_gap_mm
    sy=source_y_mm+o.source_offset_y_mm
    if o.mode=='Spot':
        source={'Type':'disk','Pos':tovoxel(0.,sy,source_z),'Dir':[0.,0.,1.,0.],
                'Param1':[o.spot_diameter_mm/(2*voxel_mm),0.,0.,0.]}
    else:
        if o.mode=='Transverse line':
            spanx,spany=o.line_length_mm,o.line_width_mm
        elif o.mode=='Longitudinal line':
            spanx,spany=o.line_width_mm,o.line_length_mm
        else:
            spanx=spany=max(o.line_length_mm,g.egg_width_mm,g.egg_length_mm)
        source={'Type':'planar','Pos':tovoxel(-spanx/2,sy-spany/2,source_z),
                'Dir':[0.,0.,1.,0.],
                'Param1':[spanx/voxel_mm,0.,0.,0.],
                'Param2':[0.,spany/voxel_mm,0.,0.]}
    # Large detector sphere catches tissue->air escapes; filter exit x/v later.
    detector={'Pos':tovoxel(0.,source_y_mm,g.egg_height_mm/2),
              'R':max(g.egg_width_mm,g.egg_length_mm)/voxel_mm}
    return {
      'Domain':{'VolumeFile':volume_filename,
                'Dim':[int(x) for x in np.round(extent_mm/voxel_mm)],
                'MediaFormat':'byte','OriginType':1,'LengthUnit':float(voxel_mm),
                'Media':[dict(mua=float(row[0]),mus=float(row[1]),g=float(row[2]),n=float(row[3])) for row in materials]},
      'Session':{'ID':'egg_mcx','Photons':int(o.photons_per_line),
                 'RNGSeed':int(o.seed),'OutputType':'flux','OutputFormat':'jnii',
                 'DoMismatch':1,'DoSaveExit':1,'DoPartialPath':1,
                 'SaveDataMask':'DPXVW'},
      'Forward':{'T0':0.,'T1':1.e-8,'Dt':1.e-8},
      'Optode':{'Source':source,'Detector':[detector]},
      'Help':{'note':'MCX is an independent voxel light transport backend; does NOT simulate camera by itself.',
              'scan_y_mm':float(source_y_mm),'nominal_angle_sd_deg':[o.divergence_x_deg,o.divergence_y_deg],
              'limitation':'MCX source exported as perfectly collimated, ignores Gaussian divergence of interactive model.'}}


def export(g: Geometry|None=None, o: Optics|None=None, materials:np.ndarray|None=None,
           output:Path|str='results/mcx', voxel_mm:float=.5, scan_lines:int=1) -> list[Path]:
    g=g or Geometry();o=o or Optics()
    m=np.array(materials if materials is not None else MATERIALS_800,dtype=np.float64)
    validate(g,o,m)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    vol,extent=voxelize(g,o,voxel_mm,scan_y_range_mm=(g.egg_length_mm*.55 if scan_lines>1 else 0.))
    (out/'egg_labels.bin').write_bytes(vol.tobytes(order='F'))
    (out/'voxel_meta.json').write_text(json.dumps({'shape_xyz':list(vol.shape),
        'voxel_mm':voxel_mm,'extent_mm':extent.tolist(),'order':'Fortran','labels':['air','shell','albumen','yolk','blood']},indent=2))
    yy=[0.] if scan_lines == 1 else np.linspace(-g.egg_length_mm*.55,g.egg_length_mm*.55,scan_lines)
    paths=[]
    for i,y in enumerate(yy):
        case=mcx_case(g,o,m,voxel_mm,extent,'egg_labels.bin',float(y))
        case['Session']['ID']=f'egg_line_{i:03d}'
        path=out/f'scan_{i:03d}.json'
        path.write_text(json.dumps(case,indent=2))
        paths.append(path)
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/mcx'))
    p.add_argument('--voxel-mm',type=float,default=.5)
    p.add_argument('--scan-lines',type=int,default=1)
    p.add_argument('--mode',choices=MODE_NAMES,default='Transverse line')
    p.add_argument('--photons',type=int,default=1000000)
    a=p.parse_args()
    if not 1<=a.scan_lines<=300:
        p.error('scan-lines must be 1–300')
    paths=export(o=Optics(mode=a.mode,photons_per_line=a.photons),output=a.output,
                 voxel_mm=a.voxel_mm,scan_lines=a.scan_lines)
    print(f'Wrote {len(paths)} MCX input(s) and voxel volume under {a.output}.')
    print(f'From that directory: mcx -f {paths[0].name} -G 1')
    print('No MCX simulation was executed; reconstructed line images require saved-exit ray filtering.')

if __name__=='__main__': main()