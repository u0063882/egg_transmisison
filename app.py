"""Streamlit frontend: native MCX transport via PMCX, no custom MC kernel."""
from __future__ import annotations
import io,json
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from eggtrans.core import Geometry,Optics,MATERIALS_800,MATERIAL_NAMES,MODE_NAMES,simulate,projected_masks
from eggtrans.mcx_engine import backend_status,build_egg_volume,source_config,source_footprint_mm
from eggtrans.schematic import apparatus_figure
from eggtrans.spectra import load_spectra_csv,at_wavelength

st.set_page_config(page_title='Egg transmission · MCX',layout='wide',page_icon='🥚')
st.title('Egg transmission · MCX / PMCX')
st.caption('Validated photon-transport engine · voxelized shell / albumen / yolk · true pushbroom geometry')
st.info('Photon transport is performed by **MCX via PMCX**, not by a custom ray tracer. The geometry, source definitions and idealized camera binning are project-specific. Egg tissue parameters are not fully calibrated.')

with st.sidebar:
    st.header('Illumination (MCX native source)')
    mode=st.selectbox('Illumination',MODE_NAMES)
    length=st.slider('Line length / broad size (mm)',10.,90.,48.,1.,disabled=mode=='Spot')
    width=st.slider('Line width (mm)',.5,15.,2.,.5,disabled=mode in ('Spot','Broad backlight'))
    diameter=st.slider('Spot diameter (mm)',1.,30.,6.,.5,disabled=mode!='Spot')
    offset=st.slider('Source offset along translation (mm)',-15.,15.,0.,.5)
    gap=st.slider('Source gap beneath egg (mm)',1.,12.,2.,.5)
    fchoice=st.selectbox('Input direction (MCX focus)',('Parallel / collimated','Divergent (virtual focus behind)','Convergent (focus ahead)'))
    focal=0. if fchoice.startswith('Parallel') else st.slider('Focal distance magnitude (mm)',20.,250.,100.,10.)*(1 if fchoice.startswith('Convergent') else -1)
    st.header('Line camera · simplified angular gate')
    acceptance=st.slider('Acceptance half-angle (°)',2.,80.,35.,1.)
    slit=st.slider('Observation slit width at camera plane (mm)',.5,12.,4.,.5)
    st.header('MCX simulation')
    voxel=st.select_slider('Voxel size (mm)',[.5,.75,1.,1.5,2.],value=.5)
    packets=st.select_slider('Photons per line',[10_000,30_000,100_000,300_000,1_000_000],value=100_000)
    nlines=st.select_slider('Scan lines',[12,24,48,96],value=24)
    npixels=st.select_slider('Camera pixels',[32,64,96,128],value=64)
    gpu=st.number_input('GPU ID (1-based)',min_value=1,max_value=16,value=1)
    seed=st.number_input('Seed',min_value=1,max_value=2_000_000_000,value=42)

with st.expander('Egg geometry and inclusion positions',expanded=False):
    x1,x2,x3=st.columns(3)
    egg_w=x1.number_input('Egg width (mm)',25.,80.,44.)
    egg_l=x2.number_input('Egg length (mm)',35.,90.,59.)
    egg_h=x3.number_input('Egg height (mm)',25.,80.,44.)
    shell=st.number_input('Shell thickness (mm)',.1,2.,.35,step=.05)
    a,b,c=st.columns(3)
    rx=a.number_input('Yolk radius x (mm)',4.,25.,14.)
    ry=b.number_input('Yolk radius y (mm)',4.,27.,16.)
    rz=c.number_input('Yolk radius z (mm)',4.,22.,14.)
    yolk=st.checkbox('Include yolk',True)
    a,b,c=st.columns(3)
    cx=a.slider('Yolk shift x (mm)',-8.,8.,-1.5,.5)
    cy=b.slider('Yolk shift y (mm)',-12.,12.,0.,.5)
    cz=c.slider('Yolk shift z (mm)',-6.,6.,-1.,.5)
    blood=st.checkbox('Include synthetic blood spot',True)
    a,b,c,d=st.columns(4)
    bx=a.number_input('Blood x (mm)',-16.,16.,6.)
    by=b.number_input('Blood y (mm)',-25.,25.,13.)
    bz=c.number_input('Blood z (mm)',-16.,16.,9.)
    br=d.number_input('Blood radius (mm)',.5,6.,2.3)

g=Geometry(egg_width_mm=egg_w,egg_length_mm=egg_l,egg_height_mm=egg_h,shell_thickness_mm=shell,
           yolk_radius_x_mm=rx,yolk_radius_y_mm=ry,yolk_radius_z_mm=rz,
           yolk_offset_x_mm=cx,yolk_offset_y_mm=cy,yolk_offset_z_mm=cz,
           include_yolk=yolk,blood_x_mm=bx,blood_y_mm=by,blood_z_mm=bz,
           blood_radius_mm=br,include_blood=blood)
o=Optics(mode=mode,line_length_mm=length,line_width_mm=width,spot_diameter_mm=diameter,
         source_offset_y_mm=offset,source_gap_mm=gap,beam_focal_length_mm=focal,
         camera_half_angle_deg=acceptance,camera_line_width_mm=slit,
         voxel_mm=voxel,photons_per_line=packets,scan_lines=nlines,camera_pixels=npixels,
         gpu_id=gpu,seed=seed)

st.subheader('Apparatus geometry and selected source')
st.plotly_chart(apparatus_figure(g,o),use_container_width=True)
st.caption('Orange = MCX illumination footprint at the current scan position; dashed blue = camera observation line. '
           'As the egg moves, a **circular spot can reconstruct as a lengthwise stripe** because the spot stays at the same x while successive scan lines are acquired. '
           'The top view is source geometry, *not* a simulated photon image.')

status,msg=backend_status()
if status:
    st.success(msg)
else:
    st.warning(msg+' Geometry preview still works, but simulation requires PMCX on a supported GPU.')
if voxel>shell:
    st.warning('Voxel size exceeds shell thickness. The shell will be only approximately represented and may have missing sections. Try a finer voxel grid.')

st.subheader('Spectral optical properties')
wcol,ccol=st.columns([1,2])
with wcol:
    wl=st.number_input('Wavelength (nm)',400,1700,800,10)
with ccol:
    upload=st.file_uploader('Optional measured properties CSV',type=['csv'])
try:
    material=at_wavelength(load_spectra_csv(upload.getvalue()),float(wl)) if upload else MATERIALS_800.copy()
except ValueError as exc:
    st.error(str(exc));st.stop()
if upload is None and wl!=800:
    st.warning('No measured spectral coefficients: **800 nm values reused at this wavelength as a what-if**. NOT a physical prediction.')
rows=pd.DataFrame(material[1:],index=MATERIAL_NAMES[1:],columns=['μa (mm⁻¹)','μs (mm⁻¹)','g','n'])
with st.expander('Inspect / edit optical coefficients',expanded=False):
    edited=st.data_editor(rows,use_container_width=True)
material[1:,:]=edited.to_numpy(dtype=float)
st.caption('Only 800 nm albumen/yolk μa, μs, g are literature-based; shell, blood and refractive indices are placeholders unless replaced by measured values.')

try:
    domain=build_egg_volume(g,o,material)
    cfgpreview=source_config(domain,o,0.)
    st.caption(f'Native MCX source: **{cfgpreview["srctype"]}** · voxel volume {" × ".join(map(str,domain.dims))} '
               f'({domain.vol.nbytes/1e6:.1f} MB labels, before GPU scratch). Detector: **+z exit-face photons**, then angular/slit selection.')
    with st.expander('Inspect MCX launch parameters'):
        st.json(cfgpreview)
except ValueError as exc:
    st.error(f'Geometry invalid: {exc}');st.stop()

c1,c2=st.columns(2)
run=c1.button('▶ Run MCX pushbroom scan',type='primary',use_container_width=True,disabled=not status)
compare=c2.button('Compare spot vs transverse line',use_container_width=True,disabled=not status)
showtruth=st.checkbox('Show geometric ground truth separately (NOT observed by camera)',False)

if run or compare:
    try:
        with st.spinner('Running MCX on GPU for each pushbroom line…'):
            if compare:
                results={name:simulate(g,replace(o,mode=name),material) for name in ('Spot','Transverse line')}
                result=results[mode] if mode in results else results['Transverse line']
            else:
                results={};result=simulate(g,o,material)
    except Exception as exc:
        st.error(f'MCX simulation error: {exc}');st.stop()
    st.session_state['result']=result
    st.session_state['compared']=results
    st.session_state['wavelength']=wl


def image_plot(image,x,y,name):
    f=go.Figure(go.Heatmap(x=x,y=y,z=image,colorscale='Inferno'))
    f.update_layout(title=name,xaxis_title='x (mm)',yaxis_title='scan position y (mm)',
                    height=500,margin=dict(l=45,r=30,t=50,b=35))
    f.update_yaxes(scaleanchor='x',scaleratio=1)
    return f

if 'result' in st.session_state:
    r=st.session_state['result']
    x,y=r['x_mm'],r['scan_y_mm']
    st.subheader(f'MCX camera transmission · {st.session_state["wavelength"]} nm')
    image=r['image']
    if showtruth:
        a,b=st.columns(2)
        a.plotly_chart(image_plot(image,x,y,'Detected photons (no geometry overlay)'),use_container_width=True)
        m=projected_masks(Geometry(**r['geometry']),x,y)
        truth=np.zeros_like(image)
        truth[m['egg']]=1;truth[m['yolk']]=2;truth[m['blood']]=3
        b.plotly_chart(image_plot(truth,x,y,'Ideal geometry · NOT a camera image'),use_container_width=True)
    else:
        st.plotly_chart(image_plot(image,x,y,'Detected photons only'),use_container_width=True)
    d=r['diagnostics']
    a,b,c=st.columns(3)
    a.metric('Photons launched',f'{d["launched_photons"]:,}')
    b.metric('Detected at upper MCX face',f'{d["top_exit_photons"]:,}')
    c.metric('Accepted by model camera',f'{d["camera_accepted_photons"]:,}')
    if d['camera_accepted_photons'] < 500:
        st.warning('Few accepted photons. Interpretation is Monte Carlo-noise dominated; increase the photon budget or camera acceptance.')
    if st.session_state.get('compared'):
        a,b=st.columns(2)
        for ax,(name,rr) in zip((a,b),st.session_state['compared'].items()):
            ax.plotly_chart(image_plot(rr['image'],rr['x_mm'],rr['scan_y_mm'],name),use_container_width=True)
    metadata={'engine':'MCX via PMCX','wavelength_nm':st.session_state['wavelength'],
              'optics':r['optics'],'geometry':r['geometry'],'materials':r['materials'].tolist(),
              'diagnostics':r['diagnostics']}
    buff=io.BytesIO()
    np.savez_compressed(buff,image=image,x_mm=x,scan_y_mm=y,metadata=json.dumps(metadata))
    st.download_button('Download raw signal (.npz)',buff.getvalue(),'mcx_egg_scan.npz')
    st.download_button('Download run metadata (.json)',json.dumps(metadata,indent=2),'mcx_egg_config.json')
else:
    st.info('Inspect the source schematic. On a computer with a supported MCX backend, click **Run MCX pushbroom scan**.')

with st.expander('Scientific scope and limitations'):
    st.markdown('''
- **MCX** performs physical photon scattering, absorption and interface transport on the voxel grid. No home-built Monte Carlo fallback exists.
- Native MCX **disk** source for spot; native **planar** for transverse, longitudinal, broad illumination. These are verified with unit tests.
- MCX detects photons leaving the **upper boundary face**. This program applies an idealized camera slit and angular gate to their positions and directions; a true lens, spectrograph and sensor are not modelled.
- MCX has one shared focal-length parameter for these sources, not the older model’s independent x/y Gaussian divergences. No false equivalence is implied.
- The shell is rasterized in voxels, potentially coarse at the typical ~0.35 mm shell thickness; at 1 mm voxel spacing shell detail is **not reliable**.
- A stationary spot can look like a stripe in a pushbroom composite because it moves relative to the egg in y, but remains at approximately fixed x. This is an expected acquisition geometry effect.
- At other wavelengths, load experimentally measured optical coefficients. Source spectrum and detector spectral sensitivity are not included; signal is per launched photon packet.
- No claim is made of having experimentally validated yolk/blood boundaries. Run the known-answer Beer–Lambert test in `tests/test_mcx_physics.py` on your GPU.
''')
