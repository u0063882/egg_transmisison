"""Streamlit interactive egg-transmission Monte Carlo prototype.

Run: streamlit run app.py
"""
from __future__ import annotations
import io
import json
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter

from eggtrans import Geometry, Optics, MATERIALS_800, MATERIAL_NAMES, MODE_NAMES, simulate, projected_masks
from eggtrans.spectra import load_spectra_csv, at_wavelength

st.set_page_config(page_title='Egg transmission | Monte Carlo', page_icon='🥚', layout='wide')
st.title('Egg transmission — Monte Carlo line-scan simulator')
st.caption('3D photon packets · ellipsoidal shell / albumen / yolk · optional blood spot · moving pushbroom scan')
st.warning('Research prototype, **not validated against real egg images**. Published 800 nm albumen/yolk scattering parameters are included; shell, blood and refractive indices are illustrative. Results are not a calibration or verified image prediction.', icon='⚠️')

with st.sidebar:
    st.header('Illumination')
    mode = st.selectbox('Geometry', MODE_NAMES)
    length = st.slider('Line/source extent (mm)', 10.0, 90.0, 48.0, 1.0)
    width = st.slider('Stripe width (mm)', 0.5, 20.0, 2.0, 0.5)
    diameter = st.slider('Spot diameter (mm)', 1.0, 30.0, 6.0, 1.0)
    offset = st.slider('Source–camera offset along travel (mm)', -15.0, 15.0, 0.0, 0.5)
    gap = st.slider('Source clearance below egg (mm)', 0.5, 20.0, 2.0, 0.5)
    st.subheader('Beam direction')
    st.caption('Angles are SD of incident ray slopes; 0° approximates collimated light. These are independent in x (along stripe) and y (along translation).')
    div_x = st.slider('Divergence x (° SD)', 0., 35., 8., 0.5)
    div_y = st.slider('Divergence y (° SD)', 0., 35., 3., 0.5)
    st.subheader('Idealised line camera')
    acceptance = st.slider('Collection half-angle (°)', 2., 75., 40., 1.)
    line_width = st.slider('Detector line acceptance at exit (mm)', 0.5, 12., 4., 0.5)
    st.caption('Angular gate + top-plane projection approximate a telecentric receiver. Does not model the exact lens, NA, PSF or spectrograph slit.')
    st.subheader('Monte Carlo precision')
    nphotons=st.select_slider('Packets per line', [500, 2000, 8000, 20000, 50000, 100000], value=8000)
    lines=st.select_slider('Scan lines', [24, 40, 64, 96, 128], value=40)
    pixels=st.select_slider('Camera pixels', [32, 48, 64, 96, 128], value=64)
    seed=st.number_input('Random seed', 0, 2_000_000_000, 42)

with st.expander('Egg geometry / artificial defect', expanded=False):
    cc1,cc2,cc3=st.columns(3)
    egg_w=cc1.number_input('Egg width x (mm)', 25., 70., 44., 1.)
    egg_l=cc2.number_input('Egg length y (mm)', 35., 90., 59., 1.)
    egg_h=cc3.number_input('Egg height z (mm)', 25., 70., 44., 1.)
    shell=st.slider('Shell thickness (mm)', .1, 1., .35, .05)
    c1,c2,c3=st.columns(3)
    yolk_x=c1.slider('Yolk x radius (mm)', 5., 20., 14., .5)
    yolk_y=c2.slider('Yolk y radius (mm)', 5., 25., 16., .5)
    yolk_z=c3.slider('Yolk z radius (mm)', 5., 20., 14., .5)
    blood=st.checkbox('Include artificial blood spot', value=True)
    d1,d2,d3,d4=st.columns(4)
    blood_x=d1.number_input('Blood x (mm)', -15., 15., 6., 1.)
    blood_y=d2.number_input('Blood y (mm)', -20., 20., 13., 1.)
    blood_z=d3.number_input('Blood z (mm)', -15., 15., 9., 1.)
    blood_radius=d4.slider('Spot radius (mm)', .5, 5., 2.3, .1)

g=Geometry(egg_width_mm=egg_w,egg_length_mm=egg_l,egg_height_mm=egg_h,shell_thickness_mm=shell,
           yolk_radius_x_mm=yolk_x,yolk_radius_y_mm=yolk_y,yolk_radius_z_mm=yolk_z,
           blood_x_mm=blood_x,blood_y_mm=blood_y,blood_z_mm=blood_z,blood_radius_mm=blood_radius,include_blood=blood)
o=Optics(mode=mode,line_length_mm=length,line_width_mm=width,spot_diameter_mm=diameter,
         source_offset_y_mm=offset,source_gap_mm=gap,divergence_x_deg=div_x,divergence_y_deg=div_y,
         camera_half_angle_deg=acceptance,camera_line_width_mm=line_width,
         photons_per_line=nphotons,scan_lines=lines,camera_pixels=pixels,seed=seed)

st.subheader('Optical properties')
wl_col,import_col=st.columns([1,2])
with wl_col:
    wavelength=st.number_input('Wavelength (nm)',400,1700,800,10)
with import_col:
    upload=st.file_uploader('Measured spectral coefficients CSV (optional)',type=['csv'],
        help='Columns: wavelength_nm,material,mu_a_mm_1,mu_s_mm_1,g,n; 4 material rows per wavelength.')
try:
    if upload is not None:
        spectra=load_spectra_csv(upload.getvalue())
        material_base=at_wavelength(spectra,float(wavelength))
        st.success('Using user-supplied wavelength-dependent coefficients (linear interpolation without extrapolation).')
    else:
        material_base=MATERIALS_800.copy()
        if wavelength != 800:
            st.warning('No measured spectra loaded. Reusing the 800 nm parameters at this wavelength as a **hypothetical what-if**, not a predicted spectrum.')
except ValueError as err:
    st.error(str(err)); st.stop()

rows=pd.DataFrame(material_base[1:],index=MATERIAL_NAMES[1:],columns=['μa (mm⁻¹)','μs (mm⁻¹)','g','n'])
st.caption('Only yolk and albumen μa, μs and g at 800 nm are literature anchors. Edit any parameter before simulation.')
with st.expander('Edit coefficients / import details',expanded=True):
    changed=st.data_editor(rows,use_container_width=True)
material=material_base.copy()
material[1:,:]=changed.to_numpy(dtype=float)

c_run,c_compare,c_vis=st.columns([1,1,2])
with c_run:
    clicked=st.button('▶ Run scan',type='primary',use_container_width=True)
with c_compare:
    compare=st.button('Compare 4 sources',use_container_width=True)
with c_vis:
    display_mode=st.radio('Display', ['Raw signal', 'Asinh visibility', 'Log10 visibility'],horizontal=True,
        help='Nonlinear displays help visualize low signal but do NOT recover lost spatial information.')


def make_heatmap(arr, x, y, title):
    fig=go.Figure(go.Heatmap(z=arr,x=x,y=y,colorscale='Inferno',colorbar={'title':'Relative'}))
    fig.update_layout(title=title,xaxis_title='Across egg x (mm)',yaxis_title='Translation y (mm)',
        height=500,margin=dict(l=40,r=40,t=55,b=40),yaxis={'scaleanchor':'x','scaleratio':1})
    return fig


def shown(img):
    if display_mode=='Asinh visibility':
        scale=max(float(np.max(img))*.03,1.e-12)
        return np.arcsinh(img/scale)
    if display_mode=='Log10 visibility':
        return np.log10(1+img/max(float(np.max(img))*.002,1.e-12))
    return img


if clicked or compare:
    try:
        with st.spinner('Tracing photons through curved shell, albumen and yolk… first run includes Numba JIT compilation.'):
            if compare:
                results={name:simulate(g,replace(o,mode=name),material) for name in MODE_NAMES}
                result=results[mode]
            else:
                results=None
                result=simulate(g,o,material)
    except (ValueError,TypeError) as err:
        st.error(f'Invalid configuration: {err}'); st.stop()
    st.session_state['scan_result']=result
    st.session_state['compare_results']=results
    st.session_state['scan_wavelength']=wavelength
    st.session_state['scan_params']={'geometry':asdict(g),'optics':asdict(o),'materials':material.tolist(),'wavelength_nm':wavelength,
                                      'provenance':'800nm paper anchors only for albumen/yolk; other values illustrative unless CSV uploaded'}

if 'scan_result' in st.session_state:
    result=st.session_state['scan_result']
    dx=result['x_mm'];dy=result['scan_y_mm']; image=result['image']
    mask=projected_masks(Geometry(**result['geometry']),dx,dy)
    st.markdown(f'### Simulated camera scan — {st.session_state["scan_wavelength"]} nm')
    a,b=st.columns(2)
    a.plotly_chart(make_heatmap(shown(image),dx,dy,'Detected transmission (Monte Carlo)'),use_container_width=True)
    reference=np.zeros_like(image)
    reference[mask['egg']]=1
    reference[mask['yolk']]=2
    if result['geometry']['include_blood']: reference[mask['blood']]=3
    b.plotly_chart(make_heatmap(reference,dx,dy,'True geometric projections (not observable ground truth)'),use_container_width=True)
    di=result['diagnostics']
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Launched packets',f'{di["launched_photons"]:,}')
    c2.metric('Camera-detected rays',f'{di["detected_ray_count"]:,}')
    c3.metric('Relative camera flux',f'{di["weighted_camera_signal"]/di["launched_photons"]:.3e}')
    c4.metric('Ballistic share (signal)',f'{di["weighted_ballistic_signal"]/max(di["weighted_camera_signal"],1.e-30):.1%}')
    if di['detected_ray_count'] < 500:
        st.warning('Few camera-detected packets: the image is Monte Carlo-noise dominated. Increase packets, collection angle or detector line width.')
    if di['max_event_terminated'] > .01*di['launched_photons']:
        st.warning('More than 1% of trajectories reached the event limit. Increase max_scattering_events in Optics for convergence.')
    st.caption('Source launch is normalized to one photon packet. No sensor quantum efficiency, spectral lamp radiance, read noise, real lens PSF or detector calibration included. A true pushbroom scan has the fixed line light and camera while the egg translates.')
    with st.expander('Spatial profile and Monte Carlo noise check'):
        middle_idx=np.argmin(np.abs(dy))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dx,y=image[middle_idx],mode='lines+markers',name='Raw line'))
        fig.add_trace(go.Scatter(x=dx,y=gaussian_filter(image,(1,1))[middle_idx],mode='lines',name='1px Gaussian preview (display only)'))
        fig.update_layout(xaxis_title='x (mm)',yaxis_title='Relative detected packets / source packet',height=330)
        st.plotly_chart(fig,use_container_width=True)
    with st.expander('Download results',expanded=True):
        buff=io.BytesIO()
        np.savez_compressed(buff,image=image,x_mm=dx,scan_y_mm=dy,egg_mask=mask['egg'],yolk_mask=mask['yolk'],blood_mask=mask['blood'],
                            metadata_json=json.dumps(st.session_state['scan_params']),diagnostics_json=json.dumps(di))
        st.download_button('Download scan + geometry (.npz)',buff.getvalue(),file_name='egg_scan.npz',mime='application/octet-stream')
        st.download_button('Download configuration (.json)',json.dumps(st.session_state['scan_params'],indent=2),file_name='egg_configuration.json',mime='application/json')
    comparisons=st.session_state.get('compare_results')
    if comparisons:
        st.subheader('Comparison — equal photon packets launched per scan line')
        cols=st.columns(2)
        for idx,name in enumerate(MODE_NAMES):
            rr=comparisons[name]
            with cols[idx%2]:
                st.plotly_chart(make_heatmap(shown(rr['image']),rr['x_mm'],rr['scan_y_mm'],name),use_container_width=True)
        st.info('Source brightness is normalized per emitted photon. For a physically equal W/mm² comparison, calibrate each illuminator’s radiant output and spectral distribution; broader sources have different emitting areas.')
else:
    st.info('Choose a source and press **Run scan**. Suggested first run: 800 nm transverse line, 8,000 packets × 40 lines.')

with st.expander('Model assumptions / limitations'):
    st.markdown('''
- **Actual scattering events are sampled in 3D**, not approximated as a Gaussian blur: exponential free paths, Henyey–Greenstein angle sampling, Beer absorption, Fresnel reflection, and Snell refraction.
- Geometries are ideal **nested ellipsoids/spherical blood inclusion**, not reconstructed anatomy. Shell thickness is only approximately uniform for an ellipsoid.
- Camera is an **ideal angularly-gated top-plane line receiver**, not the exact spectral line camera lens/pupil. Refraction, dynamic yolk position, birefringence, shell pores, polarization and heterogeneity beyond four materials are simplified.
- 150 W lamp power does not determine ray direction, and **lamp wattage is not calibrated into absolute spectral irradiance**. Angular divergence is a tunable proxy for your SCHOTT cylindrical optic.
- No real data validation has been performed; compare simulated vs measured line profiles and source beam profiles before drawing quantitative conclusions.
- The initial preview can be noisy at realistic small camera acceptance angles. Higher-fidelity research should eventually employ validated packages such as MCX or PyXOpto with measured egg-property spectra and a calibrated receiver model.
''')