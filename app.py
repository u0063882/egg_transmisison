"""Streamlit frontend: native MCX transport via PMCX, no custom MC kernel."""
from __future__ import annotations
import io,json
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from eggtrans.core import Geometry,Optics,MATERIALS_800,MATERIAL_NAMES,MODE_NAMES,CAMERA_MODES,simulate,projected_masks
from eggtrans.mcx_engine import backend_status,domain_plan,source_config,source_footprint_mm
from eggtrans.schematic import apparatus_figure
from eggtrans.visualization import camera_image_figure, camera_scene_3d
from eggtrans.readout_validation import compare_fixed_scene_readouts
from eggtrans.acquisition_comparison import compare_matched_acquisitions
from eggtrans.spectra import load_spectra_csv,at_wavelength

st.set_page_config(page_title='Egg transmission · MCX',layout='wide',page_icon='🥚')
st.title('Egg transmission · MCX / PMCX')
st.caption('App v0.9.0 · MCX photon transport · camera selection is on this main page')
st.subheader('Camera / acquisition')
camera_mode=st.radio(
    'Camera type', CAMERA_MODES,
    format_func=lambda x: 'Pushbroom (line scan)' if x=='Pushbroom' else 'Normal camera (2D area)',
    horizontal=True, key='camera_mode')
st.caption('Pushbroom: egg translates, one row per MCX run. Area: stationary egg and source, one full-frame MCX exposure.')
st.info('Photon transport is performed by **MCX via PMCX**, not by a custom ray tracer. The geometry, source definitions and idealized camera binning are project-specific. Egg tissue parameters are not fully calibrated.')

with st.sidebar:
    st.caption('Camera: ' + ('Pushbroom (line scan)' if camera_mode == 'Pushbroom' else 'Normal camera (2D area)'))
    st.header('Illumination (MCX native source)')
    mode=st.selectbox('Illumination',MODE_NAMES)
    length=st.slider('Line length / broad size (mm)',10.,90.,48.,1.,disabled=mode=='Spot')
    width=st.slider('Line width (mm)',.5,15.,2.,.5,disabled=mode in ('Spot','Broad backlight'))
    diameter=st.slider('Spot diameter (mm)',1.,30.,6.,.5,disabled=mode!='Spot')
    offset=st.slider('Source offset y (mm)' + (' · relative to camera line' if camera_mode=='Pushbroom' else ' · relative to egg centre'),-15.,15.,0.,.5)
    gap=st.slider('Source gap beneath egg (mm)',1.,12.,2.,.5)
    fchoice=st.selectbox('Input direction (MCX focus)',('Parallel / collimated','Divergent (virtual focus behind)','Convergent (focus ahead)'))
    focal=0. if fchoice.startswith('Parallel') else st.slider('Focal distance magnitude (mm)',20.,250.,100.,10.)*(1 if fchoice.startswith('Convergent') else -1)
    st.header('Camera · simplified angular gate')
    acceptance=st.slider('Acceptance half-angle (°)',2.,80.,35.,1.)
    st.header('Direct illumination rejection')
    path_mode=st.selectbox('Detector photon-history filter',
                           ('Traversed egg','Traversed interior','All exits'),
                           help='MCX path-history diagnostic, NOT a real camera baffle. '
                                'Traversed egg accepts paths through shell or interior; '
                                'Traversed interior rejects shell-only and air-only rays. '
                                'All exits shows unfiltered top-face detector photons.')
    min_path=st.slider('Minimum path inside selected egg region (mm)',0.,10.,0.,.1,
                       disabled=(path_mode=='All exits'),
                       help='Reject grazing paths shorter than the selected threshold. '
                            'At 0 mm any positive MCX-recorded tissue path qualifies. '
                            'Increasing this threshold can hide real egg rim signal.')
    st.caption('This is an idealized history filter available only because MCX tracks the photons. '
               'A physical camera cannot identify egg-traversing rays from their histories; '
               'use baffles, controlled source geometry and camera aperture/NA to suppress stray light.')
    slit=st.slider('Observation slit width (mm) — pushbroom only',.5,12.,4.,.5,disabled=camera_mode=='Area')
    st.header('MCX simulation')
    voxel=st.select_slider('Voxel size (mm)',[.2,.25,.35,.5,.75,1.,1.5,2.],value=.5)
    padding=st.select_slider('Lateral MCX air padding (mm)',[4.,6.,8.,12.,16.,24.,32.],value=8.,help='Air outside the active egg, source and camera field. Compare multiple values before trusting transmission.')
    host_budget=st.select_slider('Domain build safety budget (MiB)',[128.,256.,512.,1024.,2048.],value=512.,help='Approximate host rasterization memory, NOT guaranteed GPU memory. Exceeding this budget blocks allocation.')
    packets=st.select_slider('Photons per MCX run / area exposure',[10_000,30_000,100_000,300_000,1_000_000],value=100_000)
    nlines=st.select_slider('Pushbroom scan lines',[12,24,48,96],value=24,disabled=camera_mode=='Area')
    npixels=st.select_slider('Pixels in x',[32,64,96,128],value=64)
    area_rows=st.select_slider('Area camera pixels in y',[32,64,96,128],value=96,disabled=camera_mode=='Pushbroom')
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
o=Optics(mode=mode,camera_mode=camera_mode,line_length_mm=length,line_width_mm=width,spot_diameter_mm=diameter,
         source_offset_y_mm=offset,source_gap_mm=gap,beam_focal_length_mm=focal,
         camera_half_angle_deg=acceptance,camera_line_width_mm=slit,
         camera_path_mode=path_mode,min_egg_path_mm=min_path,
         voxel_mm=voxel,domain_padding_mm=padding,domain_build_budget_mib=host_budget,
         photons_per_line=packets,scan_lines=nlines,camera_pixels=npixels,area_rows=area_rows,
         gpu_id=gpu,seed=seed)

if mode=='Transverse line' and length < egg_w+6:
    st.warning(f'Transverse line length is {length:.0f} mm vs egg width {egg_w:.0f} mm. '
               'For full lateral illumination, try a 55–60 mm line and compare source coverage, '
               'camera signal, and MCX padding convergence. A wider source does NOT reject bypass rays.')
if mode=='Spot' and camera_mode=='Pushbroom' and diameter < egg_w:
    st.info(f'Only a {diameter:.0f} mm-wide lateral swath is directly illuminated by the spot, '
            f'versus a {egg_w:.0f} mm-wide egg. A finished pushbroom image will not necessarily '
            'reveal the full egg sides. Consider a transverse line for full-width coverage.')
st.subheader('Apparatus geometry and selected source')
st.plotly_chart(apparatus_figure(g,o),use_container_width=True)
if camera_mode=='Pushbroom':
    st.caption('Orange = MCX source footprint; dashed blue = the camera line. The egg translates along y. '
               'A spot can therefore appear elongated along y in the stacked pushbroom image. This is a geometry schematic, not a photon image.')
else:
    st.caption('Orange = MCX source footprint; dashed blue = the full 2D area-camera field of view. '
               'Egg and illumination are stationary. A spot is a disk, not a line scan. This is a geometry schematic, not a photon image.')

status,msg=backend_status()
if status:
    st.success(msg)
else:
    st.warning(msg+' Geometry preview still works, but simulation requires PMCX on a supported GPU.')
st.caption(f'Shell / voxel size = **{shell/voxel:.2f}**. MCX label grids are uniform in this version; the domain now follows the active light source, egg and detector, with adjustable padding and a host-memory safety budget.')
if voxel >= shell:
    st.warning('Voxel size is at least as large as the 0.35 mm default shell: severe boundary aliasing is possible, including missing shell voxels. Use finer voxels or a mesh/split-voxel solver.')
elif shell/voxel < 3:
    st.warning('Less than three nominal voxels span the shell. Curved surface and inner/outer boundary accuracy remain limited.')

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
    # Geometry-only preflight; do NOT rasterize millions of voxels on each UI refresh.
    plan=domain_plan(g,o,material)
    cfgpreview=source_config(plan,o,0.)
    detector_label='area x/y readout' if camera_mode=='Area' else 'moving slit readout'
    st.caption(f'**MCX source:** {cfgpreview["srctype"]} · active mode {mode} · grid '
               f'{" × ".join(map(str,plan.dims))} = **{plan.nvoxels/1e6:.2f} million voxels**. '
               f'Label volume {plan.label_memory_mib:.1f} MiB · rough Python rasterization working memory '
               f'{plan.host_build_estimate_mib:.0f} MiB / {host_budget:.0f} MiB limit. '
               f'Temporary MCX GPU buffers are additional and GPU-dependent. Detector: {detector_label}.')
    with st.expander('Inspect source parameters and padding cost'):
        st.json(cfgpreview)
        variants=[]
        for mm in (4.,8.,12.,16.,24.):
            try:
                pp=domain_plan(g,replace(o,domain_padding_mm=mm),material,enforce_budget=False)
                variants.append({'Padding (mm)':mm,'Voxel grid':' × '.join(map(str,pp.dims)),
                                 'Voxels (million)':round(pp.nvoxels/1e6,2),
                                 'Host build estimate (MiB)':round(pp.host_build_estimate_mib),
                                 'Within current host budget':pp.host_build_estimate_mib<=host_budget})
            except ValueError:
                pass
        st.dataframe(pd.DataFrame(variants),hide_index=True,use_container_width=True)
        st.caption('More padding retains more scattered photons before side-boundary escape, '
                   'but costs memory. These are geometric estimates; run the GPU padding-convergence '
                   'script before treating an image as independent of domain size.')
except ValueError as exc:
    st.error(f'Domain preflight rejected this configuration: {exc}');st.stop()

c1,c2=st.columns(2)
run=c1.button('▶ Run MCX ' + ('pushbroom scan' if camera_mode=='Pushbroom' else '2D area exposure'),type='primary',use_container_width=True,disabled=not status)
compare=c2.button('Compare spot vs transverse line',use_container_width=True,disabled=not status)

if run or compare:
    try:
        with st.spinner('Running MCX on GPU (' + ('one run per pushbroom line' if camera_mode=='Pushbroom' else 'one 2D area exposure') + ')…'):
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


def image_plot(image,x,y,name,camera_mode='Pushbroom',zmax=None):
    f=go.Figure(go.Heatmap(x=x,y=y,z=image,colorscale='Inferno',zmin=0 if zmax is not None else None,zmax=zmax))
    ylabel='scan position y (mm)' if camera_mode=='Pushbroom' else 'image position y (mm)'
    f.update_layout(title=name,xaxis_title='x (mm)',yaxis_title=ylabel,
                    height=500,margin=dict(l=45,r=30,t=50,b=35))
    f.update_yaxes(scaleanchor='x',scaleratio=1)
    return f

with st.expander('Diagnostic: compare detector readouts with identical photons', expanded=False):
    st.markdown('''**Why the images can differ:** the ordinary pushbroom experiment
    moves the source/camera relative to the egg, whereas the area mode records
    a single fixed illumination footprint. With a spot, the two images are
    **not** expected to look alike. A wider pushbroom slit also differs from
    a narrow area-camera pixel row.

    This diagnostic instead uses **one MCX run**, holds the egg and source fixed,
    and bins the *same exiting photons* twice: into 2D pixels and into an
    equivalent sequence of row-wide virtual slit measurements. The row slit
    width is set to exactly one area-camera pixel pitch. Agreement checks the
    readout implementation but is **not a validation of egg motion or MCX physics**.
    ''')
    check = st.button('Run fixed-scene detector equivalence check', disabled=not status)
    if check:
        try:
            with st.spinner('Running one MCX exposure and comparing both detector readouts…'):
                diagnostic = compare_fixed_scene_readouts(g, o, material)
        except Exception as exc:
            st.error(f'Fixed-scene readout check failed: {exc}')
        else:
            aa, bb = st.columns(2)
            aa.plotly_chart(image_plot(diagnostic['area_image'], diagnostic['x_mm'],
                                      diagnostic['y_mm'], 'Area camera · same MCX photons', 'Area'),
                            use_container_width=True)
            bb.plotly_chart(image_plot(diagnostic['line_readout_image'], diagnostic['x_mm'],
                                      diagnostic['y_mm'], 'Equivalent virtual line readout', 'Area'),
                            use_container_width=True)
            st.metric('Relative image discrepancy',
                      f"{diagnostic['diagnostics']['relative_l1_difference']:.3g}")
            st.caption('This is a stationary-source detector consistency check, not a moving-egg pushbroom reconstruction. Both images must agree apart from numerical boundary cases.')

with st.expander('Diagnostic: matched broad-illumination acquisition',expanded=False):
    st.markdown('''The ordinary pushbroom experiment moves a narrow source with the camera slit;
    it cannot be compared directly with a stationary area-camera exposure.

    This **controlled comparison** instead uses the **same broad backlight fixed relative
    to the egg** for both camera readouts, with identical angular acceptance, spatial
    sampling and photon normalization. The area image requires one MCX run; sequential
    virtual camera rows require one additional MCX run per row. Their **statistical**
    images should agree when photon counts are sufficiently high. This is not the
    actual moving-egg experiment and noise can cause large differences.
    ''')
    comparison_rows=st.select_slider('Number of comparison rows',[8,12,24,48],value=12)
    do_comp=st.button(f'Run matched comparison ({comparison_rows+1} MCX runs)',disabled=not status)
    if do_comp:
        try:
            with st.spinner('Running matched fixed-source MCX exposure and sequential rows…'):
                matched=compare_matched_acquisitions(g,o,material,rows=comparison_rows)
        except Exception as exc:
            st.error(f'Matched acquisition comparison failed: {exc}')
        else:
            c1,c2=st.columns(2)
            shared_zmax=max(float(np.max(matched['area_image'])),
                            float(np.max(matched['virtual_pushbroom_image'])),1e-12)
            c1.plotly_chart(image_plot(matched['area_image'],matched['x_mm'],matched['y_mm'],
                         'Broad illumination · 2D camera','Area',zmax=shared_zmax),use_container_width=True)
            c2.plotly_chart(image_plot(matched['virtual_pushbroom_image'],matched['x_mm'],matched['y_mm'],
                         'Same illumination · sequential virtual rows','Area',zmax=shared_zmax),use_container_width=True)
            st.metric('Integrated transmitted-signal discrepancy',
                      f"{100*matched['diagnostics']['integrated_signal_relative_difference']:.2f}%")
            st.caption('Both images are normalized per launched photon packet in each MCX run. '
                       'Low detected counts can produce very noisy differences. This does not '
                       'prove agreement of an independently moving spotlight scanner.')
            with st.expander('Comparison diagnostics'):
                st.json(matched['diagnostics'])

if 'result' in st.session_state:
    r=st.session_state['result']
    image=r['image']
    x,y=r['x_mm'],r['scan_y_mm']
    recorded_mode=r.get('camera_mode',r['optics'].get('camera_mode','Pushbroom'))
    run_g=Geometry(**r['geometry'])
    run_o=Optics(**r['optics'])
    st.subheader(f'MCX {"pushbroom image" if recorded_mode=="Pushbroom" else "single 2D exposure"} · {st.session_state["wavelength"]} nm')
    st.caption(f'Captured using **{recorded_mode}** mode; {r["diagnostics"]["mcx_runs"]} MCX run(s). '
               'The image is computed from detected photons; optional outlines are independent **model geometry**, not measured boundaries.')
    if (asdict(g)!=r['geometry'] or asdict(o)!=r['optics'] or
            int(wl)!=int(st.session_state['wavelength']) or
            not np.array_equal(material,r['materials'])):
        st.warning('The simulation inputs have changed since this image was calculated. '
                   'The overlays below use the **saved geometry and source settings of the result**. '
                   'Rerun MCX to update the transmitted signal.')

    st.markdown('**Display overlays** — all contours are model truth/source settings, not detected edges')
    oc1,oc2,oc3,oc4=st.columns(4)
    show_egg=oc1.toggle('Full egg outline',value=False,key='overlay_egg')
    show_yolk=oc2.toggle('Yolk outline',value=False,key='overlay_yolk')
    show_source=oc3.toggle('Illumination outline',value=False,key='overlay_source')
    show_3d=oc4.toggle('Interactive 3D view',value=True,key='overlay_3d')
    if recorded_mode=='Pushbroom' and (show_source or show_3d):
        selected_index=st.slider('Show instantaneous source at scan line',0,len(y)-1,
                                 value=len(y)//2,step=1,key='overlay_selected_scan')
        selected_y=float(y[selected_index])
        st.caption(f'Illumination outline represents just one source position (line {selected_index+1}/{len(y)} '
                   f'at y={selected_y:.1f} mm); the finished pushbroom image combines **all** scan positions.')
    else:
        selected_y=0.
    output_figure=camera_image_figure(
        image,x,y,run_g,run_o,show_egg=show_egg,show_yolk=show_yolk,
        show_source=show_source,scan_y_mm=selected_y,
        title='MCX detected transmission + optional geometry overlays')
    st.plotly_chart(output_figure,use_container_width=True)
    if 'image_unfiltered' in r and run_o.camera_path_mode != 'All exits':
        with st.expander('Compare detected light BEFORE and AFTER direct-path filtering'):
            st.caption('Both plots use the same MCX photon exits and a shared linear colour scale. '
                       'This is an idealized path-history selection, not a simulated optical baffle.')
            raw=r['image_unfiltered']
            shared_max=max(float(np.max(raw)),float(np.max(image)),1e-12)
            cc1,cc2=st.columns(2)
            cc1.plotly_chart(image_plot(raw,x,y,'All top-face photon exits',recorded_mode,zmax=shared_max),use_container_width=True)
            cc2.plotly_chart(image_plot(image,x,y,'Egg-path-filtered photons',recorded_mode,zmax=shared_max),use_container_width=True)
            rr=r['diagnostics']
            st.caption(f'Excluded from the camera: {rr.get("camera_rejected_bypass_packets",0):,} '
                       'photons, representing '
                       f'{100*rr.get("camera_rejected_bypass_weight",0)/max(rr.get("weighted_camera_signal_unfiltered",0),1e-15):.1f}% '
                       'of weighted raw detector signal.')
    if show_egg or show_yolk or show_source:
        st.caption('Dashed outlines are geometric information added AFTER the MCX simulation: '
                   'egg/yolk = orthographic model silhouettes; illumination = source LAUNCH footprint '
                   'before scattering. None is a reconstructed or measured internal boundary.')

    showtruth=st.checkbox('Also show material ground truth as a separate image',False)
    if showtruth:
        m=projected_masks(run_g,x,y)
        truth=np.zeros_like(image)
        truth[m['egg']]=1;truth[m['yolk']]=2;truth[m['blood']]=3
        st.plotly_chart(image_plot(truth,x,y,'Material geometry · NOT a camera image',recorded_mode),
                        use_container_width=True)

    if show_3d:
        st.markdown('**3D model and simulated detector image**')
        sc1,sc2,sc3,sc4=st.columns(4)
        show_egg_3d=sc1.toggle('3D: egg envelope',value=True,key='show_3d_egg')
        show_yolk_3d=sc2.toggle('3D: yolk',value=True,key='show_3d_yolk')
        show_source_3d=sc3.toggle('3D: illumination',value=True,key='show_3d_source')
        show_plane=sc4.toggle('3D: camera output',value=True,key='show_3d_plane')
        scene=camera_scene_3d(image,x,y,run_g,run_o,show_egg=show_egg_3d,
                              show_yolk=show_yolk_3d,show_source=show_source_3d,
                              show_pattern=show_plane,scan_y_mm=selected_y)
        st.plotly_chart(scene,use_container_width=True)
        if recorded_mode=='Pushbroom':
            st.caption('3D display is a **conceptual assembly**: the image plane contains the '
                       'finished pushbroom reconstruction from multiple egg positions. '
                       'The illumination shown is only one selected scan position. '
                       'These did not all exist simultaneously during acquisition.')
        else:
            st.caption('3D display shows one stationary MCX area-camera exposure. '
                       'The colored upper plane is the photon-exit histogram, not a '
                       'camera-lens optical projection or a recovered 3D structure.')

    d=r['diagnostics']
    a,b,c=st.columns(3)
    a.metric('Photons launched',f'{d["launched_photons"]:,}')
    b.metric('Detected at upper MCX face',f'{d["top_exit_photons"]:,}')
    c.metric('Accepted by model camera',f'{d["camera_accepted_photons"]:,}')
    if 'camera_rejected_bypass_packets' in d:
        st.caption(f'History gate: **{d["camera_path_mode"]}** · '
                   f'excluded {d["camera_rejected_bypass_packets"]:,} camera-eligible photons '
                   'because their MCX material paths did not meet the selected rule. '
                   'Rejecting bypass photons computationally is not equivalent to placing a real baffle.')
    if d['camera_accepted_photons'] < 500:
        st.warning('Few accepted photons. Interpretation is Monte Carlo-noise dominated; increase the photon budget or camera acceptance.')
    if st.session_state.get('compared'):
        st.markdown('**Source comparison** · each image uses its own saved source geometry')
        a,b=st.columns(2)
        for ax,(name,rr) in zip((a,b),st.session_state['compared'].items()):
            ro=Optics(**rr['optics'])
            rg=Geometry(**rr['geometry'])
            xy=float(rr['scan_y_mm'][len(rr['scan_y_mm'])//2]) if ro.camera_mode=='Pushbroom' else 0.
            ax.plotly_chart(camera_image_figure(
                rr['image'],rr['x_mm'],rr['scan_y_mm'],rg,ro,
                show_egg=show_egg,show_yolk=show_yolk,show_source=show_source,
                scan_y_mm=xy,title=name),use_container_width=True)
    metadata={'engine':'MCX via PMCX','wavelength_nm':st.session_state['wavelength'],
              'optics':r['optics'],'geometry':r['geometry'],'materials':r['materials'].tolist(),
              'diagnostics':r['diagnostics']}
    buff=io.BytesIO()
    np.savez_compressed(buff,image=image,image_unfiltered=r.get('image_unfiltered',image),
                        x_mm=x,y_mm=y,scan_y_mm=y,metadata=json.dumps(metadata))
    st.download_button('Download raw signal (.npz)',buff.getvalue(),'mcx_egg_scan.npz')
    st.download_button('Download run metadata (.json)',json.dumps(metadata,indent=2),'mcx_egg_config.json')
else:
    st.info('Inspect the source schematic, then run a pushbroom scan or 2D area exposure on an MCX-capable GPU.')

with st.expander('Scientific scope and limitations'):
    st.markdown('''
- **MCX** performs physical photon scattering, absorption and interface transport on the voxel grid. No home-built Monte Carlo fallback exists.
- Native MCX **disk** source for spot; native **planar** for transverse, longitudinal, broad illumination. These are verified with unit tests.
- MCX detects photons leaving the **upper boundary face**. The pushbroom readout has a moving observation slit; the 2D area camera bins the full exit footprint in a **single static MCX run**. Both use a simplified angular gate. No physical lens, spectrograph, sensor, or camera noise is modeled.
- MCX has one shared focal-length parameter for these sources, not the older model’s independent x/y Gaussian divergences. No false equivalence is implied.
- The uniform voxel grid is sized from the **active** MCX source, camera region and egg; padding and a host-rasterization budget are adjustable. GPU memory and photon-record buffers require additional capacity. Check side-boundary convergence on real MCX before relying on quantitative output.
- The shell is rasterized in voxels, potentially coarse at ~0.35 mm thickness; at 1 mm voxel spacing the shell is **not reliable**, and even at 0.2 mm it is still under two nominal voxels thick.
- A stationary spot can look like a stripe in the **pushbroom composite** because it moves relative to the egg along y; the **area-camera** image is acquired at one static source position and should not display scan-induced elongation. Internal egg scattering may blur or deform either footprint.
- At other wavelengths, load experimentally measured optical coefficients. Source spectrum and detector spectral sensitivity are not included; signal is per launched photon packet.
- The 2D area camera is **monochromatic per run**. It is not a snapshot hyperspectral sensor; to model multiple wavelengths, repeat runs with appropriate coefficients.
- Image overlay curves and the 3D yolk/egg surfaces are **geometric ground truth**, explicitly separate from MCX photon transport. The 3D upper image plane is a visualization of camera output, NOT optical tomography or inferred boundaries.
- In pushbroom mode the 3D scene combines the completed scan with a **single instantaneous source position**; do not interpret it as a simultaneous exposure.
- No claim is made of having experimentally validated yolk/blood boundaries. Run the known-answer Beer–Lambert test in `tests/test_mcx_physics.py` on your GPU.
''')
