"""Interactive Plotly apparatus schematic, tied to actual MCX source footprint."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .core import Geometry,Optics
from .mcx_engine import source_footprint_mm


def apparatus_figure(g:Geometry,o:Optics,scan_y_mm:float=0.)->go.Figure:
    fp=source_footprint_mm(o,scan_y_mm)
    f=make_subplots(rows=1,cols=2,subplot_titles=(
        'Top view · source footprint at one scan position',
        'Side view · camera above, source below'),horizontal_spacing=.10)
    t=np.linspace(0,2*np.pi,180)
    f.add_trace(go.Scatter(x=g.egg_width_mm/2*np.cos(t),y=g.egg_length_mm/2*np.sin(t),
                           mode='lines',line=dict(color='#73859c',width=3),name='Egg outline'),1,1)
    f.add_trace(go.Scatter(x=g.yolk_offset_x_mm+g.yolk_radius_x_mm*np.cos(t),
                           y=g.yolk_offset_y_mm+g.yolk_radius_y_mm*np.sin(t),mode='lines',
                           line=dict(color='#d99e2e',dash='dot',width=2),name='Yolk geometry (not detected)'),1,1)
    if fp['kind']=='circle':
        fx=fp['cx']+fp['radius']*np.cos(t)
        fy=fp['cy']+fp['radius']*np.sin(t)
    else:
        fx=[fp['xmin'],fp['xmax'],fp['xmax'],fp['xmin'],fp['xmin']]
        fy=[fp['ymin'],fp['ymin'],fp['ymax'],fp['ymax'],fp['ymin']]
    f.add_trace(go.Scatter(x=fx,y=fy,fill='toself',fillcolor='rgba(239,151,52,.35)',
                           line=dict(color='#e88123',width=2),name=f'{o.mode} · MCX emitter'),1,1)
    if o.camera_mode == 'Pushbroom':
        f.add_trace(go.Scatter(x=[-g.egg_width_mm*.57,g.egg_width_mm*.57],
                               y=[scan_y_mm,scan_y_mm], mode='lines',
                               line=dict(color='#0c9ee4',dash='dash',width=3),
                               name='Detector line'),1,1)
        f.add_annotation(x=g.egg_width_mm*.65,y=scan_y_mm+9,ax=g.egg_width_mm*.65,
                         ay=scan_y_mm-5,arrowhead=3,arrowsize=1.4,arrowwidth=2,
                         arrowcolor='#62748b',text='Translation ↑',row=1,col=1)
    else:
        xf=g.egg_width_mm*.56; yf=g.egg_length_mm*.54
        f.add_trace(go.Scatter(x=[-xf,xf,xf,-xf,-xf],y=[-yf,-yf,yf,yf,-yf],
                               mode='lines',line=dict(color='#0c9ee4',dash='dash',width=3),
                               name='Area camera FOV (2D)'),1,1)

    # side view of same egg in y-z coordinates; overlay source located beneath egg
    f.add_trace(go.Scatter(x=g.egg_length_mm/2*np.cos(t),y=g.egg_height_mm/2*np.sin(t),
                           mode='lines',line=dict(color='#73859c',width=3),
                           showlegend=False),1,2)
    f.add_trace(go.Scatter(x=g.yolk_offset_y_mm+g.yolk_radius_y_mm*np.cos(t),
                           y=g.yolk_offset_z_mm+g.yolk_radius_z_mm*np.sin(t),
                           mode='lines',line=dict(color='#d99e2e',dash='dot',width=2),
                           showlegend=False),1,2)
    # source width along y depends on source geometry
    if fp['kind']=='circle':
        source_min,source_max=fp['cy']-fp['radius'],fp['cy']+fp['radius']
    else:
        source_min,source_max=fp['ymin'],fp['ymax']
    bottom=-g.egg_height_mm/2-o.source_gap_mm
    top=g.egg_height_mm/2+o.camera_plane_gap_mm
    f.add_trace(go.Scatter(x=[source_min,source_max],y=[bottom,bottom],mode='lines',
                           line=dict(color='#e88123',width=7),showlegend=False),1,2)
    camera_half_width = 10. if o.camera_mode == 'Pushbroom' else g.egg_length_mm*.54
    f.add_trace(go.Scatter(x=[scan_y_mm-camera_half_width,scan_y_mm+camera_half_width],y=[top,top],mode='lines',
                           line=dict(color='#0c9ee4',width=6),showlegend=False),1,2)
    for yy in np.linspace(source_min,source_max,3):
        f.add_trace(go.Scatter(x=[yy,yy],y=[bottom,-g.egg_height_mm*.2],mode='lines',
                               line=dict(color='rgba(238,150,36,.45)',width=2),showlegend=False),1,2)
    f.add_annotation(x=scan_y_mm+12,y=top,text=('Line camera' if o.camera_mode == 'Pushbroom' else 'Area camera'),showarrow=False,
                     font=dict(size=11,color='#0c9ee4'),row=1,col=2)
    f.update_xaxes(title_text='Across x (mm)',scaleanchor='y',scaleratio=1,row=1,col=1)
    f.update_yaxes(title_text='Translation y (mm)',row=1,col=1)
    f.update_xaxes(title_text='Translation y (mm)',scaleanchor='y2',scaleratio=1,row=1,col=2)
    f.update_yaxes(title_text='Optical z (mm)',row=1,col=2)
    f.update_layout(height=420,margin=dict(t=50,l=36,r=20,b=36),
                    legend=dict(orientation='h',y=-.27),dragmode='pan')
    return f
