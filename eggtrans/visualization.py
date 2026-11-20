"""Plotly diagnostics for MCX camera images and **separate** geometric truth.

No light transport, optical projection, reconstruction or image simulation occurs
here. A simulated image is used only as supplied numeric data; ellipsoids and
source footprints originate from Geometry and Optics. Source outlines mark the
launch footprint, not an optical shadow or measured irradiance contour.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .core import Geometry, Optics
from .mcx_engine import source_footprint_mm


EGG_COLOR = "#92b2cf"
YOLK_COLOR = "#ffd166"
SOURCE_COLOR = "#4ee2d5"


def _validate_image(image, x_mm, y_mm):
    image = np.asarray(image, dtype=float)
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(y_mm, dtype=float)
    if image.ndim != 2 or x.ndim != 1 or y.ndim != 1 or image.shape != (len(y), len(x)):
        raise ValueError("MCX camera image must be (len(y_mm), len(x_mm)).")
    if len(x) < 2 or len(y) < 2 or not np.isfinite(image).all():
        raise ValueError("Camera axes and finite image intensities are required.")
    return image, x, y


def outline_xy(g: Geometry, which: str, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic xy silhouette of an ellipsoid; *not* MCX observations."""
    t = np.linspace(0., 2. * np.pi, n, endpoint=True)
    if which == "egg":
        return (g.egg_width_mm / 2 * np.cos(t),
                g.egg_length_mm / 2 * np.sin(t))
    if which == "yolk":
        return (g.yolk_offset_x_mm + g.yolk_radius_x_mm * np.cos(t),
                g.yolk_offset_y_mm + g.yolk_radius_y_mm * np.sin(t))
    raise ValueError(f"Unknown geometric outline: {which}")


def source_outline_xy(o: Optics, scan_y_mm: float = 0., n: int = 180):
    """Planar MCX launch footprint at one instant in the egg coordinates."""
    fp = source_footprint_mm(o, scan_y_mm)
    if fp["kind"] == "circle":
        t = np.linspace(0, 2 * np.pi, n)
        return (fp["cx"] + fp["radius"] * np.cos(t),
                fp["cy"] + fp["radius"] * np.sin(t))
    return (np.array([fp["xmin"], fp["xmax"], fp["xmax"], fp["xmin"], fp["xmin"]]),
            np.array([fp["ymin"], fp["ymin"], fp["ymax"], fp["ymax"], fp["ymin"]]))


def camera_image_figure(image, x_mm, y_mm, g: Geometry, o: Optics, *,
                        show_egg=False, show_yolk=False, show_source=False,
                        scan_y_mm: float = 0., title="MCX detected signal") -> go.Figure:
    """Detected signal with optional *explicitly identified* truth contours.

    In pushbroom mode, the source outline is one instantaneous launch footprint
    at scan_y_mm; the image is the complete time-assembled scan. In area
    mode, source geometry is fixed during the exposure.
    """
    im, x, y = _validate_image(image, x_mm, y_mm)
    fig = go.Figure(go.Heatmap(x=x, y=y, z=im, colorscale="Inferno",
                               colorbar=dict(title="Weighted signal"),
                               name="MCX output", hovertemplate="x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>signal=%{z:.3g}<extra>MCX</extra>"))
    if show_egg:
        xx, yy = outline_xy(g, "egg")
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="Egg geometry · truth",
                                 line=dict(color=EGG_COLOR, width=2.5, dash="dash"),
                                 hovertemplate="Egg model outline (not detected)<extra></extra>"))
    if show_yolk and g.include_yolk:
        xx, yy = outline_xy(g, "yolk")
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", name="Yolk geometry · truth",
                                 line=dict(color=YOLK_COLOR, width=2.8, dash="dot"),
                                 hovertemplate="Yolk model outline (not detected)<extra></extra>"))
    if show_source:
        xx, yy = source_outline_xy(o, scan_y_mm)
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines",
                                 name="Source launch footprint · not transmitted pattern",
                                 line=dict(color=SOURCE_COLOR, width=2.8, dash="dash"),
                                 hovertemplate="Source footprint BEFORE egg (not detected)<extra></extra>"))
        if o.camera_mode == "Pushbroom":
            fig.add_hline(y=scan_y_mm, line=dict(color=SOURCE_COLOR, width=1, dash="dot"),
                          annotation_text="Selected scan line", annotation_position="top right")
    ylabel = "Scan coordinate y (mm)" if o.camera_mode == "Pushbroom" else "Area camera y (mm)"
    fig.update_layout(title=title, xaxis_title="x (mm)", yaxis_title=ylabel,
                      height=530, margin=dict(l=45, r=25, t=55, b=50),
                      legend=dict(orientation="h", y=-.22), hovermode="closest")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _ellipsoid_surface(center, radii, ntheta=60, nphi=32):
    # Full three-dimensional ellipsoid with both its top and bottom surfaces.
    theta = np.linspace(0., 2. * np.pi, ntheta)
    phi = np.linspace(0., np.pi, nphi)
    th, ph = np.meshgrid(theta, phi)
    xx = center[0] + radii[0] * np.sin(ph) * np.cos(th)
    yy = center[1] + radii[1] * np.sin(ph) * np.sin(th)
    zz = center[2] + radii[2] * np.cos(ph)
    return xx, yy, zz


def camera_scene_3d(image, x_mm, y_mm, g: Geometry, o: Optics, *,
                    show_egg=True, show_yolk=True, show_source=True,
                    show_pattern=True, scan_y_mm: float = 0.) -> go.Figure:
    """3D model geometry and MCX output texture shown on a *conceptual* plane.

    The output is a static image mapped to the +z detector plane, not a render
    of rays, a camera lens or 3D optical tomography. For pushbroom it is a
    time-assembled virtual image plane, never a single physical exposure.
    """
    im, x, y = _validate_image(image, x_mm, y_mm)
    z_source = -g.egg_height_mm / 2 - o.source_gap_mm
    # Matches the z face of the box in build_egg_volume, with the requested
    # upper camera clearance (+2 voxels). Minor <voxel rounding is immaterial
    # to this *visualization*, not to photon transport.
    z_pattern = g.egg_height_mm / 2 + o.camera_plane_gap_mm + 2 * o.voxel_mm
    fig = go.Figure()

    if show_egg:
        xx, yy, zz = _ellipsoid_surface((0., 0., 0.),
                                       (g.egg_width_mm/2, g.egg_length_mm/2, g.egg_height_mm/2))
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, surfacecolor=np.zeros_like(xx),
                                 colorscale=[[0, EGG_COLOR], [1, EGG_COLOR]],
                                 opacity=.15, showscale=False, name="3D egg geometry",
                                 showlegend=True, hoverinfo="skip"))
    if show_yolk and g.include_yolk:
        xx, yy, zz = _ellipsoid_surface(
            (g.yolk_offset_x_mm, g.yolk_offset_y_mm, g.yolk_offset_z_mm),
            (g.yolk_radius_x_mm, g.yolk_radius_y_mm, g.yolk_radius_z_mm))
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, surfacecolor=np.zeros_like(xx),
                                 colorscale=[[0, YOLK_COLOR], [1, YOLK_COLOR]],
                                 opacity=.7, showscale=False, name="3D yolk geometry · truth",
                                 showlegend=True, hoverinfo="skip"))

    if show_source:
        sx, sy = source_outline_xy(o, scan_y_mm, n=75)
        # Plotly Scatter3d cannot fill polygons; use Mesh3d with a centre fan.
        # Drop repeated final boundary point before closing the fan.
        bx, by = np.asarray(sx[:-1]), np.asarray(sy[:-1])
        n = len(bx)
        cx, cy = float(np.mean(bx)), float(np.mean(by))
        vx = np.r_[cx, bx]
        vy = np.r_[cy, by]
        vz = np.full(n + 1, z_source)
        fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz,
                                i=np.zeros(n, dtype=int),
                                j=np.arange(1, n+1),
                                k=np.r_[np.arange(2, n+1), 1],
                                color=SOURCE_COLOR, opacity=.4,
                                showlegend=True, name="MCX source launch footprint",
                                hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=sx, y=sy, z=np.full(len(sx), z_source),
                                   mode="lines", line=dict(color=SOURCE_COLOR, width=6),
                                   name="Source outline", showlegend=False))

    if show_pattern:
        X, Y = np.meshgrid(x, y)
        lo, hi = float(np.min(im)), float(np.max(im))
        if hi <= lo:
            hi = lo + 1e-12
        fig.add_trace(go.Surface(
            x=X, y=Y, z=np.full_like(im, z_pattern), surfacecolor=im,
            colorscale="Inferno", cmin=lo, cmax=hi, opacity=.93,
            showscale=True, colorbar=dict(title="Weighted signal", len=.54, x=1.02),
            name="MCX output mapped to detector plane", showlegend=True,
            hovertemplate="Camera x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>MCX image plane<extra></extra>"))

    fig.update_layout(
        title=("3D egg and reconstructed pushbroom image (assembled over time)"
               if o.camera_mode == "Pushbroom" else
               "3D egg and single-exposure area camera image"),
        height=660, margin=dict(l=0, r=0, b=0, t=52),
        legend=dict(orientation="h", y=-.08),
        scene=dict(
            xaxis=dict(title="x (mm)", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="y (mm)", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="Optical axis z (mm)", backgroundcolor="rgba(0,0,0,0)"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.50, y=-1.90, z=1.15)),
        ),
        uirevision="camera-scene-v1",
    )
    return fig
