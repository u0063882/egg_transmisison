"""Numba-accelerated Monte Carlo egg transmission *research prototype*.

The optical model uses exponential free paths, Henyey–Greenstein bulk
scattering, Beer absorption, Snell refraction and Fresnel reflection.
An idealised angularly gated/telecentric *line* detector records upward
photons. It is not a validated instrument model or a substitute for MCX.
All spatial units are millimetres; coefficients are inverse millimetres.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal
import numpy as np
from numba import njit

Mode = Literal['Transverse line', 'Spot', 'Longitudinal line', 'Broad backlight']
MODE_NAMES: tuple[Mode, ...] = ('Transverse line', 'Spot', 'Longitudinal line', 'Broad backlight')


@dataclass
class Geometry:
    egg_width_mm: float = 44.0             # x = transverse to conveyor
    egg_length_mm: float = 59.0            # y = translation direction
    egg_height_mm: float = 44.0            # z = optical axis
    shell_thickness_mm: float = 0.35       # approximate concentric thickness
    yolk_radius_x_mm: float = 14.0
    yolk_radius_y_mm: float = 16.0
    yolk_radius_z_mm: float = 14.0
    yolk_offset_x_mm: float = -1.5
    yolk_offset_y_mm: float = 0.0
    yolk_offset_z_mm: float = -1.0
    blood_x_mm: float = 6.0
    blood_y_mm: float = 13.0
    blood_z_mm: float = 9.0
    blood_radius_mm: float = 2.3
    include_blood: bool = True


@dataclass
class Optics:
    mode: Mode = 'Transverse line'
    line_length_mm: float = 48.0
    line_width_mm: float = 2.0
    spot_diameter_mm: float = 6.0
    source_offset_y_mm: float = 0.0
    source_gap_mm: float = 2.0
    divergence_x_deg: float = 8.0         # angular std of slope, not full cone
    divergence_y_deg: float = 3.0
    camera_half_angle_deg: float = 40.0   # idealised angular gate
    camera_line_width_mm: float = 4.0
    camera_plane_gap_mm: float = 1.0
    photons_per_line: int = 12000
    scan_lines: int = 64
    camera_pixels: int = 72
    max_scattering_events: int = 650
    seed: int = 42


# Rows: air, shell, albumen, yolk, blood. Columns: mu_a, mu_s, HG g, n.
# IMPORTANT: only albumen and yolk first three columns are anchored to the
# 800 nm measured values in Tamura et al. (2026). ALL OTHER values illustrative.
# Optical coefficients for 400–1700 nm must be measured or supplied by user.
MATERIALS_800 = np.array([
    [0.000, 0.000, 0.00, 1.000],
    [0.040, 2.000, 0.85, 1.530],    # shell: illustrative placeholder
    [0.006, 0.069, 0.90, 1.350],    # measured albumen at 800nm (mu,g)
    [0.014, 5.350, 0.92, 1.400],    # measured yolk at 800nm (mu,g)
    [0.700, 1.000, 0.90, 1.350],    # blood: illustrative placeholder
], dtype=np.float64)
MATERIAL_NAMES = ('Air', 'Shell', 'Albumen', 'Yolk', 'Blood spot')


def validate(geometry: Geometry, optics: Optics, materials: np.ndarray) -> None:
    a = geometry
    r = (a.egg_width_mm / 2, a.egg_length_mm / 2, a.egg_height_mm / 2)
    if min(r) <= a.shell_thickness_mm or a.shell_thickness_mm <= 0:
        raise ValueError('Egg must be larger than shell thickness.')
    if min(a.yolk_radius_x_mm, a.yolk_radius_y_mm, a.yolk_radius_z_mm) <= 0:
        raise ValueError('Yolk radii must be positive.')
    for center, radius, outer in zip(
        (a.yolk_offset_x_mm, a.yolk_offset_y_mm, a.yolk_offset_z_mm),
        (a.yolk_radius_x_mm, a.yolk_radius_y_mm, a.yolk_radius_z_mm), r
    ):
        if abs(center) + radius >= outer - a.shell_thickness_mm:
            raise ValueError('Yolk is not enclosed by the inner shell. Adjust dimensions.')
    if a.include_blood:
        pos = np.array([a.blood_x_mm, a.blood_y_mm, a.blood_z_mm])
        if np.sum((pos / (np.array(r) - a.shell_thickness_mm)) ** 2) >= 0.9:
            raise ValueError('Blood spot must lie inside the shell with a margin.')
        if np.sum(((pos - np.array([a.yolk_offset_x_mm, a.yolk_offset_y_mm, a.yolk_offset_z_mm])) /
                   np.array([a.yolk_radius_x_mm, a.yolk_radius_y_mm, a.yolk_radius_z_mm])) ** 2) <= 1:
            raise ValueError('Blood spot center must be outside the yolk in this model.')
    if materials.shape != (5, 4):
        raise ValueError('Material matrix must be shape (5, 4).')
    if np.any(~np.isfinite(materials)) or np.any(materials[:, :2] < 0) or np.any(np.abs(materials[:, 2]) >= 1) or np.any(materials[:, 3] < 1):
        raise ValueError('Invalid material coefficients (mu >= 0; |g| < 1; n >= 1).')
    if optics.mode not in MODE_NAMES:
        raise ValueError('Unknown illumination mode.')
    if min(optics.line_length_mm, optics.line_width_mm, optics.spot_diameter_mm) <= 0:
        raise ValueError('All emitter dimensions must be positive.')
    if optics.scan_lines < 2 or optics.camera_pixels < 2 or optics.photons_per_line < 1:
        raise ValueError('Invalid scan dimensions / photon count.')
    if optics.max_scattering_events < 1:
        raise ValueError('max_scattering_events must be >= 1.')
    if not 0 < optics.camera_half_angle_deg < 89:
        raise ValueError('Camera half-angle must be between 0 and 89 degrees.')
    if min(optics.camera_line_width_mm, optics.source_gap_mm, optics.camera_plane_gap_mm) <= 0:
        raise ValueError('Camera/illumination clearances and line width must be positive.')
    if min(optics.divergence_x_deg, optics.divergence_y_deg) < 0 or max(optics.divergence_x_deg, optics.divergence_y_deg) > 60:
        raise ValueError('Angular divergence SD must be 0–60 degrees.')


def _geometry_vector(g: Geometry) -> np.ndarray:
    return np.array([
        g.egg_width_mm / 2, g.egg_length_mm / 2, g.egg_height_mm / 2,
        g.shell_thickness_mm, g.yolk_radius_x_mm, g.yolk_radius_y_mm,
        g.yolk_radius_z_mm, g.yolk_offset_x_mm, g.yolk_offset_y_mm,
        g.yolk_offset_z_mm, g.blood_x_mm, g.blood_y_mm, g.blood_z_mm,
        g.blood_radius_mm, float(g.include_blood)
    ], dtype=np.float64)


@njit(cache=True)
def _inside(x, y, z, cx, cy, cz, rx, ry, rz):
    return ((x-cx)/rx)**2 + ((y-cy)/ry)**2 + ((z-cz)/rz)**2 < 1.0


@njit(cache=True)
def _medium(x, y, z, q):
    if not _inside(x,y,z,0.,0.,0.,q[0],q[1],q[2]):
        return 0
    if not _inside(x,y,z,0.,0.,0.,q[0]-q[3],q[1]-q[3],q[2]-q[3]):
        return 1
    if q[14] > 0.5 and _inside(x,y,z,q[10],q[11],q[12],q[13],q[13],q[13]):
        return 4
    if _inside(x,y,z,q[7],q[8],q[9],q[4],q[5],q[6]):
        return 3
    return 2


@njit(cache=True)
def _ellipsoid_hit(x,y,z,dx,dy,dz,cx,cy,cz,rx,ry,rz):
    u,v,w = (x-cx)/rx,(y-cy)/ry,(z-cz)/rz
    a,b,c = dx/rx,dy/ry,dz/rz
    aa = a*a+b*b+c*c
    bb = u*a+v*b+w*c
    cc = u*u+v*v+w*w-1.0
    discr = bb*bb-aa*cc
    if discr < 0 or aa <= 0:
        return 1.e30
    root = np.sqrt(discr)
    near = (-bb-root)/aa
    far = (-bb+root)/aa
    if near > 2.e-6: return near
    if far > 2.e-6: return far
    return 1.e30


@njit(cache=True)
def _next_boundary(x,y,z,dx,dy,dz,q):
    dist = 1.e30
    boundary = -1
    for k in range(4):
        if k == 3 and q[14] <= 0.5:
            continue
        if k == 0:
            cx,cy,cz,rx,ry,rz = 0.,0.,0.,q[0],q[1],q[2]
        elif k == 1:
            cx,cy,cz,rx,ry,rz = 0.,0.,0.,q[0]-q[3],q[1]-q[3],q[2]-q[3]
        elif k == 2:
            cx,cy,cz,rx,ry,rz = q[7],q[8],q[9],q[4],q[5],q[6]
        else:
            cx,cy,cz,rx,ry,rz = q[10],q[11],q[12],q[13],q[13],q[13]
        d = _ellipsoid_hit(x,y,z,dx,dy,dz,cx,cy,cz,rx,ry,rz)
        if d < dist:
            dist,boundary = d,k
    return dist,boundary


@njit(cache=True)
def _normal(x,y,z,k,q):
    if k == 0:
        cx,cy,cz,rx,ry,rz = 0.,0.,0.,q[0],q[1],q[2]
    elif k == 1:
        cx,cy,cz,rx,ry,rz = 0.,0.,0.,q[0]-q[3],q[1]-q[3],q[2]-q[3]
    elif k == 2:
        cx,cy,cz,rx,ry,rz = q[7],q[8],q[9],q[4],q[5],q[6]
    else:
        cx,cy,cz,rx,ry,rz = q[10],q[11],q[12],q[13],q[13],q[13]
    nx,ny,nz = (x-cx)/(rx*rx),(y-cy)/(ry*ry),(z-cz)/(rz*rz)
    nn = np.sqrt(nx*nx+ny*ny+nz*nz)
    return nx/nn,ny/nn,nz/nn


@njit(cache=True)
def _hg_direction(dx,dy,dz,g):
    r = np.random.random()
    if np.abs(g) < 1.e-8:
        c = 2*r - 1
    else:
        v = (1-g*g)/(1-g+2*g*r)
        c = (1+g*g-v*v)/(2*g)
    c = max(-1., min(1., c))
    s = np.sqrt(max(0., 1-c*c))
    phi = 2*np.pi*np.random.random()
    if abs(dz) < 0.999:
        nn=np.sqrt(dx*dx+dy*dy)
        ux,uy,uz = -dy/nn,dx/nn,0.
    else:
        ux,uy,uz=1.,0.,0.
    vx,vy,vz = dy*uz-dz*uy, dz*ux-dx*uz, dx*uy-dy*ux
    cp,sp=np.cos(phi),np.sin(phi)
    ox = c*dx+s*(cp*ux+sp*vx)
    oy = c*dy+s*(cp*uy+sp*vy)
    oz = c*dz+s*(cp*uz+sp*vz)
    return ox,oy,oz


@njit(cache=True)
def _refract_or_reflect(dx,dy,dz,nx,ny,nz,n1,n2):
    # Orient surface normal along the incoming direction (towards medium 2).
    ci = dx*nx+dy*ny+dz*nz
    if ci < 0.:
        nx,ny,nz,ci=-nx,-ny,-nz,-ci
    ratio = n1/n2
    sin2_t = ratio*ratio*(1.-ci*ci)
    if sin2_t >= 1.:
        R = 1.
    else:
        ct=np.sqrt(max(0.,1.-sin2_t))
        rs=(n1*ci-n2*ct)/(n1*ci+n2*ct)
        rp=(n2*ci-n1*ct)/(n2*ci+n1*ct)
        R=0.5*(rs*rs+rp*rp)
    if np.random.random() < R:
        return dx-2*ci*nx, dy-2*ci*ny, dz-2*ci*nz, False
    f = ct-ratio*ci
    return ratio*dx+f*nx,ratio*dy+f*ny,ratio*dz+f*nz,True


@njit(cache=True)
def _run_mc(q, mu, beam, mode, photon_count, nx, ny, max_events, seed):
    np.random.seed(seed)
    out=np.zeros((ny,nx),dtype=np.float64)
    diag=np.zeros(6,dtype=np.float64)  # launched; upward exits; camera hits; recorded weighted; zero-scatter weighted; truncated
    xmax=q[0] * 1.12
    ystep=2*q[1] * 1.10 / ny
    angle_tan=np.tan(np.deg2rad(beam[8]))
    sig_x=np.tan(np.deg2rad(beam[6]))
    sig_y=np.tan(np.deg2rad(beam[7]))
    top_plane=q[2] + beam[10]
    for row in range(ny):
        scan_y = -q[1]*1.10 + (row+0.5)*ystep
        for pp in range(photon_count):
            diag[0]+=1
            if mode == 0: # transverse x-directed line
                x=(np.random.random()-.5)*beam[0]
                y=scan_y+beam[4]+(np.random.random()-.5)*beam[1]
            elif mode == 1: # circular spot
                rr=np.sqrt(np.random.random())*beam[2]/2
                aa=2*np.pi*np.random.random()
                x=rr*np.cos(aa)
                y=scan_y+beam[4]+rr*np.sin(aa)
            elif mode == 2: # longitudinal y-directed line
                x=(np.random.random()-.5)*beam[1]
                y=scan_y+beam[4]+(np.random.random()-.5)*beam[0]
            else: # broad rectangular source
                x=(np.random.random()-.5)*max(beam[0],2*q[0]*1.1)
                y=scan_y+beam[4]+(np.random.random()-.5)*max(beam[0],2*q[1]*1.1)
            z=-q[2]-beam[5]
            dx=np.random.normal(0.,sig_x)
            dy=np.random.normal(0.,sig_y)
            dz=1.
            dn=np.sqrt(dx*dx+dy*dy+1.)
            dx,dy,dz=dx/dn,dy/dn,dz/dn
            mat=0
            weight=1.
            scatter_count=0
            # Guard air propagation and scattering; max events includes boundaries.
            for event in range(max_events):
                d_edge,which=_next_boundary(x,y,z,dx,dy,dz,q)
                if which < 0 or d_edge > 1.e6:
                    break
                mu_s=mu[mat,1]
                d_sc=-np.log(max(1.e-12,np.random.random()))/mu_s if mu_s > 1.e-15 else 1.e30
                if d_sc < d_edge:
                    weight *= np.exp(-mu[mat,0]*d_sc)
                    x+=d_sc*dx; y+=d_sc*dy; z+=d_sc*dz
                    scatter_count+=1
                    dx,dy,dz=_hg_direction(dx,dy,dz,mu[mat,2])
                else:
                    weight *= np.exp(-mu[mat,0]*d_edge)
                    x+=d_edge*dx; y+=d_edge*dy; z+=d_edge*dz
                    newmat=_medium(x+2.e-4*dx,y+2.e-4*dy,z+2.e-4*dz,q)
                    if newmat != mat:
                        nx_,ny_,nz_=_normal(x,y,z,which,q)
                        dx,dy,dz,crossed=_refract_or_reflect(dx,dy,dz,nx_,ny_,nz_,mu[mat,3],mu[newmat,3])
                        if crossed:
                            mat=newmat
                            if mat == 0:  # final exit to air; upward only
                                if dz > 0 and z > 0:
                                    diag[1]+=weight
                                    tx=dx/dz;ty=dy/dz
                                    if tx*tx+ty*ty <= angle_tan*angle_tan:
                                        jump=max(0.,top_plane-z)
                                        xx=x+dx/dz*jump
                                        yy=y+dy/dz*jump
                                        if abs(yy-scan_y) <= beam[9]/2 and abs(xx) < xmax:
                                            index=int((xx+xmax)/(2*xmax)*nx)
                                            if 0<=index<nx:
                                                out[row,index]+=weight
                                                diag[2]+=1
                                                diag[3]+=weight
                                                if scatter_count == 0:
                                                    diag[4]+=weight
                                break
                    # Step epsilon into the correct region after boundary event.
                    x+=3.e-4*dx; y+=3.e-4*dy; z+=3.e-4*dz
                if weight < 1.e-6:
                    # Roulette to keep low-weight trajectories statistically unbiased.
                    if np.random.random() > 0.1:
                        break
                    weight*=10.
            else:
                diag[5]+=1
    return out,diag


def simulate(geometry: Geometry | None = None,
             optics: Optics | None = None,
             materials: np.ndarray | None = None) -> dict:
    """Run forward packet transport and return absolute relative detected flux.

    Photon counts are equal *launched* photons per row, not equal detected
    signal. The result is thus affected by source power/shape, camera angular
    gate, and scatter. No invented per-wavelength optical coefficients.
    """
    g=geometry or Geometry()
    o=optics or Optics()
    m=np.array(MATERIALS_800 if materials is None else materials, dtype=np.float64)
    validate(g,o,m)
    beam=np.array([o.line_length_mm,o.line_width_mm,o.spot_diameter_mm,
                   0.,o.source_offset_y_mm,o.source_gap_mm,
                   o.divergence_x_deg,o.divergence_y_deg,
                   o.camera_half_angle_deg,o.camera_line_width_mm,
                   o.camera_plane_gap_mm], dtype=np.float64)
    img,diagnostics=_run_mc(_geometry_vector(g),m,beam,MODE_NAMES.index(o.mode),
                            o.photons_per_line,o.camera_pixels,o.scan_lines,
                            o.max_scattering_events,o.seed)
    img=img / o.photons_per_line
    return {'image':img, 'x_mm':np.linspace(-g.egg_width_mm/2*1.12,g.egg_width_mm/2*1.12,o.camera_pixels),
            'scan_y_mm':np.linspace(-g.egg_length_mm/2*1.10,g.egg_length_mm/2*1.10,o.scan_lines),
            'diagnostics':{'launched_photons':int(diagnostics[0]),
                          'weighted_upward_exit':float(diagnostics[1]),
                          'detected_ray_count':int(diagnostics[2]),
                          'weighted_camera_signal':float(diagnostics[3]),
                          'weighted_ballistic_signal':float(diagnostics[4]),
                          'max_event_terminated':int(diagnostics[5])},
            'geometry':asdict(g), 'optics':asdict(o), 'materials':m.copy()}


def projected_masks(g: Geometry, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Perfect geometric top projections, NOT predicted optical visibility."""
    xx,yy=np.meshgrid(x,y)
    return {'egg':(xx/(g.egg_width_mm/2))**2+(yy/(g.egg_length_mm/2))**2<=1,
            'yolk':((xx-g.yolk_offset_x_mm)/g.yolk_radius_x_mm)**2+
                   ((yy-g.yolk_offset_y_mm)/g.yolk_radius_y_mm)**2<=1,
            'blood':((xx-g.blood_x_mm)**2+(yy-g.blood_y_mm)**2<=g.blood_radius_mm**2)}