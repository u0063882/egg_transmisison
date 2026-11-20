"""PMCX adapter: geometry/source definitions and detected-photon camera binning.

All photon physics is performed by the established MCX engine through `pmcx`.
This module DOES NOT compute optical paths or simulate scattering. MCX records
photons exiting the +z domain face; Python bins them into a moving pushbroom
receiver's slit and angular acceptance, or into a stationary 2D area image.

MCX docs: https://github.com/fangq/mcx, `mcxlab/README.txt`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import numpy as np

from .core import Geometry, Optics, MATERIALS_800, validate


@dataclass
class VoxelDomain:
    vol: np.ndarray  # MCX labels: 1 air, 2 shell, 3 albumen, 4 yolk, 5 blood
    grid_mm: float
    origin_mm: tuple[float,float,float]
    props: np.ndarray
    plan: "DomainPlan | None" = None

    @property
    def dims(self) -> tuple[int,int,int]:
        return tuple(int(a) for a in self.vol.shape)

    def mm_to_voxel(self, xyz):
        return (np.asarray(xyz,dtype=float)-np.asarray(self.origin_mm))/self.grid_mm

    def voxel_to_mm(self, xyz):
        return np.asarray(xyz,dtype=float)*self.grid_mm+np.asarray(self.origin_mm)


@dataclass(frozen=True)
class DomainPlan:
    """Preflight geometry estimate: dimensions/limits without allocating voxels.

    host_build_estimate_mib is an approximate upper-range planning allowance for
    Python temporary arrays during ellipsoid rasterization, not a GPU forecast.
    The 1-byte/voxel labels and detector buffers are reported separately.
    """
    dims: tuple[int, int, int]
    origin_mm: tuple[float, float, float]
    voxel_size_mm: float
    nvoxels: int
    label_memory_mib: float
    host_build_estimate_mib: float
    detected_buffer_estimate_mib: float
    domain_padding_mm: float
    active_source_mode: str
    acquisition: str
    x_bounds_mm: tuple[float, float]
    y_bounds_mm: tuple[float, float]

    @property
    def grid_mm(self) -> float:
        return self.voxel_size_mm

    def mm_to_voxel(self, xyz):
        return (np.asarray(xyz,dtype=float)-np.asarray(self.origin_mm))/self.grid_mm


def scan_positions_mm(g: Geometry, o: Optics) -> np.ndarray:
    """Position of the source/detector pair in the egg-relative coordinate frame."""
    if o.camera_mode == 'Area':
        return np.array([0.], dtype=float)
    return np.linspace(-g.egg_length_mm*.54, g.egg_length_mm*.54, o.scan_lines)


def domain_plan(g: Geometry, o: Optics, materials: np.ndarray = MATERIALS_800,
                *, enforce_budget: bool = True) -> DomainPlan:
    """Source-specific bounded air volume, sized once for a complete scan.

    Include the egg, camera FOV, and the *active* MCX source at both endpoints
    of its motion. All sources move linearly along y, so endpoints suffice.
    For directional/divergent light, lateral padding is a user-controlled
    modelling parameter: a larger value may be needed to avoid lost photons.
    Never change the selected voxel size silently.
    """
    validate(g, o, materials)
    x_half_cam = g.egg_width_mm * .56
    y_half_cam = g.egg_length_mm * .54
    v = float(o.voxel_mm)
    xlo = -max(g.egg_width_mm/2, x_half_cam)
    xhi = max(g.egg_width_mm/2, x_half_cam)
    y_extent = y_half_cam + (o.camera_line_width_mm/2 if o.camera_mode == 'Pushbroom' else 0.)
    ylo = -max(g.egg_length_mm/2, y_extent)
    yhi = max(g.egg_length_mm/2, y_extent)

    # The exact configured footprint, not the largest *available* mode.
    for scan_y in (float(scan_positions_mm(g,o)[0]), float(scan_positions_mm(g,o)[-1])):
        fp = source_footprint_mm(o, scan_y)
        if fp['kind'] == 'circle':
            source_bounds=(fp['cx']-fp['radius'],fp['cx']+fp['radius'],
                           fp['cy']-fp['radius'],fp['cy']+fp['radius'])
        else:
            source_bounds=(fp['xmin'],fp['xmax'],fp['ymin'],fp['ymax'])
        sx0,sx1,sy0,sy1=source_bounds
        xlo=min(xlo,sx0); xhi=max(xhi,sx1)
        ylo=min(ylo,sy0); yhi=max(yhi,sy1)

    padding=float(o.domain_padding_mm)
    xlo-=padding; xhi+=padding
    ylo-=padding; yhi+=padding
    # Preserve the original launch plane and camera/top-face clearances.
    zlo = -g.egg_height_mm/2 - o.source_gap_mm - 2*v
    zsize = g.egg_height_mm + o.source_gap_mm + o.camera_plane_gap_mm + 4*v
    dims=tuple(int(np.ceil(d/v)) for d in (xhi-xlo,yhi-ylo,zsize))
    nvoxels=int(np.prod(dims,dtype=np.int64))
    mib=1024.**2
    # Approximate in-memory Python masks and temporaries. MCX's GPU scratch
    # and photon records are device/build-specific and *not* certified here.
    label_mib=nvoxels/mib  # uint8 MCX label volume
    host_mib=16*nvoxels/mib  # conservative working bound (rough, NOT guaranteed)
    detector_mib=48*int(o.photons_per_line)/mib  # conservative per saved photon, varies with MCX
    plan=DomainPlan(dims=dims,origin_mm=(float(xlo),float(ylo),float(zlo)),
                    voxel_size_mm=v,nvoxels=nvoxels,label_memory_mib=label_mib,
                    host_build_estimate_mib=host_mib,
                    detected_buffer_estimate_mib=detector_mib,
                    domain_padding_mm=padding,active_source_mode=o.mode,
                    acquisition=o.camera_mode,
                    x_bounds_mm=(float(xlo),float(xlo+dims[0]*v)),
                    y_bounds_mm=(float(ylo),float(ylo+dims[1]*v)))
    if enforce_budget and host_mib > o.domain_build_budget_mib:
        raise ValueError(
            f'Estimated domain build working memory {host_mib:.0f} MiB exceeds '
            f'the configured {o.domain_build_budget_mib:.0f} MiB safety budget '
            f'({dims[0]} × {dims[1]} × {dims[2]} = {nvoxels:,} voxels). '
            'Increase the host-build budget if RAM is available, or decrease padding '
            'after checking convergence. Do not reduce voxel resolution without consideration.')
    return plan


def build_egg_volume(g: Geometry, o: Optics, materials: np.ndarray) -> VoxelDomain:
    """Rasterize anatomy into an MCX label volume; all transport belongs to MCX.

    Air is a nonzero material within the domain. A configurable padded domain
    reduces artificial side exits; run a padding-convergence study before
    treating transmitted intensity as physically independent of box size.
    """
    plan=domain_plan(g,o,materials)
    v=plan.voxel_size_mm
    dims=plan.dims
    origin=plan.origin_mm
    # output voxel centers; x/y meshgrid convention consistent with MCX [x,y,z]
    xs=origin[0]+(np.arange(dims[0])+0.5)*v
    ys=origin[1]+(np.arange(dims[1])+0.5)*v
    zs=origin[2]+(np.arange(dims[2])+0.5)*v
    xx=xs[:,None,None]
    yy=ys[None,:,None]
    zz=zs[None,None,:]
    outer=(xx/(g.egg_width_mm/2))**2+(yy/(g.egg_length_mm/2))**2+(zz/(g.egg_height_mm/2))**2<=1
    inner=(xx/(g.egg_width_mm/2-g.shell_thickness_mm))**2+(yy/(g.egg_length_mm/2-g.shell_thickness_mm))**2+(zz/(g.egg_height_mm/2-g.shell_thickness_mm))**2<=1
    vol=np.ones(tuple(dims),dtype=np.uint8)
    vol[outer]=2
    vol[inner]=3
    if g.include_yolk:
        yolk=((xx-g.yolk_offset_x_mm)/g.yolk_radius_x_mm)**2 + ((yy-g.yolk_offset_y_mm)/g.yolk_radius_y_mm)**2 + ((zz-g.yolk_offset_z_mm)/g.yolk_radius_z_mm)**2<=1
        vol[yolk & inner]=4
    if g.include_blood:
        blood=(xx-g.blood_x_mm)**2+(yy-g.blood_y_mm)**2+(zz-g.blood_z_mm)**2<=g.blood_radius_mm**2
        vol[blood & inner]=5
    # 0 is MCX exterior medium, 1 is volume air; duplicate air props.
    props=np.vstack([materials[0],materials]).astype(np.float32)
    return VoxelDomain(np.asfortranarray(vol),v,tuple(map(float,origin)),props,plan)


def source_footprint_mm(o: Optics, scan_y_mm: float=0.) -> dict:
    """Single source-of-truth for UI footprint and MCX source builder."""
    cy=scan_y_mm+o.source_offset_y_mm
    if o.mode=='Spot':
        return {'kind':'circle','cx':0.,'cy':float(cy),'radius':o.spot_diameter_mm/2}
    if o.mode=='Transverse line':
        sx,sy=o.line_length_mm,o.line_width_mm
    elif o.mode=='Longitudinal line':
        sx,sy=o.line_width_mm,o.line_length_mm
    elif o.mode=='Broad backlight':
        sx,sy=o.line_length_mm,o.line_length_mm
    else:
        raise ValueError('Unknown source mode '+str(o.mode))
    return {'kind':'rectangle','xmin':-sx/2,'xmax':sx/2,'ymin':cy-sy/2,'ymax':cy+sy/2}


def source_config(domain: VoxelDomain | DomainPlan, o: Optics, scan_y_mm: float) -> dict:
    """Native MCX disk or planar sources; never silently approximate spot by line."""
    footprint=source_footprint_mm(o,scan_y_mm)
    # physical launch plane: 1 voxel inside the bottom of MCX air volume
    plane_z=domain.origin_mm[2]+2.*domain.grid_mm
    direction=[0.,0.,1.,float(o.beam_focal_length_mm/domain.grid_mm)]
    base={'srcdir':direction, 'srcparam1':[0.,0.,0.,0.], 'srcparam2':[0.,0.,0.,0.]}
    if footprint['kind']=='circle':
        loc=domain.mm_to_voxel([footprint['cx'],footprint['cy'],plane_z])
        base.update(srctype='disk',srcpos=loc.tolist(),
                    srcparam1=[footprint['radius']/domain.grid_mm,0.,0.,0.])
    else:
        loc=domain.mm_to_voxel([footprint['xmin'],footprint['ymin'],plane_z])
        base.update(srctype='planar',srcpos=loc.tolist(),
                    srcparam1=[(footprint['xmax']-footprint['xmin'])/domain.grid_mm,0.,0.,0.],
                    srcparam2=[0.,(footprint['ymax']-footprint['ymin'])/domain.grid_mm,0.,0.])
    # Validate that every source footprint fits the mesh; critical for edge scans.
    if footprint['kind']=='circle':
        r=footprint['radius']/domain.grid_mm
        if not (r<base['srcpos'][0]<domain.dims[0]-r and r<base['srcpos'][1]<domain.dims[1]-r):
            raise ValueError('Spot source exceeds MCX domain; enlarge margins or reduce dimensions.')
    else:
        p=np.asarray(base['srcpos'])
        if (p[:2]<0).any() or p[0]+base['srcparam1'][0]>domain.dims[0] or p[1]+base['srcparam2'][1]>domain.dims[1]:
            raise ValueError('Planar source exceeds MCX domain.')
    return base


def mcx_config(domain:VoxelDomain,o:Optics,scan_y_mm:float,seed:int) -> dict:
    """Build a PMCX cfg with the TOP MCX box face as photon detector.

    12-char `bc`: six Fresnel faces; last six mark which exit faces detect.
    Only +z is flagged. This avoids confusing internal fluence with camera
    transmission. MCX does the full transport through the voxelized egg.
    """
    c=dict(nphoton=int(o.photons_per_line),vol=domain.vol,prop=domain.props,
           unitinmm=float(domain.grid_mm),issrcfrom0=1,
           isreflect=1,isrefint=1,
           bc='______000001',issaveexit=1, savedetflag='DPXV',
           maxdetphoton=int(o.photons_per_line),issave2pt=0,
           tstart=0.,tend=5e-9,tstep=5e-9,
           gpuid=int(o.gpu_id), seed=int(seed),autopilot=1,
           flog=os.devnull)
    c.update(source_config(domain,o,scan_y_mm))
    return c


def backend_status()->tuple[bool,str]:
    try:
        import pmcx
    except (ImportError,OSError) as exc:
        return False, f'PMCX not installed/loadable: {exc}. Run pip install pmcx.'
    try:
        info=pmcx.gpuinfo()
        if info is None or len(info)==0:
            return False,'PMCX installed, but no CUDA-compatible NVIDIA GPU detected.'
    except (Exception) as exc:
        return False,f'PMCX cannot initialize a CUDA GPU: {exc}'
    return True,'PMCX initialized successfully (CUDA GPU present).'


def top_face_photons(detp:dict, prop:np.ndarray, pmcx_module) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Decode MCX-returned exit positions/directions and physical weights.

    MCX `detp['p']` and `detp['v']` are *exit* (not source) values. `detweight`
    belongs to PMCX itself; this adapter deliberately does not reimplement
    absorption or photon weights.
    """
    if not isinstance(detp,dict):
        raise ValueError('Expected pmcx.mcxlab detector dictionary, not raw matrix.')
    if 'p' not in detp or 'v' not in detp:
        raise RuntimeError('MCX detector output is missing exit position/direction. Check savedetflag="DPXV" and bc.')
    p=np.asarray(detp['p'],dtype=float)
    v=np.asarray(detp['v'],dtype=float)
    if p.size==0:
        return np.empty((0,3)),np.empty((0,3)),np.empty(0)
    if p.ndim!=2 or v.ndim!=2:
        raise ValueError('Unexpected MCX detector-array shape')
    if p.shape[0]==3 and p.shape[1]!=3:
        p=p.T
    if v.shape[0]==3 and v.shape[1]!=3:
        v=v.T
    if p.shape[1]!=3 or v.shape!=p.shape:
        raise ValueError('PMCX exit positions/directions must have shape (N,3).')
    w=np.asarray(pmcx_module.detweight(detp,prop),dtype=float).reshape(-1)
    if len(w)!=len(p):
        raise ValueError('PMCX weight count does not match exit photons.')
    return p,v,w


def camera_bin(p:np.ndarray,v:np.ndarray,weights:np.ndarray,domain:VoxelDomain,
               o:Optics,scan_y_mm:float,x_edges_mm:np.ndarray)->tuple[np.ndarray,int]:
    """Thin geometry-only camera mask: angular gate + moving slit + x histogram.

    This is a *model* of telecentric detection, NOT physical photon transport.
    """
    if len(p)==0:
        return np.zeros(len(x_edges_mm)-1),0
    xyz=domain.voxel_to_mm(p)
    theta=np.rad2deg(np.arctan2(np.hypot(v[:,0],v[:,1]),v[:,2]))
    keep=(v[:,2]>0)&(theta<=o.camera_half_angle_deg)&(np.abs(xyz[:,1]-scan_y_mm)<o.camera_line_width_mm/2)
    hits,_=np.histogram(xyz[keep,0],bins=x_edges_mm,weights=weights[keep])
    return hits,int(np.count_nonzero(keep))


def camera_bin_area(p: np.ndarray, v: np.ndarray, weights: np.ndarray,
                    domain: VoxelDomain, o: Optics,
                    y_edges_mm: np.ndarray, x_edges_mm: np.ndarray) -> tuple[np.ndarray, int]:
    """Stationary telecentric-like area detector, one 2D exposure.

    Bin MCX *exit-face photon positions* to [y,x] with a direction gate.
    This is geometric readout only. It is not a physical lens, pupil, PSF,
    or a camera looking at the egg from a finite perspective distance.
    """
    shape = (len(y_edges_mm)-1, len(x_edges_mm)-1)
    if len(p) == 0:
        return np.zeros(shape, dtype=float), 0
    xyz = domain.voxel_to_mm(p)
    theta = np.rad2deg(np.arctan2(np.hypot(v[:, 0], v[:, 1]), v[:, 2]))
    keep = (v[:, 2] > 0) & (theta <= o.camera_half_angle_deg)
    keep &= (xyz[:, 0] >= x_edges_mm[0]) & (xyz[:, 0] <= x_edges_mm[-1])
    keep &= (xyz[:, 1] >= y_edges_mm[0]) & (xyz[:, 1] <= y_edges_mm[-1])
    image, _, _ = np.histogram2d(xyz[keep, 1], xyz[keep, 0],
                                bins=(y_edges_mm, x_edges_mm), weights=weights[keep])
    return image, int(np.count_nonzero(keep))


def simulate_mcx(g: Geometry, o: Optics, materials: np.ndarray, run_backend=None) -> dict:
    """Execute MCX photon physics for either line-scan or area imaging.

    - Pushbroom: one MCX run per y scan position, stationary detector/source
      pair in the lab (translated together relative to the simulated egg).
    - Area: one MCX run at the fixed source position, all x/y exit pixels
      sampled simultaneously, no egg translation and no observation slit.

    The optional injected backend is for unit tests, not an alternative
    physical simulator; real runs always use official pmcx.mcxlab.
    """
    validate(g, o, materials)
    if run_backend is None:
        try:
            import pmcx as run_backend
        except (ImportError, OSError) as exc:
            raise RuntimeError('PMCX is required; install `pip install pmcx` on an NVIDIA/CUDA system. No custom fallback.') from exc
        available, message = backend_status()
        if not available:
            raise RuntimeError(message)

    domain = build_egg_volume(g, o, materials)
    x_edges = np.linspace(-g.egg_width_mm * .56, g.egg_width_mm * .56,
                          o.camera_pixels + 1)
    x_mm = .5 * (x_edges[:-1] + x_edges[1:])
    area = o.camera_mode == 'Area'
    if area:
        y_edges = np.linspace(-g.egg_length_mm * .54, g.egg_length_mm * .54,
                              o.area_rows + 1)
        y_mm = .5 * (y_edges[:-1] + y_edges[1:])
        image = np.zeros((o.area_rows, o.camera_pixels), dtype=float)
        positions = [0.]       # single exposure of a stationary egg and source
    else:
        y_mm = scan_positions_mm(g, o)
        image = np.zeros((o.scan_lines, o.camera_pixels), dtype=float)
        positions = y_mm      # move the source/camera pair across the egg

    exit_total = 0
    capture_total = 0
    captured_weight = 0.
    for i, sy in enumerate(positions):
        cfg = mcx_config(domain, o, float(sy), max(1, (int(o.seed)+i) % 2147483647))
        try:
            result = run_backend.mcxlab(cfg)
        except Exception as exc:
            raise RuntimeError(f'MCX failed on {"area exposure" if area else f"scan line {i+1}/{o.scan_lines}"}: {exc}') from exc
        detp = result.get('detp') if isinstance(result, dict) else None
        if detp is None:
            raise RuntimeError('MCX returned no detected photon dictionary; check +z face detection.')
        p, v, w = top_face_photons(detp, domain.props, run_backend)
        if area:
            image, n_camera = camera_bin_area(p, v, w, domain, o, y_edges, x_edges)
            captured_weight += float(image.sum())
        else:
            line, n_camera = camera_bin(p, v, w, domain, o, float(sy), x_edges)
            image[i, :] = line
            captured_weight += float(line.sum())
        exit_total += len(p)
        capture_total += n_camera

    # Same intensity convention as the line scan: each row/exposure is
    # weighted photon energy per launched packet, NOT radiant power per area.
    image /= float(o.photons_per_line)
    launched = int(o.photons_per_line * len(positions))
    return {'image': image, 'x_mm': x_mm, 'y_mm': y_mm,
            'scan_y_mm': y_mm,  # backwards-compatible key for saved data
            'camera_mode': o.camera_mode,
            'geometry': asdict(g), 'optics': asdict(o), 'materials': materials.copy(),
            'diagnostics': {'engine': 'MCX via PMCX', 'camera_mode': o.camera_mode,
                            'mcx_runs': len(positions), 'photons_per_run': int(o.photons_per_line),
                            'launched_photons': launched,
                            'top_exit_photons': exit_total, 'camera_accepted_photons': capture_total,
                            'weighted_camera_signal': captured_weight,
                            'fraction_collected': captured_weight / launched,
                            'voxel_size_mm': domain.grid_mm, 'volume_shape': domain.dims,
                            'source_type': source_config(domain, o, 0)['srctype'],
                            'domain_plan': asdict(domain.plan)}}


def homogeneous_slab_config(mu_a_mm1:float=0.07,thickness_mm:int=12,nphotons:int=30_000)->dict:
    """Known-answer slab for MCX engine-level Beer-Lambert test (no scatter).

    A non-scattering, index-matched absorbing slab of integer thickness L
    illuminated by a collimated pencil must transmit exp(-mua*L).
    """
    shape=(28,28,42)
    vol=np.ones(shape,dtype=np.uint8)
    vol[:,:,12:12+thickness_mm]=2
    return dict(nphoton=nphotons,vol=np.asfortranarray(vol),
                prop=np.asarray([[0,0,1,1],[0,0,1,1],[mu_a_mm1,0,1,1]],dtype=np.float32),
                unitinmm=1.,issrcfrom0=1,srcpos=[14.,14.,1.],srcdir=[0.,0.,1.],
                srctype='pencil',bc='______000001',isreflect=1,isrefint=1,
                savedetflag='DPXV',issaveexit=1,issave2pt=0,
                maxdetphoton=nphotons,seed=123,autopilot=1,
                tstart=0.,tend=5e-9,tstep=5e-9,flog=os.devnull)


def run_slab_reference(mu_a_mm1:float=0.07,thickness_mm:int=12,nphotons:int=30_000,pmcx_module=None)->dict:
    """True MCX run, compared to analytical Beer-Lambert transmission."""
    if pmcx_module is None:
        import pmcx as pmcx_module
    cfg=homogeneous_slab_config(mu_a_mm1,thickness_mm,nphotons)
    res=pmcx_module.mcxlab(cfg)
    p,v,w=top_face_photons(res['detp'],cfg['prop'],pmcx_module)
    measured=float(np.sum(w)/nphotons)
    expected=float(np.exp(-mu_a_mm1*thickness_mm))
    return {'expected':expected,'measured':measured,'relative_error':abs(measured-expected)/expected,
            'detected_packets':len(p),'nphoton':nphotons}


def source_shape_reference_configs(nphotons:int=60_000)->dict[str,dict]:
    """Two known-shape MCX source phantoms in transparent non-scattering air.

    No egg present. A native 'disk' must produce circular transmitted footprint;
    native 'planar' rectangle must produce a broad transverse illuminated stripe.
    Both source launch rays and the top-face exit records are computed by MCX.
    """
    common=dict(nphoton=nphotons,vol=np.asfortranarray(np.ones((100,100,24),dtype=np.uint8)),
                prop=np.array([[0,0,1,1],[0,0,1,1]],dtype=np.float32),
                unitinmm=1.,issrcfrom0=1,srcdir=[0.,0.,1.],
                bc='______000001',issaveexit=1,savedetflag='DPXV',issave2pt=0,
                maxdetphoton=nphotons,seed=1361,autopilot=1,
                tstart=0.,tend=5e-9,tstep=5e-9,flog=os.devnull)
    return {
        'Spot': dict(common,srctype='disk',srcpos=[50.,50.,1.],
                     srcparam1=[6.,0.,0.,0.],srcparam2=[0.,0.,0.,0.]),
        'Transverse line':dict(common,srctype='planar',srcpos=[20.,49.,1.],
                              srcparam1=[60.,0.,0.,0.],srcparam2=[0.,2.,0.,0.])
    }


def run_source_shape_reference(pmcx_module=None)->dict[str,dict]:
    """MCX engine-level source verification, independent of egg geometry."""
    if pmcx_module is None:
        import pmcx as pmcx_module
    out={}
    for name,c in source_shape_reference_configs().items():
        r=pmcx_module.mcxlab(c)
        p,v,w=top_face_photons(r['detp'],c['prop'],pmcx_module)
        if len(p)<.95*c['nphoton']:
            raise RuntimeError(f'Unexpectedly few photons exited top face in {name}: {len(p)}')
        sx=float(np.std(p[:,0]));sy=float(np.std(p[:,1]))
        out[name]={'exit_photons':len(p),'std_x_mm':sx,'std_y_mm':sy,'aspect_x_over_y':sx/max(sy,1e-12)}
    return out


def run_area_camera_source_reference(pmcx_module=None) -> dict[str, dict]:
    """Optional *real MCX* known-answer transmission of source footprints.

    A transparent, index-matched, nonscattering block is illuminated from
    below by a native MCX disk and plane. The one-shot area-camera image must
    recover the disk and transverse stripe, respectively. This also tests the
    area binning on actual MCX exit photons (not synthetic response packets).
    """
    if pmcx_module is None:
        import pmcx as pmcx_module
    result = {}
    o = Optics(camera_mode='Area', camera_pixels=100, area_rows=100,
               camera_half_angle_deg=30.)
    dummy = VoxelDomain(vol=np.ones((100,100,24), dtype=np.uint8), grid_mm=1.,
                        origin_mm=(0.,0.,0.), props=np.array([[0,0,1,1],[0,0,1,1]],dtype=np.float32))
    edges = np.linspace(0.,100.,101)
    xc = .5*(edges[:-1]+edges[1:])
    yc = xc
    for name, cfg in source_shape_reference_configs().items():
        run = pmcx_module.mcxlab(cfg)
        p,v,w = top_face_photons(run['detp'],cfg['prop'],pmcx_module)
        im,n = camera_bin_area(p,v,w,dummy,o,edges,edges)
        yy,xx = np.meshgrid(yc,xc,indexing='ij')
        total = float(im.sum())
        if total <= 0:
            raise RuntimeError('Area-camera reference: no photons detected.')
        mean_x = float((im*xx).sum()/total)
        mean_y = float((im*yy).sum()/total)
        vx = float((im*(xx-mean_x)**2).sum()/total)
        vy = float((im*(yy-mean_y)**2).sum()/total)
        result[name] = {'accepted_photons':n,'weighted_flux':total/float(cfg['nphoton']),
                        'image_shape':list(im.shape), 'centroid_x_mm':mean_x,
                        'centroid_y_mm':mean_y,
                        'aspect_x_over_y':float(np.sqrt(vx/max(vy,1.e-12)))}
    return result
