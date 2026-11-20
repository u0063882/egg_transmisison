# Egg transmission simulator — v0.7 (MCX)

**Python · Streamlit · validated MCX photon-transport engine through `pmcx` · pushbroom line-scan **or stationary 2D area camera**

This project investigates whether a transmitted-light hyperspectral camera can recover spatial structures (e.g. yolk boundary, synthetic blood inclusion) in a moving egg. The project now uses **MCX/PMCX exclusively for the Monte Carlo photon transport**. The previous self-written Monte Carlo kernel was removed from the active simulation path.

> **Scientific status:** The **transport solver is established** (MCX), but this **egg/instrument model is *not* experimentally validated**. The 800 nm albumen and yolk optical properties are literature anchors; the shell, blood and refractive indices are illustrative defaults. At wavelengths other than 800 nm, the default properties **do not change** unless a CSV with measurements is supplied. A real camera lens, spectrograph, shell microtexture and measured 150 W SCHOTT emitter beam profiles are *not* modeled.

## Install with Conda (Windows / NVIDIA GPU)

```powershell
conda create -n eggtrans python=3.11 -y
conda activate eggtrans
cd path\to\egg_transmisison
python -m pip install -r requirements.txt
python -m pip install pytest
python -c "import pmcx; print(pmcx.gpuinfo())"
python -m streamlit run app.py
```

**PMCX package version:** requires `pmcx>=0.8.1,<0.9` (not the nonexistent `pmcx>=2025.10`). After pulling repository updates, run `python -m pip install -r requirements.txt` again. Check the runtime with `python -m eggtrans.verify_physics` on your GPU.

**PMCX needs a compatible NVIDIA CUDA GPU** for its standard binary backend; NVIDIA drivers must be installed. The pip wheel typically includes the CUDA runtime, so a full CUDA toolkit is normally unnecessary. Without an operational MCX backend, the application can show its geometry schematic but **will not silently switch to a home-built ray tracer**. For Intel/AMD/OpenCL machines, MCX-CL/`pmcxcl` may be suitable as an additional future backend, but it is not wired or verified in this version.

## Camera modes

The Streamlit **main page**, immediately below its title, now offers **Camera type: Pushbroom (line scan)** or
**Normal camera (2D area)**. It shows **App v0.5.1** at the top to identify a fresh checkout. The sidebar retains the illumination controls. Both rely exclusively on the established MCX
photon transport engine through PMCX; only the detector readout and source
movement differ.

| | Pushbroom | Normal 2D camera |
|---|---|---|
| Motion | Egg translates under fixed light and camera line | Egg and source stay stationary |
| MCX runs | One per successive y scan line | **One** per exposure |
| Detector | x histogram within y slit | 2D y/x histogram over entire field |
| MCX source | Native disk or planar source at each scan position | Same native disk or planar source, fixed |
| Typical spot result | May form a stripe after stacking scan lines | Localized 2D transmission footprint, blurred by scattering |
| Camera geometry | Idealized directional line receiver | Idealized telecentric-like 2D receiver |

The blue overlay in the interactive apparatus schematic changes from a
**horizontal camera line** to a **rectangular field of view**. The new mode
is an area image at **one wavelength**, not a snapshot hyperspectral sensor.

For a quick spot-vs-line comparison choose **Normal camera (2D area)**,
then **Compare spot vs transverse line**. Keep the photon budget equal; the
sources have very different illuminated areas, so equal photon counts should
not be confused with equal irradiance at the egg.

Command-line examples:

```powershell
python -m eggtrans.cli --camera Area --mode Spot --photons 1000000 --area-rows 96 --pixels 96 --output results/area_spot
python -m eggtrans.cli --camera Area --mode "Transverse line" --photons 1000000 --area-rows 96 --pixels 96 --output results/area_line
python -m eggtrans.cli --camera Pushbroom --mode Spot --photons 100000 --lines 48 --output results/scan_spot
```

## Camera result overlays and interactive 3D projection

After running a simulation, use the three **Display overlays** switches to add:

- **Full egg outline:** orthographic silhouette of the input ellipsoid.
- **Yolk outline:** input yolk ellipsoid silhouette, including x/y offset (not shown if yolk disabled).
- **Illumination outline:** MCX **source launch footprint** at the input plane, before photons travel through the egg. This is *not* the transmitted illumination footprint or a recovered boundary.

All outlines default **off** and are added after photon transport. The camera heatmap is always the actual returned MCX detector histogram. Geometry overlays use the **saved inputs from the image calculation**: changing sliders without rerunning cannot move the overlays relative to the old result. For pushbroom acquisition, the **Show instantaneous source at scan line** slider chooses which source position to visualize; the finished pushbroom image includes all line positions.

**Interactive 3D view** displays a translucent ellipsoidal egg, optional yolk, one instantaneous 3D source footprint at the source plane, and the simulated camera image mapped to a flat upper detector plane. The camera output is a 2D image, *not* a reconstruction of the interior. In pushbroom mode, the image is an **assembled scan from multiple times**; the displayed source is **one selected instant**. The view is therefore an explanatory composite, not a frozen simultaneous physical scene. The 3D viewer has independent visibility switches for the egg envelope, yolk, source, and camera output plane, initially enabled so the full scene is visible. Turn off **3D: camera output** to see the interior more clearly. The 3D egg envelope does not resolve the actual submillimetre shell thickness.

The renderer uses Plotly for all graphics and uses no custom transport, ray-tracing, or inference algorithms. The output color plane is made from actual MCX image values; it is never synthesized from the egg/yolk geometry.

## Camera readout consistency check

The ordinary pushbroom and area experiments are *not* directly comparable when illumination is spatially localized. In pushbroom mode, a fixed source and camera slit move relative to the egg. A spot makes a line-like strip across the completed scan; a stationary area camera captures a local spot. Broad backlighting may produce more similar images when the illuminated area covers the full translation range and the camera sampling/throughput are matched.

The Streamlit expander **Diagnostic: compare detector readouts with identical photons** runs **one** MCX simulation with a fixed scene and then bins the *identical exiting photons* using both an area detector and a virtual line detector with one area-row-width slit. The resulting 2D images should agree. This tests readout binning but does **not** test moving geometry or validate the MCX physics engine.

## Voxel size and the 0.35 mm eggshell

Standard MCX label volumes use a **uniform isotropic grid**: the same voxel size applies to the egg interior, thin shell, and air. The Streamlit selector offers **0.35 mm** in addition to 0.5–2 mm, but 0.35 mm represents only one nominal voxel across the shell and remains geometrically crude. For the default source and egg geometry this is approximately **10.2 million voxels** at 0.35 mm; the current simulator rejects label volumes larger than **12 million voxels**. Refining to 0.1 mm without changing grid strategy would expand the volume by about 43 times relative to 0.35 mm and exceed this safety bound.

Established sub-voxel/mesh alternatives: [MCX split-voxel SVMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687934/) for curved interfaces and [MMC](https://github.com/fangq/mmc) with tetrahedral meshes. Neither is connected to this project yet; do not claim locally adaptive voxels in standard MCX. A shell thinner than a coarse voxel may contain both surfaces in a single voxel, which is problematic even for simple split-voxel representations.

## Source geometries: important spot/line distinction

In **pushbroom mode**, the camera observes a fixed horizontal line x across the egg. The egg translates along y. A fixed light source beneath the camera appears to move along the egg in the egg-relative reference frame.

| Selection | Native MCX `srctype` | Instantaneous footprint on underside of egg |
|---|---|---|
| **Spot** | `disk` | Circular disk of user-specified diameter |
| **Transverse line** | `planar` | Long in x, narrow in y (perpendicular to translation) |
| **Longitudinal line** | `planar` | Narrow in x, long in y (parallel to translation) |
| **Broad backlight** | `planar` | Square area in xy |

![Instantaneous MCX source footprints](docs/illumination_footprints.png)

Both the *launch parameters* and the *Streamlit top-view schematic* use `source_footprint_mm()` as a single source of truth. Tests assert the type, size, orientation and y displacement of each illumination mode.

**Why a spot can look like a line in a pushbroom reconstructed image:** at every scan position the spot illuminates approximately the same x region of the **moving** egg. Stacking measurements of its successive y positions can therefore create a **longitudinal bright stripe**. This is expected with a pushbroom scan and does not in itself indicate the wrong light source. The schematic independently shows the **instantaneous** circular spot.

The MCX source focus uses the documented native planar/disk focusing parameter (`srcdir[3]`): zero = collimated, negative = diverging, positive = converging. Unlike the original custom simulator, the interface no longer advertises independently configurable x/y divergence; MCX's native focal setting does not guarantee those controls. A narrow source is not necessarily an angularly collimated one.

## Transport and camera model

1. `eggtrans/mcx_engine.py` voxelizes nested ellipsoids into **nonzero** MCX volume labels: air=1, shell=2, albumen=3, yolk=4, blood=5. The exterior/background row 0 and air row 1 both have air properties. Shell thickness is only approximately represented by voxelization.
2. In pushbroom mode, each scan line produces an **MCX native disk/planar** source at that scan position; in area mode a **single static MCX run** is used; `pmcx.mcxlab(cfg)` performs **all** scattering, absorption, Fresnel/refraction and photon transport.
3. MCX records photons leaving the **+z simulation box face** (via MCX `bc='______000001'`, save exit positions and directions). This is a transmission-detection surface, not internal fluence.
4. The project applies only a **geometrical pixel readout** to those recorded exit photons: accept ray directions within the camera half-angle, then either restrict to the moving y slit and bin x (pushbroom), or bin all accepted x/y photons into a 2D image (area camera). Photon weights are computed by `pmcx.detweight()` rather than a local Beer–Lambert implementation.
5. The result is one spatial 2D image per wavelength. A complete HSI cube requires repeating with measured optical coefficients at every wavelength.

**Camera approximation:** Both modes use idealized telecentric/angular-gated detection at the upper MCX box face; the area camera does not apply an observation slit. It does not trace the real camera lens or model numerical aperture, aperture stop, PSF, slit throughput, sensor sensitivity or shot/read noise. One 1 mm voxel near a 0.35 mm shell is *not* sufficient to resolve shell geometry; prefer a finer grid or later migrate to MCX split-voxel media.

**Source normalization:** images are reported per launched photon packet; a 150 W lamp does **not** automatically translate to absolute irradiance or wavelength-dependent brightness. Separate the spot/line irradiance, angular emission and spectral power when comparing hardware.

## Reference tests — two independent physical expectations

The project includes two tests that **actually call PMCX**, rather than verifying a custom implementation against itself:

**1. Beer–Lambert slab in transmission:** non-scattering, index-matched slab of thickness **12 mm**. For the independent values μa = 0, 0.03 and 0.10 mm⁻¹, expected total transmission is `exp(-μa × 12 mm)`. MCX must produce those exit-weight fractions within 4.5% relative tolerance.

**2. Transparent source footprints:** no egg, no absorption, no scattering, perfectly collimated. A **native MCX disk** must project a nearly circular spot (x/y standard-deviation ratio 0.85–1.15). A **native MCX planar stripe** must produce a long transverse line (x/y ratio >10). These tests detect a spot accidentally being configured as a line.

Run the actual MCX physics checks on a CUDA-equipped machine:

```powershell
conda activate eggtrans
python -m eggtrans.verify_physics
# Or using pytest:
$env:RUN_MCX_PHYSICS = '1'
python -m pytest -q tests/test_mcx_physics.py
```

For the rest of the software tests (no GPU needed):

```powershell
python -m pytest -q
```

**Test status, November 2026:** The backend-independent configuration, geometry, source and readout tests pass locally. GPU transport and the analytical slab tests were **not executed in the authoring environment**, because neither PMCX nor a supported GPU was available. The MCX test file is intentionally skipped without `RUN_MCX_PHYSICS=1`. The above commands are provided so you can run the definitive engine checks on your workstation. No fictitious transport validation or simulated camera images are reported.

## Known optical properties

At 800 nm only, albumen and yolk values are derived from Tamura et al. (2026) as used in the previous version:

| Region | μa (mm⁻¹) | μs (mm⁻¹) | g |
|---|---:|---:|---:|
| Albumen | 0.006 | 0.069 | 0.90 |
| Yolk | 0.014 | 5.350 | 0.92 |

The shell and synthetic blood parameters and *all* default refractive indices are illustrative assumptions. A wavelength selector without a property CSV does not change these coefficients and should not be interpreted as a real 1200/1500 nm prediction.

CSV schema:

```csv
wavelength_nm,material,mu_a_mm_1,mu_s_mm_1,g,n
```

Provide **Shell**, **Albumen**, **Yolk**, **Blood spot** rows for each wavelength; interpolation only, never extrapolation. Air is fixed analytically.

## Files

- `app.py`: Streamlit UI and plots
- `eggtrans/core.py`: egg geometry, shared dataclasses, and public entry point only (**no custom photon transport**)
- `eggtrans/mcx_engine.py`: PMCX configuration, voxel-label setup, photo exit-face readout, known-answer phantoms
- `eggtrans/schematic.py`: interactive top/side views tied to MCX source parameters
- `eggtrans/readout_validation.py`: MCX photon-exit detector equivalence check, without custom optical transport
- `eggtrans/visualization.py`: 2D image overlays and conceptual interactive 3D plot, without any photon transport
- `eggtrans/verify_physics.py`: execute the reference phantoms using real MCX
- `eggtrans/spectra.py`: measured spectral CSV support
- `eggtrans/cli.py`: batch run using same MCX entry point
- `eggtrans/mcx_export.py`: export inspectable **PMCX** inputs sharing the same source definitions; not a second transport engine
- `tests/`: source geometry, mapping, pushbroom/area camera readout and optional GPU MCX reference tests

Previous illustrated reports under `reports/` are retained as **historical v0.1–v0.3 results generated with the *retired custom* kernel**, not as new MCX predictions. The old batch scripts in `scripts/` are historical as well; use the new Streamlit app or `python -m eggtrans.cli` for future MCX simulations.

## Primary open-source sources

- [MCX: Fang & Boas (2009), GPU-accelerated Monte Carlo photon transport](https://doi.org/10.1364/OE.17.020178) and [MCX repository / current documentation](https://github.com/fangq/mcx)
- [PMCX: official Python bindings](https://pypi.org/project/pmcx/)
- [MCX source types and upper-face detectors](https://github.com/fangq/mcx/blob/master/mcxlab/README.txt)
- [Tamura et al. (2026), egg optical properties](https://doi.org/10.1016/j.jfoodeng.2026.113182)
- [PyXOpto](https://github.com/xopto/pyxopto) is another established alternative but is not used here; mixing two transport engines would make validation and reproducibility more complex.
