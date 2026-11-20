# Egg transmission simulator — v0.4 (MCX)

**Python · Streamlit · validated MCX photon-transport engine through `pmcx` · pushbroom line-scan camera**

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

**PMCX needs a compatible NVIDIA CUDA GPU** for its standard binary backend; NVIDIA drivers must be installed. The pip wheel typically includes the CUDA runtime, so a full CUDA toolkit is normally unnecessary. Without an operational MCX backend, the application can show its geometry schematic but **will not silently switch to a home-built ray tracer**. For Intel/AMD/OpenCL machines, MCX-CL/`pmcxcl` may be suitable as an additional future backend, but it is not wired or verified in this version.

## Source geometries: important spot/line distinction

The camera observes a fixed horizontal line x across the egg. The egg translates along y. A fixed light source beneath the camera appears to move along the egg in the egg-relative reference frame.

| Selection | Native MCX `srctype` | Instantaneous footprint on underside of egg |
|---|---|---|
| **Spot** | `disk` | Circular disk of user-specified diameter |
| **Transverse line** | `planar` | Long in x, narrow in y (perpendicular to translation) |
| **Longitudinal line** | `planar` | Narrow in x, long in y (parallel to translation) |
| **Broad backlight** | `planar` | Square area in xy |

![Instantaneous MCX source footprints](docs/illumination_footprints.png)

Both the *launch parameters* and the *Streamlit top-view schematic* use `source_footprint_mm()` as a single source of truth. Tests assert the type, size, orientation and y displacement of each illumination mode.

**Why a spot can look like a line in the reconstructed image:** at every scan position the spot illuminates approximately the same x region of the **moving** egg. Stacking measurements of its successive y positions can therefore create a **longitudinal bright stripe**. This is expected with a pushbroom scan and does not in itself indicate the wrong light source. The schematic independently shows the **instantaneous** circular spot.

The MCX source focus uses the documented native planar/disk focusing parameter (`srcdir[3]`): zero = collimated, negative = diverging, positive = converging. Unlike the original custom simulator, the interface no longer advertises independently configurable x/y divergence; MCX's native focal setting does not guarantee those controls. A narrow source is not necessarily an angularly collimated one.

## Transport and camera model

1. `eggtrans/mcx_engine.py` voxelizes nested ellipsoids into **nonzero** MCX volume labels: air=1, shell=2, albumen=3, yolk=4, blood=5. The exterior/background row 0 and air row 1 both have air properties. Shell thickness is only approximately represented by voxelization.
2. Every pushbroom scan line produces an **MCX native disk/planar** source at that scan position; `pmcx.mcxlab(cfg)` performs **all** scattering, absorption, Fresnel/refraction and photon transport.
3. MCX records photons leaving the **+z simulation box face** (via MCX `bc='______000001'`, save exit positions and directions). This is a transmission-detection surface, not internal fluence.
4. The project applies only a **geometrical pixel readout** to those recorded exit photons: accept ray directions within the camera half-angle, require y within the camera's observation slit, bin transmitted energy by x. Photon weights are computed by `pmcx.detweight()` rather than a local Beer–Lambert implementation.
5. The result is one spatial 2D image per wavelength. A complete HSI cube requires repeating with measured optical coefficients at every wavelength.

**Camera approximation:** This is an idealized telecentric/angular-gated line receiver at the upper MCX box face. It does not trace the real camera lens or model numerical aperture, aperture stop, PSF, slit throughput, sensor sensitivity or shot/read noise. One 1 mm voxel near a 0.35 mm shell is *not* sufficient to resolve shell geometry; prefer a finer grid or later migrate to MCX split-voxel media.

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
- `eggtrans/verify_physics.py`: execute the reference phantoms using real MCX
- `eggtrans/spectra.py`: measured spectral CSV support
- `eggtrans/cli.py`: batch run using same MCX entry point
- `eggtrans/mcx_export.py`: export inspectable **PMCX** inputs sharing the same source definitions; not a second transport engine
- `tests/`: source geometry, mapping, detector readout and optional GPU MCX reference tests

Previous illustrated reports under `reports/` are retained as **historical v0.1–v0.3 results generated with the *retired custom* kernel**, not as new MCX predictions. The old batch scripts in `scripts/` are historical as well; use the new Streamlit app or `python -m eggtrans.cli` for future MCX simulations.

## Primary open-source sources

- [MCX: Fang & Boas (2009), GPU-accelerated Monte Carlo photon transport](https://doi.org/10.1364/OE.17.020178) and [MCX repository / current documentation](https://github.com/fangq/mcx)
- [PMCX: official Python bindings](https://pypi.org/project/pmcx/)
- [MCX source types and upper-face detectors](https://github.com/fangq/mcx/blob/master/mcxlab/README.txt)
- [Tamura et al. (2026), egg optical properties](https://doi.org/10.1016/j.jfoodeng.2026.113182)
- [PyXOpto](https://github.com/xopto/pyxopto) is another established alternative but is not used here; mixing two transport engines would make validation and reproducibility more complex.
