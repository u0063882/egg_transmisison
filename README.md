# Egg transmission line-scan simulator

**Python | Monte Carlo photon transport | interactive Streamlit explorer**

Simulate the spatial pattern seen from above a whole chicken egg in transmission under a **fixed pushbroom camera and moving egg**, for various fibre-fed tungsten-halogen illumination geometries. Designed for an existing **150 W SCHOTT/SCHOLZ-type halogen lamp, gold reflector, no IR cutoff, and fibre line/spot optics**, but the model is generic.

> **Status: v0.1 research prototype — not a quantitatively validated imaging model.** The core is physically motivated 3D Monte Carlo code compiled by **Numba**, with tested geometry, Fresnel/Snell interfaces, exponential free paths, HG scattering and absorption. It is **not** a wrapper around the independently validated MCX or PyXOpto engines. Never use it to claim an anatomical structure is visible without comparison to a measured image. Photon count/angles and detector aperture are simplified.

## New: off-centre yolk falsification test (800 nm)

[Report: no-yolk vs central and displaced yolk, with real camera images **without overlays**](reports/yolk_displacement_800nm.md). The code now has `Geometry.include_yolk`, yolk x/y/z offset controls in Streamlit, and a homogeneous-phantom negative control. The prior cyan contours were **ground-truth geometry**, not photon-inferred boundaries. Reproduce with `python scripts/validate_yolk_shift.py --photons 16000 --lines 112 --pixels 128 --seeds 881 882`.

## Latest simulation comparison (580 / 800 / 1200 nm)

[Critical review and illustrated three-wavelength comparison](reports/illumination_critical_review_580_800_1200.md) — four principal illumination geometries, plus a broad-backlight throughput reference, reproducible code and a two-seed Monte Carlo check. **580 and 1200 nm are explicitly hypothetical absorption-sensitivity scenarios, not measured spectral predictions.** Reproduce with `python scripts/compare_three_wavelengths.py --photons 18000 --lines 112 --pixels 128`.

## Installation (Windows / macOS / Linux, Python 3.11–3.12)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

A local browser interface opens. The first scan includes one-time Numba compilation. Start at **800 nm**, choose **Transverse line**, then press **Run scan**. Increase packets if the signal is sparse. Use **Compare 4 sources** to compare illumination types with equal launched photon counts. Test beam divergence in x and y separately and source–camera offset; vary camera angular acceptance to emulate directional collection.

Alternative batch mode:

```bash
python -m eggtrans.cli --mode "Transverse line" --photons 20000 --lines 64 --pixels 96 --output results/line_800
python -m eggtrans.cli --mode Spot --photons 20000 --output results/spot_800
pytest -q
```

Outputs: `.npz` (raw relative per-launched-packet image + ground-truth projections + metadata), `.png`, `.svg`, `.json`. The **asinh** PNG display is visibility-enhanced; the NPZ contains the unmodified Monte Carlo signal.

## Coordinate system and experiment

- **x**: across conveyor, parallel to the instantaneous camera line.
- **y**: egg translation direction (along long axis of the egg in this demo).
- **z**: optical axis, halogen illumination from below, camera above.
- Camera and illuminator stay *fixed in laboratory coordinates*; mathematical equivalent is to keep the egg fixed and shift the source and camera scan line together across the egg.
- Sources: transverse line (along x), longitudinal line (along y), circular spot, broad rectangular backlight. Source–detector longitudinal offset can be changed.
- Source rays follow independent Gaussian x/y slopes around +z, with separately adjustable divergence standard deviations. The model is not an optical design calculation of the real SCHOTT cylindrical lens. **A narrow focused line is not necessarily collimated**.
- An idealised receiver measures rays leaving the *top* shell, projects them onto a plane just above the egg, accepts only those within a chosen angular half-cone and within a finite y-width line. This approximates a telecentric line receiver, **not a calibrated camera lens or spectrograph**.
- Perfect suppression of illumination bypassing the egg is assumed.

## Model / optical parameters

Geometry is nested concentric ellipsoids for shell/albumen and an offset ellipsoid for yolk, plus an optional spherical synthetic blood inclusion. Refractive-index jumps are calculated via Fresnel and Snell; volume scattering via the **Henyey–Greenstein phase function**, with absorption along paths. Single wavelength per run; the `wavelength` entry alone does **not** change the physics unless optical coefficients are supplied or edited.

The **only literature-anchored numbers** supplied in code are Tamura et al. (2026), **800 nm**:

| Tissue | μa (mm⁻¹) | μs (mm⁻¹) | g |
|---|---:|---:|---:|
| Albumen | 0.006 | 0.069 | 0.90 |
| Yolk | 0.014 | 5.350 | 0.92 |

All shell, artificial-blood inclusion and refractive-index parameters are **illustrative placeholders**, *not measurements*. No 400–1700 nm curves are bundled. **Without an optical-property CSV, the 800 nm coefficients are reused unchanged at other wavelengths, which is only a what-if experiment.** Actual spectral predictions require measured μa(λ), μs(λ), g(λ), n(λ) for each tissue, the source spectrum and detector response.

For measured data, upload a CSV to the UI or use `--spectra-csv your_measurements.csv`. Header:

```csv
wavelength_nm,material,mu_a_mm_1,mu_s_mm_1,g,n
```

Supply one row per wavelength for each of `Shell`, `Albumen`, `Yolk`, and `Blood spot`. Wavelengths must span the requested simulation band for **every** tissue. Interpolation inside the supplied range is linear; extrapolation is explicitly rejected. There is no fictional measured 1450 nm default.

## Interpretation and convergence

The app shows side-by-side (1) detected photon density and (2) ideal geometric material projections. The second view is **ground truth, not what the camera sees**. Roughness and shell porosity are not currently included; the egg boundaries are ideal smooth ellipsoids. A large dark central region in the Monte Carlo image **does not prove accurate yolk-boundary retrieval**.

Performance tips:

- Begin with ~8k packets × 40 lines and 64 camera pixels. Expect visible stochastic noise.
- For more rigorous comparisons use 50k–100k packets per line, repeat with independent seeds, and check convergence of profiles / contrast before interpreting shapes.
- Tighter angular detector acceptance and thinner line widths substantially reduce Monte Carlo collection efficiency. The small sample preview is **not** a high-fidelity instrument simulation.
- Equal emitted photons are **not equal radiant power** or irradiance at the lower shell; for physical comparisons, calibrate the effective source area, angular emission, spectral lamp power and camera throughput.
- `max_scattering_events` truncates rare long trajectories; monitor the warning/diagnostics. Refraction and shell scattering are simplified; finite focal length, polarization, egg shape asymmetry and internal interfaces are not represented in detail.

## Research sources and library foundation

1. **Tamura et al. (2026)**, *Light propagation in an intact egg: Combining bulk optical properties measurements with Monte Carlo simulations*, Journal of Food Engineering. DOI: [10.1016/j.jfoodeng.2026.113182](https://doi.org/10.1016/j.jfoodeng.2026.113182). Reported μa, μs, g for albumen/yolk at 800 nm; also demonstrates voxel-based MC simulation of the egg.
2. **Wang, Jacques & Zheng (1995)**, *MCML — Monte Carlo modeling of light transport in multi-layered tissues*, Computer Methods and Programs in Biomedicine 47(2), 131–146. DOI: [10.1016/0169-2607(95)01640-F](https://doi.org/10.1016/0169-2607(95)01640-F). Standard Monte Carlo framework for scattering tissue.
3. **Fang & Boas (2009)**, *Monte Carlo simulation of photon migration in 3D turbid media accelerated by graphics processing units*, Optics Express. DOI: [10.1364/OE.17.020178](https://doi.org/10.1364/OE.17.020178). Validated photon transport reference for **MCX**: https://github.com/fangq/mcx
4. **Naglič et al. (2022)**, *MCDataset: a public reference dataset of Monte Carlo simulated quantities for multilayered and voxelated tissues…*, Journal of Biomedical Optics. DOI: [10.1117/1.JBO.27.8.083012](https://doi.org/10.1117/1.JBO.27.8.083012). **PyXOpto**: https://github.com/xopto/pyxopto
5. **Ahmed et al. (2026)**, *Hyperspectral imaging dataset for non-destructive fertility and structural evaluation of chicken eggs*, Scientific Data. https://www.nature.com/articles/s41597-026-06556-1 — relevant pushbroom/transillumination experiment.

**For quantitative scientific publication:** Treat this tool as a geometry/illumination hypothesis generator. Benchmark transport and boundary behavior against MCX/PyXOpto, measure the actual egg optical properties over 400–1700 nm, photograph the SCHOTT beam profiles at several heights, model camera pupil/NA/spectral sensitivity, and validate the apparent contour against opened-egg or MRI ground truth.

## Files

- `eggtrans/core.py`: physical transport and geometry kernel (Numba)
- `eggtrans/spectra.py`: optical-property input and interpolation
- `app.py`: interactive Streamlit viewer
- `eggtrans/cli.py`: batch runs and PNG/SVG/NPZ output
- `tests/`: smoke, geometry, reproducibility, spectral import tests

## Optional export for validated MCX photon transport

An **independent cross-check** can be prepared as MCX-compatible voxel labels and JSON inputs with:

```bash
python -m eggtrans.mcx_export --output results/mcx_case --voxel-mm 0.5 --scan-lines 24 --mode "Transverse line" --photons 10000000
cd results/mcx_case
mcx -f scan_000.json -G 1
```

This export uses the verified open-source [MCX format](https://github.com/fangq/mcx) (`Domain.Media`, x/y/z uint8 labels in column-major order, `Optode` source, output settings) and creates one JSON input for each scan position. **The generated MCX inputs and voxel maps have been structurally tested, but have not been executed against MCX in this environment**, because the MCX GPU executable was not installed. A full MCX camera image still requires extracting the detected photons' saved exit positions/directions (`SaveDataMask: DPXVW`) and applying a real lens/line-pupil model. MCX `flux` is *not* camera irradiance. The exporter currently uses collimated planar/disk sources and does not reproduce the independent x/y angular divergence sliders.

For accurate shell thickness, reduce voxel size to ≤0.25 mm (at the cost of memory). The fast Numba simulator remains the functioning interactive preview, not a substitute for independent MCX validation.
