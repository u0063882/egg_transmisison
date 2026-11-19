# Does the Monte Carlo simulator really see the yolk?

**A falsification test using an absent, centred and displaced yolk.**  
800 nm · 112 scan lines × 128 camera pixels · 16,000 photons per scan line · 2 independent seeds (881, 882) per case.  
**Status:** internal software consistency check, **not** experimental validation.

## Why this test was needed

Earlier figures superimposed a **cyan dashed ground-truth yolk contour** on every camera simulation. This was drawn from the *input geometry*, not inferred from the transported photons. It makes otherwise ambiguous patterns look more convincing than they are. We no longer draw ground truth on simulated camera images.

The default yolk was also close to the centre of an ellipsoidal egg, where the illumination is naturally strongest. That cannot establish localization. The test below deliberately removes the yolk and displaces it to either side.

## What was simulated

- **Source/detector:** stationary transverse line and stationary 6 mm spot, each paired with the same idealized pushbroom line receiver; the egg moves. A static-egg simulation with the source/receiver scanned together is mathematically equivalent in this model.
- **Yolk cases:** absent; centred at `(0, 0, −1) mm`; displaced to `(−5, −7, −1) mm`; displaced to `(+5, −7, −1) mm`. Each existing yolk has unchanged 14 × 16 × 14 mm ellipsoidal radii. The blood inclusion was disabled.
- **Materials:** default 800 nm coefficients. Literature anchors: albumen μa=0.006 mm⁻¹, μs=0.069 mm⁻¹, g=0.90; yolk μa=0.014 mm⁻¹, μs=5.350 mm⁻¹, g=0.92. **Shell optics, refractive indices, and detector are still hypothetical**.
- **Fair display:** each illumination source has **one shared linear color scale for all four yolk cases**. The two source types have different scales, so source-to-source brightness is not directly comparable. No independently normalized panels or contour overlays.
- **Monte Carlo:** two seeds averaged; each source/geometry combination launches 3,584,000 photon packets total, or 28,672,000 packets across the eight combinations. Input power and actual spectral radiance are not calibrated.

## What the simulated camera alone sees

![Unannotated camera output; shared scale within each source](yolk_displacement_800nm/camera_raw_no_overlays.jpg)

**Important:** In the **no-yolk** case there is **still a central light pattern**. This comes from the combined illumination, egg shape, shell and receiver geometry. It must not be mistaken for a yolk.

Putting a strongly scattering yolk in that central region greatly reduces detected intensity. Moving it to the left or right moves the darkest region, although the response is broadened and geometry-dependent.

## Isolate only the effect of adding the yolk

We can subtract a *counterfactual simulation with no yolk*: `signal(no yolk) − signal(with yolk)`. Positive red regions represent loss of detected signal due to the yolk; blue regions indicate local gain/redistribution. These are **not** maps that can be directly acquired from a real egg, because the identical real egg cannot simultaneously exist with and without its yolk.

![Yolk-caused change relative to the no-yolk control](yolk_displacement_800nm/yolk_effect_no_overlays.jpg)

The true programmed geometry appears **only in the separate figure below**, not as an overlay on the camera output.

![Input ground-truth geometry: not camera-detected](yolk_displacement_800nm/geometry_truth_only.jpg)

## Quantitative check: does the signal follow the yolk position?

Apparent centres are **weighted centroids of positive, 1.25-pixel-smoothed differential signal**, not detected anatomical boundaries. A true displacement of ±5 mm in x and −7 mm in y is deliberately substantial while keeping the yolk enclosed by the shell.

| Actual yolk centre, x / y (mm) | Line: apparent signal-loss centroid (mm) | Spot: apparent signal-loss centroid (mm) |
|---|---|---|
| 0 / 0 | −0.00 / +0.04 | +0.04 / +0.02 |
| −5 / −7 | **−2.31 / −4.10** | **−0.77 / −4.72** |
| +5 / −7 | **+2.24 / −4.08** | **+0.86 / −4.68** |

The line retains more lateral positional information: for a 5 mm x-displacement, the detected-response centroid moves only about 2.3 mm. The spot is much worse in x, moving only about 0.8 mm. These underestimates are **properties of this particular assumed system**, not experimental findings.

Other useful model diagnostics:

| Model case | Line: fraction of no-yolk flux lost | Spot: fraction of no-yolk flux lost |
|---|---:|---:|
| Centred | 66.2% | 74.2% |
| Shifted left | 53.0% | 63.6% |
| Shifted right | 53.1% | 63.7% |

Comparing *reflected differential images* for the left/right cases gives correlation ≈0.995 (line) and ≈0.998 (spot). This is a symmetry **software check**, not an independent test of optical accuracy.

## Negative controls and code review

1. **Explicit no-yolk transport** was added to the core (`Geometry.include_yolk=False`), removing the internal yolk interface entirely. Before this, the model could draw a yolk boundary even when that boundary was not inferred from photons.
2. A **homogeneous, non-scattering, index-matched phantom** produces **identical simulated images with and without the yolk geometry**. The refraction routine now avoids consuming a random number at an index-matched interface, making this test exact for a fixed seed.
3. Moving the yolk left and right produces appropriately mirrored optical effects under symmetric illumination and fixed shell. This is expected for a physically responsive model but cannot confirm the accuracy of scattering and lens assumptions.
4. Both raw and differential figures are kept **separate from input ground-truth masks**, and use common within-source scaling. Raw `.npz` images contain no overlaid annotation.

## What this does—and does not—prove

**Supported:** the algorithm responds causally to a changed high-scattering interior, moves the response in the correct general direction, is symmetric to lateral reversal, and does not synthesize a false yolk in a completely homogeneous phantom.

**Not supported:** that this response has the same magnitude as in a real egg; that its dark region is a reliable *yolk segmentation*; that a moving point illumination source provides true spatial resolution; or that the current code has been independently verified against an established transport solver.

The largest unresolved physical uncertainties are the eggshell's scattering/refraction (including membrane, pores and surface roughness), the real source angular radiance, the instrument collection pupil/PSF, and spectral optical coefficients outside the measured 800 nm albumen/yolk baseline. The code is an original **Numba Monte Carlo research prototype**; using well-established Monte Carlo formulas does not by itself validate the implementation.

**Next scientific validation:** benchmark photon flux and exit-angle/position statistics against **MCX or PyXOpto**, then compare real egg scans using a characterized SCHOTT source. Any claimed yolk boundary should be checked against an independent anatomical reference (e.g., opened egg or MRI).

## Reproduce

```bash
python scripts/validate_yolk_shift.py --photons 16000 --lines 112 --pixels 128 --seeds 881 882
pytest -q
```

Results in [`reports/yolk_displacement_800nm/`](yolk_displacement_800nm/): the committed figures and [`summary.json`](yolk_displacement_800nm/summary.json). Running the script additionally generates eight `.npz` files (each with the two individual seed images, metadata and underlying configuration). Those larger arrays are included in the downloadable project ZIP rather than stored in GitHub.

Run the interactive app with `streamlit run app.py`; the **Include yolk** checkbox and yolk x/y/z offset controls are available in **Egg geometry**. Ground-truth display is **off by default**.

### References

- Tamura et al. (2026), *Light propagation in an intact egg: Combining bulk optical properties measurements with Monte Carlo simulations*, *Journal of Food Engineering*. [DOI 10.1016/j.jfoodeng.2026.113182](https://doi.org/10.1016/j.jfoodeng.2026.113182).
- Fang & Boas (2009), *Monte Carlo simulation of photon migration in 3D turbid media accelerated by graphics processing units*, *Optics Express*. [DOI 10.1364/OE.17.020178](https://doi.org/10.1364/OE.17.020178).
- Damagatla et al. (2026), *Time-resolved diffuse optical techniques reveal the integrating-sphere behavior of intact chicken eggs*. [Journal link](https://www.sciencedirect.com/science/article/pii/S2950636026001568).
