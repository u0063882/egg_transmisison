# MCX domain sizing and camera-mode comparison — v0.8

## Issue and correction

The v0.7 rectangular simulation volume was padded according to the **largest available** light-source dimension, even when that source dimension was disabled for the active mode. It also imposed a hard-coded 12,000,000-voxel limit irrespective of the actual memory budget. A fixed-area spot or a moving transverse line could therefore fail for non-optical reasons.

In v0.8, `domain_plan()` computes the union of:

- The actual ellipsoid egg envelope.
- The current camera field of view (including the pushbroom slit extent).
- The **selected** MCX source launch footprint; for pushbroom acquisition, its first and last scanning positions.
- A user-configurable lateral air margin around the union (default 8 mm).

The original z-direction source/egg/camera clearances remain. The *uniform voxel size is never changed automatically*. The new host-rasterization budget is estimated conservatively at 16 bytes per voxel, covering the label volume and temporary rasterization masks; it is **not** a measurement or guarantee of GPU memory. MCX exit photon records and runtime buffers are additional.

For the default 44 × 59 × 44 mm egg, 0.35 mm voxels, a 60 mm line setting and 8 mm lateral padding:

| Acquisition | Source | v0.7 voxel count | v0.8 voxel count | Change |
|---|---|---:|---:|---:|
| Pushbroom | Transverse line | ~12.84 M | 7.53 M | −41% |
| Pushbroom | Spot, 60 mm inactive line setting | ~12.84 M | 6.60 M | −49% |
| Area | Spot | ~7.64 M | 6.14 M | −20% |

These are **analytical domain-size calculations**, not photon-simulation measurements. They do not establish physical equivalence or sufficient boundary padding.

## GPU-required physical convergence check

```powershell
conda activate eggtrans
python scripts/run_domain_convergence.py --dry-run --camera Pushbroom --mode "Transverse line" --voxel .35 --line-length 60 --padding 6 8 12 16
python scripts/run_domain_convergence.py --camera Area --mode "Transverse line" --voxel .5 --padding 8 12 16 --photons 300000 --output reports/padding_convergence.json
```

The executable MCX comparison runs the **same voxel size, geometry, source and detector** with different lateral padding. It compares total weighted camera signal and a pixelwise normalized L1 difference to the largest box. Photon noise must be assessed independently by raising the packet budget and repeating seeds. Images that look similar are not necessarily converged in flux, and vice versa.

## Camera modes

The usual laboratory geometries remain different: a pushbroom spot is scanned relative to a moving egg and forms a longitudinal trace; a fixed 2D camera observes one spot exposure. It would be misleading to expect the two images to match in that case.

Two controlled diagnostics are now available:

1. **Same-photon detector check (already available in v0.6):** One MCX run is binned in two equivalent ways (area histogram versus virtual slit rows). This checks software readout consistency and should agree almost exactly.
2. **Matched broad-illumination acquisition (new):** A broad backlight is held fixed relative to the egg. The area detector records one MCX exposure and a sequence of virtual narrow slit rows is recorded from separate MCX runs with the *same source position*, camera acceptance, y-pixel spacing and per-packet normalization. This checks that the modes agree **statistically under equivalent illumination**. It is **not** a physical model of translating an egg beneath a narrow co-moving source, and Monte Carlo noise can lead to substantial differences for low detected counts.

The Streamlit diagnostic presents both images on a **shared colour scale**, reports the integrated transmitted-signal difference and the accepted-packet counts, and clearly identifies this controlled acquisition.

## Scientific limitations

- No GPU/PMCX transport simulations were run during implementation. Actual boundary-padding convergence, physical Beer–Lambert and source-footprint tests must be run on a supported machine.
- The GPU memory footprint cannot be inferred reliably from the 1-byte-per-label domain alone: MCX photon buffers, preprocessing, GPU architecture and output settings affect memory.
- Even at a 0.2 mm voxel, a 0.35 mm eggshell spans fewer than two nominal voxels. A finer uniform grid or a validated split-voxel/mesh model is required for accurate shell transmission.
- The camera remains an idealized angularly gated detector at the MCX upper exit face, not a ray-traced lens/spectrograph. All optical scattering/absorption/refraction is performed by MCX, not local Python.

## References

- Official [MCX and MCXLAB documentation](https://github.com/fangq/mcx/blob/master/mcxlab/README.txt) for `vol`, `unitinmm`, detector fields, and source configurations.
- [MCX command-line reference](https://mcx.space/wiki/index.cgi/wiki/index.cgi?action=browse&id=Doc%2Fmcx_help&revision=3) for `maxdetphoton`, source and output control.
