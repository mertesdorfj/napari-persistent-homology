# napari-persistent-homology

[![License BSD-3](https://img.shields.io/pypi/l/napari-persistent-homology.svg?color=green&v=2)](https://github.com/mertesdorfj/napari-persistent-homology/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-persistent-homology.svg?color=green&v=2)](https://pypi.org/project/napari-persistent-homology)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-persistent-homology.svg?color=green&v=2)](https://python.org)
[![tests](https://github.com/mertesdorfj/napari-persistent-homology/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/mertesdorfj/napari-persistent-homology/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/mertesdorfj/napari-persistent-homology/branch/main/graph/badge.svg?v=2)](https://codecov.io/gh/mertesdorfj/napari-persistent-homology)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-persistent-homology)](https://napari-hub.org/plugins/napari-persistent-homology)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)

**3D shape analysis of binary segmentations using persistent homology.**

----------------------------------

## Overview

`napari-persistent-homology` is an interactive [napari] dock widget for measuring the **size and spacing of structures in 3D binary segmentations** — directly, reproducibly, and without manual point-picking.

Given a segmented 3D volume (a `Labels` mask), the plugin quantifies properties such as the thickness of sheet-like structures, the radius of tubular ones, the distance between separate objects, and the regularity of their spacing inside a parent compartment. Each of these questions is reduced to a single characteristic length. The measurement is based on **persistent homology**: The structure is gradually eroded or dilated and the number of objects / enclosed holes is counted at each step. The shape of the resulting count curve encodes a characteristic length scale (a radius, thickness, or spacing) together with a spread (FWHM) that reflects how uniform or variable that length is. Because morphology grows or shrinks the structure uniformly in every direction, the result is **directionally unbiased** and independent of where measurement endpoints would have been placed by hand, making it well suited to large, automated 3D analyses.

The method is not specific to any particular kind of data: Any 3D binary mask is supported, whether the structures are sheet-like, tubular, or blob-like (e.g. membranes, fibres, vesicles, pores, cells). It was introduced for characterising **mitochondrial cristae** in FIB-SEM electron-microscopy data by [Wang et al., 2024](https://doi.org/10.1038/s42003-024-06045-4) (*Communications Biology* **7**:377); this plugin is based on their work and provides a point-and-click napari interface to their analysis code, removing the need for scripting.

The plugin works on any 3D binary `Labels` layer and supports three analysis modes:

| Mode | What it measures | Example use cases |
|---|---|---|
| **Object radius / half-thickness (erosion)** | The typical radius of solid objects, or half-thickness of sheet/slab structures | Cristae half-thickness, fibre radius, membrane half-width. The plugin also reports the full width / diameter. |
| **Object spacing (dilation)** | The typical distance between separate objects | Inter-mitochondria spacing, fibre-to-fibre distance |
| **Internal spacing (dilation in container)** | The typical spacing between objects inside a parent compartment | Cristae spacing inside a mitochondrion |

In every mode, the plugin builds a **count curve**: At each erosion or dilation step, it records how many separate objects (or enclosed holes) currently exist. This curve rises to a peak and then falls, and two numbers summarise it:

- **Peak location**: *Where* the curve peaks, which corresponds to the characteristic length that is measured. As shown in the paper, this peak sits at *half* the relevant distance, so the plugin reports both the raw peak (a radius / half-thickness, or a half-spacing) and twice it (the full width / diameter, or the full inter-object distance).
- **FWHM (full-width at half-maximum)**: *How wide* the peak is, which indicates how much that length varies across the structure: A sharp peak means a uniform size, a broad peak means a mix of sizes.

Results appear in an embedded plot and can be exported to CSV together with all the parameters used. Under the hood, the analysis runs on a background thread, so the napari UI stays responsive even on large volumes.

## How it works

The method combines mathematical morphology with persistent homology, following Wang et al. (2024):

![Persistent homology standardises distance measurements](https://raw.githubusercontent.com/mertesdorfj/napari-persistent-homology/main/docs/images/Wang_et_al_figure3.jpg)

*The idea behind the method, illustrated on mitochondrial cristae (figure from [Wang et al., 2024](https://doi.org/10.1038/s42003-024-06045-4), CC BY 4.0).* **(a–c)** A distance such as a crista width (purple) or the gap between cristae (yellow) can be drawn in many equally plausible ways — the choice of start- and end-point is subjective, so manual measurements are hard to reproduce. **(d–e)** Persistent homology removes that ambiguity. The structure is grown step by step (red arrows); as two opposing surfaces approach, the gap between them closes into an enclosed **hole** (yellow hatching) that later vanishes once the surfaces merge. The step at which the most holes are open (here, dilation round 2) is the moment most surfaces just touch — and since each surface has grown by the same amount, that step equals **half** the gap between them, independent of orientation. **(d vs. e)** A smoother surface (d) opens and closes its holes over a narrow range of steps, whereas a more curved or rough surface (e) keeps them open for longer; this shows up as a **wider count-curve peak (larger FWHM)**, which is why the FWHM acts as a measure of surface curvature / roughness.

The method by Chenhao Wang et al. turns this idea into three steps:

1. **Subpixel morphology**: The binary mask is repeatedly dilated or eroded by a *fraction* of a voxel per step. Rather than the one-voxel-at-a-time classical operators, the volume is evolved under the level-set PDE `dU/dt = ±|∇U|` using a first-order Osher–Sethian upwind scheme (paper eqs. 3–6). With the default step `λ = 0.1`, ten steps equal one full voxel layer added (dilation) or removed (erosion).

2. **Counting**: After each step, the (soft) mask is binarised at 0.5 and passed to a 26-connected 3D connected-components labeller:
   - **Erosion → object count**: As the object shrinks, thin regions pinch off and the foreground briefly splits into more components before vanishing.
   - **Dilation → hole count**: As objects grow, the gaps between them close into enclosed holes that later disappear when surfaces merge. (The outer background is always one extra component, so it is subtracted off). In *internal-spacing* mode, holes are counted only inside the container compartment: The dilated mask is restricted to the container first, so gaps outside it are ignored. In Wang et al. (2024), this is used to confine the analysis to the interior of each mitochondrion.

3. **Feature extraction**: The count curve is summarised by two numbers (the first few steps are skipped according to a user-defined offset value, since the curve is noisiest there):
   - **Peak location**: Detected directly on the **raw** count curve as the step at which the count is highest. Rather than a plain global maximum, the algorithm identifies every local maximum from the first difference of the curve and applies a noise filter that discards a spurious early peak when it is followed by a stable `count = 1` plateau and then a real peak later on. This peak is the step where the most surfaces are simultaneously touching, which happens when each surface has moved inward (erosion) or outward (dilation) by half the distance separating it from its neighbour — so the peak measures *half* of that distance (a radius / half-thickness in erosion, or a half-gap in dilation), and the plugin reports both this value and twice it (the full thickness or full spacing).
   - **FWHM**: The width of the count-curve peak at half its height, a proxy for surface roughness / curvature: rougher, more curved surfaces keep holes and objects alive over more rounds, widening the peak. The half-max threshold is set from the raw peak value, and the left / right edges of the FWHM are found on a **moving-average-smoothed** version of the curve (window size = one full voxel round, i.e. `round(1 / λ)` samples). Smoothing the curve for the edge walks stabilises the width against sample-to-sample noise; keeping the peak location / height from the raw curve preserves the noise-filtering behaviour above.

Both values are returned already in voxel units (each internally multiplied by `λ`) and, if a physical voxel size is set in the widget, then optionally converted to nm / µm from there.

## Installation

You can install `napari-persistent-homology` via [pip]:

```bash
pip install napari-persistent-homology
```

If napari is not already installed, you can install the plugin together with napari and a Qt backend via:

```bash
pip install "napari-persistent-homology[all]"
```

To install the latest development version straight from GitHub:

```bash
pip install git+https://github.com/mertesdorfj/napari-persistent-homology.git
```

The plugin requires Python ≥ 3.10.

## Quick start

1. Launch napari (`napari` from your terminal, or from your IDE).
2. Open the widget from the menu bar under either:
   - **Plugins → Persistent Homology (Persistent Homology 3D)**, or
   - **Layers → Measure → Persistent Homology (Persistent Homology 3D)**.
3. Load a 3D binary segmentation (your own `.tif` / `.npy` file, or the bundled sample under **File → Open Sample → Cristae binary mask 3D (Persistent Homology 3D)**).
4. Choose an analysis mode and click **Run Analysis**.
5. Inspect the count curve and the resulting measurements (radius / spacing and the full-width at half-maximum, FWHM) in the **Results** section.
6. Click **Save Measurement Results** to export the summary values to CSV, or **Save Count Curve Data & Figures** to export the count-curve data (CSV) plus all count curve plots (PNG).

## How to use the widget

![The widget on startup](https://raw.githubusercontent.com/mertesdorfj/napari-persistent-homology/main/docs/images/plugin_screenshot_start.png)

*The widget docked in napari on startup, before any analysis has been run.*

The widget is laid out top to bottom:

1. **Input** — pick the `Labels` layer that contains your segmentation. Two extra controls decide *how* the labels are analysed:
   - **Analyze** — `All combined` (the default: every selected label is merged into one binary volume and produces a single population-average curve, exactly as in the paper) or `Each object` (every selected label is analysed separately, giving one curve and one set of measurements per object).
   - **Label IDs** — `all` (every non-zero label) or a comma-separated list such as `1,3,5`. In *Each object* mode the listed labels are analysed one at a time; in *All combined* mode they are merged. A plain binary mask (only label `1`) behaves identically either way.

   In *Internal spacing* mode, a second dropdown appears for selecting the container layer.
2. **Analysis** — pick one of the three modes (see table above). Per-object analysis works in all three.
3. **Parameters** — basic parameters are always visible; click **▶ Advanced mode** to reveal Connectivity, Offset, and the "Rank peaks by smoothed value" checkbox.
4. **Physical Scale** — enter your voxel size and physical unit if you want results in nm / µm in addition to voxels.
5. **Run Analysis** — starts the computation on a background thread. In per-object mode the single progress bar spans all objects.
6. **Plot** — the raw and smoothed count curve, with the detected peak marked (dashed vertical line) and the FWHM shown as a dashed horizontal bar. In per-object mode a **Show object** selector appears above the plot: pick `Object <label>` to display that object's curve (and update the Results below), or choose one of the two overlay entries — **All (overlay – smoothed)** or **All (overlay – raw)** — to overlay every object's curve in one plot for comparison (offered for up to 10 objects). In the overlays, each object's curve is drawn in the **same colour napari assigns its label**, so you can match curves to objects at a glance. A **Highlight in viewer** checkbox (on by default) isolates the selected object in the source `Labels` layer using napari's native single-label view.
7. **Results** — a caption naming the current object (`All combined`, or `Object <label>`), then the raw peak (radius / half-spacing), twice that (full width / inter-object spacing), and the full-width at half-maximum (FWHM) in voxels (and, if voxel size is set, in physical units too).
8. **Save Measurement Results** and **Save Count Curve Data & Figures** — two separate export buttons. The first writes a small CSV with just the summary values from the Results section (one row per object); the second writes the raw + smoothed count curve to CSV *and* saves **every** count curve plot as PNG(s) — in per-object mode one per object plus both overlays (collected in a subfolder named after your chosen filename), not just the plot currently on screen. See **Outputs** below.

### Input parameters

| Parameter | Where | Default | Description |
|---|---|---|---|
| **Segmentation layer** | Input | — | The `Labels` layer containing your segmentation. In *All combined* mode all selected labels are treated as one binary foreground; in *Each object* mode each label is analysed separately. |
| **Analyze** | Input | All combined | `All combined` merges the selected labels into a single binary volume and reports one population-average curve (the original behaviour, matching the paper). `Each object` analyses every selected label separately, producing one curve and one set of measurements per object. |
| **Label IDs** | Input | all | Which labels take part. `all` (or an empty field) selects every non-zero label; a list like `1,3,5` (brackets and spaces are tolerated) selects specific ones. Combined with the **Analyze** toggle above: *Each object* + `1,3,5` analyses those three one at a time; *All combined* + `1,3,5` merges just those three. |
| **Container layer** | Input (internal-spacing mode only) | — | A second `Labels` layer defining the parent compartment for *Internal spacing* mode. In per-object mode it is cropped to each object's bounding box automatically. |
| **Mode** | Analysis | Object radius / half-thickness | One of the three analysis modes described above. |
| **Lambda** | Parameters | 0.1 (minimum) | Subpixel step size in voxel-length units, in the range `0.1`–`1.0`. `0.1` means 10 morphology steps per voxel. Smaller values are more accurate but slower; the minimum is set to `0.1` because smaller steps quickly become impractical without offering further accuracy gains. |
| **Max steps** | Parameters | 100 | Total number of morphology steps. Limits the maximum measurable distance to `max_steps × Lambda` voxels (e.g. 100 × 0.1 = 10 voxels). |
| **Connectivity** | Advanced | 26 | 3D connected-component neighbourhood used by the counters (options: `6`, `18`, `26`). `26` is the full 3D neighbourhood (face + edge + corner) and is recommended for most datasets. |
| **Offset** | Advanced | `int(1 / Lambda)` | Number of initial steps to skip when searching for the peak. The very first steps are dominated by noise and small irregularities in the surface rather than real structure, which can create a misleading spike near step 0; ignoring them keeps the search on the true peak. The default tracks `Lambda` so that one full voxel layer is always skipped (e.g. `10` at the default `Lambda = 0.1`); whenever `Lambda` is changed in the widget, the offset is updated automatically. The user is free to override it manually afterwards — the next `Lambda` change resets it again. |
| **Rank peaks by smoothed value** | Advanced | unchecked | Optional peak-selection modifier. Candidate peaks (local maxima) are always identified on the raw count curve. When this checkbox is **unticked** (default), the tallest candidate is picked by comparing raw count values — the fastest / cleanest behaviour on well-formed segmentations. When **ticked**, candidates are ranked by their moving-average smoothed value instead, which prevents a single-sample noise spike on a noisy curve from winning the argmax. It has **no visible effect on curves that expose only one clear peak** (typical for erosion / object-count curves), so it is most useful for noisy dilation / hole-count curves. |
| **Voxel size X / Y / Z** | Physical Scale | `—` in `vox` mode, `1.0` in `nm` / `µm` mode | The physical voxel size along each axis, used to convert results from voxels to nm / µm. Defaults to `1.0` the first time you switch the unit to `nm` or `µm` - adjust to your actual voxel size. Stays at `—` and is skipped while the selected unit is `vox`. All three values must be greater than 0 when the unit is `nm` or `µm`; clicking 'Run Analysis' with a 0 in any axis raises an error and blocks the run. |
| **Unit** | Physical Scale | vox | The physical unit in which results are reported (available options: `vox`, `nm`, `µm`). |

> **Anisotropic voxels.** The analysis pipeline operates on the binary volume isotropically - it has no internal knowledge of physical voxel anisotropy. When you provide the voxel size, the plugin converts the voxel-unit results to physical units using the **arithmetic mean** of the X / Y / Z voxel sizes. For datasets with strong anisotropy, the physical-unit result is therefore an approximation and should be treated with caution; consider resampling to an isotropic grid first if exact physical sizes matter (might be added in a future version of this plugin).

### Outputs

![The widget displaying analysis results](https://raw.githubusercontent.com/mertesdorfj/napari-persistent-homology/main/docs/images/plugin_screenshot_result.png)

*The widget following a completed run in **Object radius / half-thickness (erosion)** mode, showing the count curve with the detected peak and FWHM bar, and the corresponding measurements in the Results section below.*

After a successful run, the widget displays:

- **Plot** — the **raw** count curve (light blue), the **moving-average-smoothed** curve (darker blue), a dashed vertical line marking the detected peak, and a dashed orange horizontal bar at half-peak height spanning the FWHM. A legend reports the peak and FWHM values in voxels. The x-axis is the morphology-step index (not voxels). The peak marker sits at the location returned by the noise-tolerant peak detector, which works on the **raw** curve; the FWHM bar spans the range where the **smoothed** curve stays above half of the raw peak height (see the note on raw vs. smoothed below).
- **Radius / half-thickness (erosion)** — *shown only in erosion mode.* The peak location of the object-count curve, detected on the **raw** curve. The typical radius of solid objects, or the half-thickness of sheet-like structures.
- **Width / thickness (erosion)** — *shown only in erosion mode.* Exactly twice the radius / half-thickness. For solid objects this is the full diameter; for sheet/slab structures it is the full thickness.
- **Half-spacing (dilation)** — *shown only in dilation modes.* The peak location of the hole-count curve, detected on the **raw** curve. Per the paper this peak is *half* the average gap between objects, so it is reported as a half-spacing.
- **Inter-object spacing (dilation)** — *shown only in dilation modes.* Exactly twice the half-spacing. The full typical distance between separate objects.
- **Full-width at half-maximum (FWHM)** — *shown in every mode.* The spread of the count curve around its peak, measured at half of the peak height. Larger values indicate more variability in the size distribution of the analysed structures (a proxy for surface roughness / curvature). The half-max threshold is set from the **raw** peak value; the left / right edges of the FWHM are then found on the **moving-average-smoothed** curve so the width is stable against sample-to-sample noise.

> **Raw vs. smoothed at a glance.** By default, the reported peak *location* and *height* come from the raw count curve (with a noise filter that discards spurious early peaks). The FWHM's *edges* are measured on the moving-average-smoothed curve, using the raw peak value as the half-max threshold. The smoothed curve in the plot is the same one the FWHM edge walks use.
>
> Ticking **"Rank peaks by smoothed value"** in the Advanced section keeps the local-max candidates and the noise filter on the raw curve, but ranks candidates by their smoothed value when picking the tallest — helpful when a single-sample noise spike on a noisy hole-count curve would otherwise win the argmax. See the parameter table above for the full description.

In every mode, values are shown in the chosen physical unit first with the voxel value in brackets (e.g. `25.00 nm  (5.00 vox)`) when a physical unit is selected and the voxel sizes are set. With unit `vox`, only the voxel value is shown.

> **Note on switching modes.** The Results section is "frozen" - it always shows the labels and values from the *last* run. The labels only update when you click 'Run Analysis' again.

There are **two export buttons** below the Results section:

**Save Measurement Results**: Prompts for a CSV path and writes the summary values shown in the Results compartment, **one row per analysed object**:

- A title line — `# napari-persistent-homology — Summary Results`.
- A comment header — `# Mode: …`, `# Parameters: …` (which now also records `analyze=combined|each` and the `label_ids`), and `# Voxel size: …` (or a note that it was not set).
- A table led by a `Label_ID` column (`all` for a combined run, the label value for each per-object row), followed by three metric columns in voxels (and `_<unit>` variants when a physical voxel size is set, e.g. `_nm` / `_um`):
  - In erosion mode: `Radius_half_thickness_vox`, `Width_thickness_vox`, `FWHM_vox`
  - In dilation modes: `Half_spacing_vox`, `Inter_object_spacing_vox`, `FWHM_vox`
  - Objects with no detected peak are written as `NaN`.

**Save Count Curve Data & Figures**: Prompts for a single path and writes the count-curve CSV plus **all** count curve plots as PNGs:

- `<base>.csv` — title line `# napari-persistent-homology — Object Count Curve` (erosion mode) or `# napari-persistent-homology — Hole Count Curve` (dilation modes), the same metadata header as the summary file, plus the full per-round count curve in tidy/long form. Columns are `Label_ID`, `Erosion_round` or `Dilation_round`, `Count_raw`, and `Count_smoothed`, with one block of rows per object (`all` for a combined run). The CSV always sits directly in the directory you chose.
- The plot PNGs (all rendered at 150 DPI with a tight bounding box):
  - **Single-object analysis** (aggregate / binary case, or a per-object run with a single object): the one plot is saved directly next to the CSV as `<base>.png`.
  - **Multi-object per-object analysis:** all plots go into a subfolder named exactly after your chosen filename (the CSV stem, no suffix appended). Inside, files are named by what the curve counts: `object_count_obj_<label>.png` (erosion) or `hole_count_obj_<label>.png` (dilation) for each object with a detected peak, plus `<prefix>_overlay_smoothed.png` and `<prefix>_overlay_raw.png` when the overlays are available (2–10 objects). Objects with no detected peak are skipped (they have no curve).

Each Save dialog **prefills a structured default name** — `<Mode>_<analyze>_<kind>.csv`, e.g. `Object_radius_per_object_measurements.csv` or `Object_spacing_combined_objects_count_curve_data.csv` — which you can accept or edit. `<Mode>` is `Object_radius` (erosion), `Object_spacing` (object-spacing dilation) or `Internal_spacing` (internal-spacing); `<analyze>` is `per_object` or `combined_objects`. If you pick e.g. `result.csv`, a single-object erosion run writes `result.csv` + `result.png`; a multi-object erosion run writes `result.csv` + a `result/` folder alongside it. **Existing files are never overwritten** — if the name already exists, it is auto-incremented (`result_2`, `result_3`, …), so saving several runs into the same directory keeps every one. Any `.csv` or `.png` extension you type is stripped first so the CSV and single-plot PNG share the base name.


### Sample data

Two 3D FIB-SEM samples are bundled with the plugin, both under **File → Open Sample** (Persistent Homology 3D):

- **Cristae binary mask 3D** — a 114 × 163 × 234 `uint8` binary volume. A quick way to confirm the plugin works and to explore the '*Object radius / half-thickness (erosion)*' and '*Object spacing (dilation)*' modes.
- **Cristae multi-label mask 3D** — an image + labels pair (opens two layers) with 5 individually-labelled cristae (IDs 1–5). Ideal for trying **per-object analysis**: set **Analyze** to *Each object*, leave **Label IDs** on `all`, run, then step through the objects with the **Show object** selector.

## Citation

If you use this plugin in your research, please cite the original paper that the analysis pipeline is based on:

> Wang, C., Østergaard, L., Hasselholt, S., & Sporring, J. (2024). *A semi-automatic method for extracting mitochondrial cristae characteristics from 3D focused ion beam scanning electron microscopy data*. Communications Biology, 7, 377. https://doi.org/10.1038/s42003-024-06045-4

## Contributing

Contributions are very welcome. Please:

1. Fork the repository and create a feature branch.
2. Install the development environment (the `--group` flag needs pip ≥ 25.1, which is the first version to support PEP 735 dependency groups):
   ```bash
   pip install -e . --group dev
   ```
   On older pip, install the dev tools by hand instead:
   ```bash
   pip install -e .
   pip install "napari[qt]" pytest pytest-cov pytest-qt
   ```
3. Make sure the test suite still passes and add tests for any new behaviour:
   ```bash
   python -m pytest tests/ -v
   ```
4. Open a pull request describing your change.

Please ensure the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license, `napari-persistent-homology` is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description (napari version, plugin version, OS, and the smallest input that reproduces the problem if possible).

## Acknowledgements

This plugin wraps the research code originally written by **Chenhao Wang** and colleagues. The napari plugin scaffolding was generated from the official [napari plugin template](https://github.com/napari/napari-plugin-template).

[napari]: https://github.com/napari/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[file an issue]: https://github.com/mertesdorfj/napari-persistent-homology/issues
[pip]: https://pypi.org/project/pip/
