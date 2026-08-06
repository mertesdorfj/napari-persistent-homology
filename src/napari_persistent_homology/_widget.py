"""
Persistent-homology dock widget for napari.

Provides an interactive Qt dock widget that wraps the analysis functions in
'ph_functions.py' ('persistent_homology_erosion',
'persistent_homology_dilation', and
'persistent_homology_dilation_internal_object') for non-programmer users.
The widget lets the user pick one of the three analysis modes, choose a
Labels layer (and optionally a container Labels layer for the internal-
spacing mode), tune the relevant parameters, run the computation on a
background thread, and inspect / save the results.

Architecture
------------
1. UI construction lives in '_build_ui' and is purely declarative — it just
   builds the Qt widget tree. Every interactive control is connected to a
   matching '_on_*' callback further down.
2. Heavy computation runs on a background thread via napari's
   'thread_worker' decorator (see '_run_analysis'). The main Qt thread
   stays responsive while erosion / dilation steps execute.
3. Progress is reported back to the GUI through a thread-safe
   'QObject' + 'Signal' bridge ('_ProgressEmitter') — Qt signals deliver
   queued events on the main thread, so the worker can call into it from
   any thread without locking.
4. When the worker finishes, '_on_result' converts the subpixel-step
   measurements to voxel units (and to nm / µm if a voxel pitch was set),
   updates the plot and the result labels, and enables the two save buttons
   ("Save Results" and "Save Curve & Plot").
"""

import csv
import os
from math import ceil

import matplotlib as mpl
import napari.layers
import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg,  # type: ignore[no-redef]
    )

from matplotlib.figure import Figure

from .ph_functions import (
    compute_homology_stats_v2,
    label_subvolume,
    moving_average,
    persistent_homology_dilation,
    persistent_homology_dilation_internal_object,
    persistent_homology_erosion,
)

##############################################################################
# Mode constants
#
# Human-readable mode labels shown in the Analysis-Mode dropdown, and a
# lookup table mapping each label to the short key used internally by
# '_run_analysis' to pick the right persistent-homology function.
##############################################################################

_MODE_EROSION = 'Object radius / half-thickness  (erosion)'
_MODE_DILATION = 'Object spacing  (dilation)'
_MODE_DILATION_INTERNAL = 'Internal spacing  (dilation in container)'

_MODE_KEY = {
    _MODE_EROSION: 'erosion',
    _MODE_DILATION: 'dilation',
    _MODE_DILATION_INTERNAL: 'dilation_internal',
}

# Per-object analysis: the 'Analyze:' toggle values, and the maximum object
# count for which the "All (overlay)" plot entry is offered (beyond this the
# overlay is too cluttered to read, so only the one-at-a-time selector remains).
_ANALYZE_COMBINED = 'All combined'
_ANALYZE_EACH = 'Each object'
_MAX_OVERLAY = 10

# Curve label per mode (what the count curve measures). Object count for
# erosion, hole count for the two dilation variants.
_CURVE_LABEL = {
    'erosion': 'Object count',
    'dilation': 'Hole count',
    'dilation_internal': 'Hole count (internal)',
}

# Mode + analyze tokens used to prefill a structured default filename in the
# Save dialogs (e.g. 'Object_radius_per_object_measurements.csv'). The user can
# freely edit the name; the chosen stem is then reused verbatim for the CSV, the
# single-object PNG, and the multi-object plot subfolder — so nothing long is
# appended on top and the structure lives in the (editable) default name.
_MODE_NAME_TOKEN = {
    'erosion': 'Object_radius',
    'dilation': 'Object_spacing',
    'dilation_internal': 'Internal_spacing',
}
_ANALYZE_NAME_TOKEN = {True: 'per_object', False: 'combined_objects'}

# Filename prefix for each saved per-object PNG, naming what the curve counts:
# object count for erosion, hole count for the two dilation variants (matches
# '_CURVE_LABEL'). Used to build 'object_count_obj_3.png' / 'hole_count_obj_1.png'.
_COUNT_FILE_PREFIX = {
    'erosion': 'object_count',
    'dilation': 'hole_count',
    'dilation_internal': 'hole_count',
}


def parse_label_ids(text: str, available_ids) -> list[int]:
    """Parse the 'Label IDs' text field into a list of label values.

    Parameters
    ----------
    text: str
        Raw text from the field. ``'all'`` (case-insensitive) or an empty
        string selects every available label; otherwise a list of integers
        written as ``'1,3,5'``, ``'[1, 3, 5]'`` or ``'1 3 5'``.
    available_ids: sequence of int
        The label values actually present in the segmentation (non-zero).

    Returns
    -------
    list of int
        Sorted, de-duplicated label IDs to analyse.

    Raises
    ------
    ValueError
        If the text cannot be parsed, or references a label that is not
        present in the segmentation.
    """
    available = sorted(int(v) for v in available_ids)
    stripped = text.strip().lower()
    if stripped in ('', 'all'):
        return available

    tokens = (
        stripped.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    )
    try:
        ids = [int(tok) for tok in tokens]
    except ValueError:
        raise ValueError(
            f"Could not parse label IDs '{text}'. "
            "Use 'all' or a list like 1,3,5."
        ) from None
    if not ids:
        raise ValueError(
            f"Could not parse label IDs '{text}'. "
            "Use 'all' or a list like 1,3,5."
        )

    available_set = set(available)
    missing = sorted({i for i in ids if i not in available_set})
    if missing:
        raise ValueError(
            f'Label(s) {missing} not present in the segmentation. '
            f'Available: {available}.'
        )
    return sorted(set(ids))


##############################################################################
# Thread-safe progress emitter
##############################################################################


class _ProgressEmitter(QObject):
    """
    Bridge progress updates from the worker thread to the main Qt thread.

    Qt 'Signal' objects deliver events via the event loop of the thread that
    owns the receiver. Because this 'QObject' is created on the main GUI
    thread, emitting 'progress_changed' from the background worker results
    in a queued connection — the slot runs safely on the GUI thread, even
    though 'emit' was called from elsewhere. This is the standard pattern
    for thread-to-thread communication in Qt.
    """

    # Emitted as (current_step, total_steps). Connected to the widget's
    # '_on_progress' slot to drive the QProgressBar.
    progress_changed = Signal(int, int)

    def step(self, current: int, total: int) -> None:
        """Forward one progress tick to the GUI thread via the Qt signal."""
        self.progress_changed.emit(current, total)


##############################################################################
# Background analysis worker
#
# Module-level (not a method of the widget) so that '@thread_worker' can
# decorate it cleanly. Runs the chosen persistent-homology function on a
# napari worker thread, then computes radius / FWHM in voxel units.
##############################################################################


def _make_step_callback(base, total_steps, step_callback):
    """Return a per-object progress callback offset into the global range.

    Each 'persistent_homology_*' function reports progress as
    'step_callback(step + 1, max_steps)'. For a per-object run of N objects
    we want one continuous bar over 'N * max_steps', so this wrapper shifts
    each object's local step by 'base' (= object_index * max_steps) and
    reports the shared total.
    """

    def cb(step, _local_total):
        if step_callback is not None:
            step_callback(base + step, total_steps)

    return cb


def _compute_series(
    obj_mask, container_mask, mode_key, max_steps, Lambda, connectivity, cb
):
    """Run one persistent-homology analysis and return its count curve.

    Picks the object-count curve for erosion and the hole-count curve for
    the two dilation variants — the same choice the single-curve pipeline
    made previously.
    """
    if mode_key == 'erosion':
        obj_counts, _ = persistent_homology_erosion(
            obj_mask,
            max_steps=max_steps,
            Lambda=Lambda,
            Connectivity=connectivity,
            step_callback=cb,
        )
        return obj_counts
    if mode_key == 'dilation':
        _, hole_counts = persistent_homology_dilation(
            obj_mask,
            max_steps=max_steps,
            Lambda=Lambda,
            Connectivity=connectivity,
            step_callback=cb,
        )
        return hole_counts
    _, hole_counts = persistent_homology_dilation_internal_object(
        obj_mask,
        container_mask,
        max_steps=max_steps,
        Lambda=Lambda,
        Connectivity=connectivity,
        step_callback=cb,
    )
    return hole_counts


@thread_worker
def _run_analysis(
    label_data: np.ndarray,
    container_data,
    mode_key: str,
    analyze_each: bool,
    label_ids: list,
    Lambda: float,
    max_steps: int,
    connectivity: int,
    offset: int,
    rank_peaks_by_smoothed: bool,
    step_callback,
):
    """
    Run the selected persistent-homology analysis on a background thread.

    Handles both the aggregate ('analyze_each=False' → one binarized volume,
    one curve — the original behaviour) and the per-object mode
    ('analyze_each=True' → one cropped binary sub-volume per label, one curve
    each). For each job it dispatches to one of the three analysis functions
    in 'ph_functions.py' based on 'mode_key', picks the relevant count curve
    (object-count for erosion, hole-count for the two dilation variants), and
    reduces it to two scalar shape descriptors using
    'compute_homology_stats_v2'.

    Parameters
    ----------
    label_data: ndarray
        Raw (non-binarized) 3D label volume from the segmentation layer.
    container_data: ndarray or None
        Raw container mask, required when 'mode_key == "dilation_internal"',
        ignored otherwise.
    mode_key: str
        One of 'erosion', 'dilation', 'dilation_internal'.
    analyze_each: bool
        When True, analyse each label in 'label_ids' separately (one cropped
        sub-volume per label); when False, binarize the union of 'label_ids'
        into a single volume — the original aggregate behaviour.
    label_ids: list of int
        Label values to include.
    Lambda, max_steps, connectivity:
        Forwarded to the underlying persistent-homology function. See
        'ph_functions.persistent_homology_*' for details.
    offset:
        Initial samples to skip when locating the peak — forwarded to
        'compute_homology_stats_v2'.
    rank_peaks_by_smoothed: bool
        Optional v2 flag forwarded to 'compute_homology_stats_v2'.
        When True, the argmax that picks the tallest surviving
        local maximum ranks candidates by their moving-average
        smoothed value instead of the raw count. Candidate
        identification and the noise filter still run on the raw
        curve. Useful on noisy data where a single-sample raw spike
        would otherwise win the argmax; has no visible effect on
        curves with only one clear peak (typical erosion).
    step_callback: callable
        Per-step progress reporter, normally
        '_ProgressEmitter.step'.

    Returns
    -------
    dict
        '{"analyze_each", "mode_key", "curve_label", "objects"}' where
        'objects' is a list of per-object dicts, each
        '{"label_id", "series", "radius_vox", "fwhm_vox"}' with
        'radius_vox' / 'fwhm_vox' already converted to voxel units by
        'compute_homology_stats_v2'. 'label_id' is 'None' for the aggregate
        (combined) job.
    """
    from napari.utils import progress

    curve_label = _CURVE_LABEL[mode_key]
    want_container = mode_key == 'dilation_internal'

    # Build the job list: (label_id, obj_mask, container_mask).
    jobs = []
    if analyze_each:
        for lid in label_ids:
            obj_mask, cont_mask, _ = label_subvolume(
                label_data,
                lid,
                container_data if want_container else None,
            )
            jobs.append((int(lid), obj_mask, cont_mask))
    else:
        # Aggregate: whole binarized volume (no crop). For the default 'all'
        # label set (and any plain binary mask) this is byte-for-byte identical
        # to the pre-per-object 'data > 0' path; a combined *subset* of labels
        # intentionally analyses only 'np.isin(label_data, label_ids)'.
        obj_mask = np.isin(label_data, label_ids).astype(np.uint8)
        cont_mask = (
            (container_data > 0).astype(np.uint8) if want_container else None
        )
        jobs.append((None, obj_mask, cont_mask))

    n_jobs = len(jobs)
    total_steps = n_jobs * max_steps
    objects = []

    with progress(total=0):  # indeterminate spinner in napari activity bar
        for j, (label_id, obj_mask, cont_mask) in enumerate(jobs):
            base = j * max_steps

            # Internal mode with an empty cropped container: hole counting is
            # meaningless, so flag this object as "no peak" instead of
            # crashing, and jump the progress bar past its slice.
            if want_container and (cont_mask is None or cont_mask.sum() == 0):
                objects.append(
                    {
                        'label_id': label_id,
                        'series': np.zeros(max_steps + 1),
                        'radius_vox': 0.0,
                        'fwhm_vox': 0.0,
                    }
                )
                if step_callback is not None:
                    step_callback(base + max_steps, total_steps)
                continue

            cb = _make_step_callback(base, total_steps, step_callback)
            series = _compute_series(
                obj_mask,
                cont_mask,
                mode_key,
                max_steps,
                Lambda,
                connectivity,
                cb,
            )
            # v2 returns (FWHM, max_location, max_count) already in voxel
            # units (multiplied by Lambda), with internal noise-tolerant peak
            # detection. 'rank_peaks_by_smoothed' picks between raw / smoothed
            # weighting when choosing the tallest local peak — see
            # 'find_max_location_v2'.
            fwhm_vox, radius_vox, _max_count = compute_homology_stats_v2(
                series,
                offset=offset,
                Lambda=Lambda,
                rank_peaks_by_smoothed=rank_peaks_by_smoothed,
            )
            objects.append(
                {
                    'label_id': label_id,
                    'series': series,
                    'radius_vox': float(radius_vox),
                    'fwhm_vox': float(fwhm_vox),
                }
            )

    return {
        'analyze_each': analyze_each,
        'mode_key': mode_key,
        'curve_label': curve_label,
        'objects': objects,
    }


##############################################################################
# Widget
##############################################################################


class PersistentHomologyWidget(QWidget):
    """
    Napari dock widget for 3D persistent-homology shape analysis.

    The widget is registered in 'napari.yaml' and opens via either
    'Plugins > Persistent Homology' or 'Layers > Measure > Persistent
    Homology Analysis'.

    Layout (top to bottom)
    ----------------------
    1. **Input** — segmentation layer dropdown (always shown) plus an
       optional container layer dropdown (only visible in internal-spacing
       mode).
    2. **Analysis** — mode dropdown (object radius / object spacing /
       internal spacing).
    3. **Parameters** — 'Lambda', 'max_steps', and a collapsible
       "Advanced mode" panel with 'Connectivity', 'Sigma', and 'Offset'.
    4. **Physical scale** — X / Y / Z voxel-size spinboxes and a unit
       selector ('vox' / 'nm' / 'µm'). Switching directly between nm and
       µm leaves the entered values unchanged.
    5. **Run** — button, progress bar, and a status label.
    6. **Plot** — embedded matplotlib figure showing the raw + smoothed
       count curve with the detected peak marked.
    7. **Results** — radius and FWHM, in voxels and physical units.
    8. **Save** — exports series, parameters, and results to CSV.

    Threading
    ---------
    The widget itself lives on the main Qt thread. The heavy computation
    runs in '_run_analysis', a 'thread_worker' that executes in a Qt
    worker thread. Progress updates are pushed back via the
    '_ProgressEmitter' signal bridge so the UI stays responsive.

    Internal state
    --------------
    - 'self._results': list of per-object display records from the most
      recent run (one entry in aggregate mode, N in per-object mode), used
      by the object selector and both save buttons. Empty until the first
      run completes successfully.
    - 'self._run_params': dict of parameter values captured at the start
      of the run; written into the CSV header by '_on_save_clicked'.
    - 'self._previous_unit': last selected unit string, used by
      '_on_unit_changed' to detect a 'vox → physical unit' transition
      and seed default voxel sizes on that switch.
    """

    def __init__(self, viewer: Viewer):
        """
        Build the widget and wire it up to the napari viewer.

        Connects to 'viewer.layers.events.inserted / removed' so the
        Labels-layer dropdowns stay in sync as the user adds or removes
        layers.

        Parameters
        ----------
        viewer: napari.viewer.Viewer
            The napari viewer instance, injected automatically by napari
            when the widget is created from the Plugins menu.
        """
        super().__init__()
        self._viewer = viewer
        # Per-object display records from the last successful run (one entry
        # in aggregate mode, N in per-object mode). Empty enables/disables the
        # Save buttons and gates the object selector.
        self._results: list = []
        self._curve_label = (
            ''  # what the count curve measures (mode-dependent)
        )
        self._run_params: dict = {}

        # Floor the dock at the natural width of the longest row — the
        # Physical Scale line with X / Y / Z voxel-size spinboxes plus
        # the unit dropdown. Below this width the row clips and the
        # napari dark-theme transient scrollbar starts overlaying the
        # right-edge controls.
        self.setMinimumWidth(520)

        self._build_ui()
        self._refresh_layer_combos()

        viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        viewer.layers.events.removed.connect(self._refresh_layer_combos)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """
        Construct the Qt widget tree (called once from '__init__').

        Each numbered section in the body corresponds to one row in the
        widget layout described in the class docstring above.
        """
        # Wrap all content in a QScrollArea so the dock gracefully
        # acquires a vertical scrollbar when napari is shrunk below the
        # plugin's natural height. 'setWidgetResizable(True)' makes the
        # inner widget resize to match the scroll area's width (so
        # descriptions still wrap horizontally instead of triggering a
        # horizontal scrollbar). 'main_layout' from here on is the
        # layout of the inner widget — the rest of the build code is
        # unchanged.
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        # Add a little vertical breathing room inside every section box so
        # the first content row doesn't sit flush against the QGroupBox
        # title. Applied once here so all sections look consistent.
        inner.setStyleSheet('QGroupBox { padding-top: 10px; }')
        main_layout = QVBoxLayout(inner)
        main_layout.setSpacing(6)
        # Tight margins on the left / top / bottom; the right margin is
        # kept generous so napari's dark-theme transient scrollbar
        # (~15 px wide) lands in clear space instead of overlapping
        # controls and descriptions on the right edge.
        main_layout.setContentsMargins(6, 6, 18, 6)

        # 1 ── Input layers ─────────────────────────────────────────────────
        input_group = QGroupBox('Input')
        input_form = QFormLayout()
        input_group.setLayout(input_form)

        self._seg_combo = QComboBox()
        input_form.addRow('Segmentation layer:', self._seg_combo)

        self._container_label = QLabel('Container layer:')
        self._container_combo = QComboBox()
        input_form.addRow(self._container_label, self._container_combo)
        self._container_label.hide()
        self._container_combo.hide()

        # Per-object selection: an 'Analyze:' toggle (combine every selected
        # label into one volume, or analyse each label separately) plus a
        # 'Label IDs' text field naming which labels take part. The default
        # ('All combined' + 'all') reproduces the original single-curve
        # behaviour, so a plain binary mask is unaffected.
        self._analyze_combo = QComboBox()
        self._analyze_combo.addItems([_ANALYZE_COMBINED, _ANALYZE_EACH])
        input_form.addRow('Analyze:', self._analyze_combo)

        self._label_ids_edit = QLineEdit('all')
        self._label_ids_edit.setPlaceholderText('all   or   1,3,5')
        input_form.addRow('Label IDs:', self._label_ids_edit)

        label_ids_desc = QLabel(
            "'all' = every label. In 'Each object' mode a comma-separated "
            'list (e.g. 1,3,5) analyses those labels one at a time; in '
            "'All combined' mode the listed labels are merged into a single "
            'volume.'
        )
        label_ids_desc.setWordWrap(True)
        label_ids_desc.setStyleSheet(
            'color: gray; font-style: italic; padding-left: 12px;'
        )
        input_form.addRow(label_ids_desc)

        main_layout.addWidget(input_group)

        # 2 ── Analysis mode ────────────────────────────────────────────────
        analysis_group = QGroupBox('Analysis')
        analysis_form = QFormLayout()
        analysis_group.setLayout(analysis_form)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(
            [_MODE_EROSION, _MODE_DILATION, _MODE_DILATION_INTERNAL]
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        analysis_form.addRow('Mode:', self._mode_combo)

        main_layout.addWidget(analysis_group)

        # 3 ── Parameters ───────────────────────────────────────────────────
        params_group = QGroupBox('Parameters')
        params_layout = QVBoxLayout()
        params_group.setLayout(params_layout)

        # Description-label helper: small gray italic explanation shown
        # on its own row directly below each parameter control.
        # 'heightForWidth' on the size policy tells the grid layout to
        # ask the wrapped label how tall it actually needs to be at the
        # column's current width, instead of falling back on the
        # single-line 'sizeHint().height()' (which would clip the
        # wrapped text). Spanning the full row width keeps the
        # resulting height small (1–2 lines) and avoids the dock-
        # inflation feedback loop that an in-row narrow column would
        # otherwise cause.
        def _desc(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                'color: gray; font-style: italic; padding-left: 12px;'
            )
            lbl.setMinimumWidth(50)
            sp = lbl.sizePolicy()
            sp.setHeightForWidth(True)
            lbl.setSizePolicy(sp)
            return lbl

        # Compact width for spinboxes / combos so the description column
        # to their right gets the rest of the row.
        _SPIN_WIDTH = 95

        # ── Basic parameter grid ─────────────────────────────────────────
        # Each parameter occupies two rows: a control row (label + spin)
        # then a description row spanning the full layout width.
        basic_grid = QGridLayout()
        basic_grid.setColumnStretch(2, 1)
        basic_grid.setHorizontalSpacing(8)
        basic_grid.setVerticalSpacing(4)

        # Lambda
        self._lambda_spin = QDoubleSpinBox()
        self._lambda_spin.setRange(0.1, 1.0)
        self._lambda_spin.setSingleStep(0.05)
        self._lambda_spin.setValue(0.1)
        self._lambda_spin.setDecimals(2)
        self._lambda_spin.setMaximumWidth(_SPIN_WIDTH)
        basic_grid.addWidget(QLabel('Lambda:'), 0, 0)
        basic_grid.addWidget(self._lambda_spin, 0, 1)
        basic_grid.addWidget(
            _desc(
                'Subpixel step size. 0.1 = 10 steps/voxel; '
                'smaller is more accurate but slower.'
            ),
            1,
            0,
            1,
            3,
        )

        # Max steps
        self._max_steps_spin = QSpinBox()
        self._max_steps_spin.setRange(1, 10000)
        self._max_steps_spin.setValue(100)
        self._max_steps_spin.setMaximumWidth(_SPIN_WIDTH)
        basic_grid.addWidget(QLabel('Max steps:'), 2, 0)
        basic_grid.addWidget(self._max_steps_spin, 2, 1)
        basic_grid.addWidget(
            _desc(
                'Total morphology steps. 100 is a good starting '
                'point; larger values mean longer runtimes.'
            ),
            3,
            0,
            1,
            3,
        )

        params_layout.addLayout(basic_grid)

        # Collapsible Advanced-mode section toggle
        self._advanced_toggle = QToolButton()
        self._advanced_toggle.setText('▶  Advanced mode')
        self._advanced_toggle.setCheckable(True)
        self._advanced_toggle.setChecked(False)
        self._advanced_toggle.setStyleSheet(
            'QToolButton { border: none; font-weight: bold; }'
        )
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        params_layout.addWidget(self._advanced_toggle)

        self._advanced_widget = QWidget()
        adv_grid = QGridLayout()
        adv_grid.setContentsMargins(12, 0, 0, 0)
        adv_grid.setColumnStretch(2, 1)
        adv_grid.setHorizontalSpacing(8)
        adv_grid.setVerticalSpacing(4)
        self._advanced_widget.setLayout(adv_grid)
        # Forward heightForWidth from 'adv_grid' (whose description labels
        # have heightForWidth=True) up to the surrounding 'params_layout'.
        # Without this, the intermediate QWidget reports a single-line
        # sizeHint and the wrapped descriptions get clipped.
        sp = self._advanced_widget.sizePolicy()
        sp.setHeightForWidth(True)
        self._advanced_widget.setSizePolicy(sp)
        self._advanced_widget.hide()

        # Connectivity
        self._connectivity_combo = QComboBox()
        self._connectivity_combo.addItems(['6', '18', '26'])
        self._connectivity_combo.setCurrentIndex(2)
        self._connectivity_combo.setMaximumWidth(_SPIN_WIDTH)
        adv_grid.addWidget(QLabel('Connectivity:'), 0, 0)
        adv_grid.addWidget(self._connectivity_combo, 0, 1)
        adv_grid.addWidget(
            _desc(
                '3D neighbour connectivity. 26 (full neighbourhood) '
                'is recommended.'
            ),
            1,
            0,
            1,
            3,
        )

        # Offset
        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(0, 20)
        self._offset_spin.setValue(int(ceil(1.0 / self._lambda_spin.value())))
        self._offset_spin.setMaximumWidth(_SPIN_WIDTH)
        adv_grid.addWidget(QLabel('Offset:'), 2, 0)
        adv_grid.addWidget(self._offset_spin, 2, 1)
        adv_grid.addWidget(
            _desc(
                'Auto-set to int(1 / Lambda) (one full voxel layer); '
                'recommended to leave unchanged.'
            ),
            3,
            0,
            1,
            3,
        )

        # Argmax weighting on smoothed values — off by default (matches
        # Chenhao's original v2). Local-max identification still runs on
        # the raw curve; this flag only changes which candidate wins
        # the "tallest" argmax by ranking them via the moving-average
        # smoothed curve rather than the raw counts. Prevents a single-
        # sample noise spike from winning; has no visible effect on
        # curves that expose only one local maximum (typical erosion
        # object-count curves).
        self._rank_peaks_by_smoothed_check = QCheckBox(
            'Rank peaks by smoothed value'
        )
        self._rank_peaks_by_smoothed_check.setChecked(False)
        adv_grid.addWidget(self._rank_peaks_by_smoothed_check, 4, 0, 1, 3)
        adv_grid.addWidget(
            _desc(
                'When enabled, the tallest peak is picked by ranking '
                'candidates on the moving-average smoothed curve '
                'instead of the raw count. Prevents a single-sample '
                'noise spike from winning the argmax; has no visible '
                'effect on curves with only one clear peak.'
            ),
            5,
            0,
            1,
            3,
        )

        # Keep the offset tied to '1 / Lambda' whenever Lambda is changed.
        # Connecting only now (after the offset spinbox exists) avoids
        # firing the slot during the initial 'setValue(0.1)' above.
        self._lambda_spin.valueChanged.connect(self._on_lambda_changed)

        params_layout.addWidget(self._advanced_widget)
        main_layout.addWidget(params_group)

        # 4 ── Physical scale ───────────────────────────────────────────────
        scale_group = QGroupBox('Physical Scale')
        scale_vbox = QVBoxLayout()
        scale_group.setLayout(scale_vbox)

        scale_hint = QLabel(
            "Set before running. Leave on 'vox' to show results in voxels only."
        )
        scale_hint.setWordWrap(True)
        scale_hint.setStyleSheet(
            'color: gray; font-style: italic; padding-left: 12px;'
        )
        scale_vbox.addWidget(scale_hint)

        xyz_row = QHBoxLayout()

        self._vx_spin = QDoubleSpinBox()
        self._vy_spin = QDoubleSpinBox()
        self._vz_spin = QDoubleSpinBox()
        for spin, lbl in (
            (self._vx_spin, 'X:'),
            (self._vy_spin, 'Y:'),
            (self._vz_spin, 'Z:'),
        ):
            spin.setRange(0.0, 100000.0)
            spin.setSingleStep(0.1)
            spin.setValue(0.0)
            spin.setDecimals(2)
            spin.setSpecialValueText(
                '—'
            )  # shows "—" when value == minimum (0.0)
            spin.setEnabled(False)
            xyz_row.addWidget(QLabel(lbl))
            xyz_row.addWidget(spin)

        self._unit_combo = QComboBox()
        self._unit_combo.addItems(['vox', 'nm', 'µm'])
        self._previous_unit = 'vox'
        self._unit_combo.currentTextChanged.connect(self._on_unit_changed)
        xyz_row.addWidget(self._unit_combo)

        scale_vbox.addLayout(xyz_row)
        main_layout.addWidget(scale_group)

        # 5 ── Run button + progress + status ──────────────────────────────
        self._run_btn = QPushButton('Run Analysis')
        self._run_btn.clicked.connect(self._on_run_clicked)
        main_layout.addWidget(self._run_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat('%p%')
        self._progress_bar.setFixedHeight(18)
        main_layout.addWidget(self._progress_bar)

        self._status_label = QLabel('')
        self._status_label.setWordWrap(True)
        main_layout.addWidget(self._status_label)

        # 6 ── Embedded matplotlib plot ─────────────────────────────────────
        # Per-object selector row above the canvas: pick which object's curve
        # to show (plus an "All (overlay)" entry for ≤ 10 objects), and a
        # checkbox that highlights the picked object back in the viewer's
        # Labels layer. The whole row is hidden in aggregate mode (one curve).
        self._object_row = QWidget()
        object_row_layout = QHBoxLayout()
        object_row_layout.setContentsMargins(0, 0, 0, 0)
        self._object_row.setLayout(object_row_layout)

        object_row_layout.addWidget(QLabel('Show object:'))
        self._object_selector = QComboBox()
        self._object_selector.currentIndexChanged.connect(
            self._on_object_selected
        )
        object_row_layout.addWidget(self._object_selector)

        self._highlight_check = QCheckBox('Highlight in viewer')
        self._highlight_check.setChecked(True)
        self._highlight_check.toggled.connect(self._on_highlight_toggled)
        object_row_layout.addWidget(self._highlight_check)
        object_row_layout.addStretch()

        self._object_row.setVisible(False)
        main_layout.addWidget(self._object_row)

        self._figure = Figure(figsize=(4, 2.5), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(200)
        self._ax = self._figure.add_subplot(111)
        self._ax.text(
            0.5,
            0.5,
            'Run analysis to see the count curve',
            ha='center',
            va='center',
            transform=self._ax.transAxes,
            fontsize=9,
            color='gray',
        )
        self._ax.set_axis_off()
        main_layout.addWidget(self._canvas)

        # 7 ── Statistics ───────────────────────────────────────────────────
        stats_group = QGroupBox('Results')
        stats_form = QFormLayout()
        stats_group.setLayout(stats_form)

        # Caption naming which object the values below belong to
        # ('All combined', 'Object k of N (label i)', or a hint in overlay
        # mode). Empty until the first run.
        self._object_caption = QLabel('')
        self._object_caption.setStyleSheet('font-weight: bold;')
        stats_form.addRow(self._object_caption)

        # Row labels are kept as instance attributes because their text
        # changes between modes: the first row is the raw peak (erosion
        # 'Radius / half-thickness', dilation 'Half-spacing') and the second
        # is twice that (erosion 'Width / thickness', dilation 'Inter-object
        # spacing'). Both rows are shown in every mode; only the text differs.
        self._radius_row_label = QLabel('Radius / half-thickness (erosion):')
        self._width_row_label = QLabel('Width / thickness:')
        self._radius_label = QLabel('—')
        self._width_label = QLabel('—')
        self._fwhm_label = QLabel('—')
        stats_form.addRow(self._radius_row_label, self._radius_label)
        stats_form.addRow(self._width_row_label, self._width_label)
        stats_form.addRow('Full-width at half-maximum:', self._fwhm_label)

        main_layout.addWidget(stats_group)

        # 8 ── Save buttons ────────────────────────────────────────────────
        # Two buttons side-by-side: the first writes only the summary
        # values (mirror of the Results section), the second writes the
        # raw + smoothed count curve as CSV and the embedded plot as PNG.
        save_row = QHBoxLayout()

        self._save_btn = QPushButton('Save Measurement Results')
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_clicked)
        save_row.addWidget(self._save_btn)

        self._save_curve_btn = QPushButton('Save Count Curve Data && Figures')
        self._save_curve_btn.setEnabled(False)
        self._save_curve_btn.clicked.connect(self._on_save_curve_clicked)
        save_row.addWidget(self._save_curve_btn)

        main_layout.addLayout(save_row)

        main_layout.addStretch()

    ##########################################################################
    # Callbacks (slots)
    #
    # Every interactive control built in '_build_ui' is wired to one of the
    # methods below. They are intentionally small so the data flow remains
    # easy to trace: user input -> slot -> state change / worker dispatch.
    ##########################################################################

    def _on_mode_changed(self, index: int) -> None:
        """
        Show / hide the container-layer dropdown depending on mode.

        The container layer is only meaningful for the internal-spacing
        analysis ('persistent_homology_dilation_internal_object'), so it
        is hidden in the other two modes to keep the UI tidy.
        """
        is_internal = self._mode_combo.currentText() == _MODE_DILATION_INTERNAL
        self._container_label.setVisible(is_internal)
        self._container_combo.setVisible(is_internal)

    def _on_advanced_toggled(self, checked: bool) -> None:
        """
        Expand or collapse the Advanced parameter section.

        Also rotates the disclosure triangle character in the toggle
        button label so the visual state matches the panel state.
        """
        self._advanced_widget.setVisible(checked)
        self._advanced_toggle.setText(
            '▼  Advanced mode' if checked else '▶  Advanced mode'
        )

    def _on_lambda_changed(self, new_lambda: float) -> None:
        """
        Keep the offset value in sync with '1 / Lambda'.

        The offset masks the noisy initial segment of the count curve;
        the recommended value scales with 'Lambda' (= one full voxel
        layer). Whenever the user changes 'Lambda', the offset is reset
        to 'int(ceil(1 / Lambda))' so it stays consistent without
        requiring a manual adjustment. The user is free to override the
        offset afterwards — the next 'Lambda' change will reset it
        again.
        """
        if new_lambda > 0:
            self._offset_spin.setValue(int(ceil(1.0 / new_lambda)))

    def _on_unit_changed(self, unit: str) -> None:
        """
        Handle a change of the physical-units dropdown.

        Two effects:
        1. Enable / disable the X/Y/Z spinboxes — they are disabled in
           'vox' mode because voxel measurements need no scale.
        2. When switching from 'vox' to 'nm' or 'µm', seed any spinbox
           that is still at 0 with a default of 1.0 so the user can run
           the analysis immediately without having to set the values
           manually first.

        Switching directly between 'nm' and 'µm' leaves the spinbox
        values unchanged — the new unit is just reinterpreted as a label
        for the same numerical value.
        """
        enabled = unit != 'vox'
        prev = self._previous_unit

        # Switching from 'vox' to a physical unit: provide a sensible
        # default so the user gets a result on the first Run click.
        if prev == 'vox' and unit != 'vox':
            for spin in (self._vx_spin, self._vy_spin, self._vz_spin):
                if spin.value() == 0.0:
                    spin.setValue(1.0)

        for spin in (self._vx_spin, self._vy_spin, self._vz_spin):
            spin.setEnabled(enabled)
        self._previous_unit = unit

    def _refresh_layer_combos(self, event=None) -> None:
        """
        Repopulate the Labels-layer dropdowns from the current viewer state.

        Called once at startup and whenever a layer is added or removed.
        Tries to preserve the previous selection if the same name still
        exists, so a fresh layer add does not reset the user's choice.
        """
        labels_names = [
            layer.name
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]
        for combo in (self._seg_combo, self._container_combo):
            prev = combo.currentText()
            combo.clear()
            combo.addItems(labels_names)
            idx = combo.findText(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _reset_results(self) -> None:
        """Reset all result-displaying widgets to their empty initial state.

        Called at the very start of every Run click and whenever an error
        invalidates the previous results, so the user never sees stale
        radius / width / FWHM numbers paired with a new error message.

        Resets:
        - Radius / Width / FWHM value labels back to "—" and the object caption
        - The row labels (and the width-row visibility) to match the
          *current* mode — this is the only place result-row labels are
          allowed to change, so switching modes without clicking Run
          leaves the previous result fully intact.
        - The object selector (cleared + row hidden) and the viewer highlight
        - The matplotlib plot back to its grey placeholder
        - The cached per-object results to an empty list
        - The Save buttons to disabled
        """
        # Restore the viewer's Labels layer before dropping the run params
        # that name it.
        self._clear_highlight()

        self._object_caption.setText('')
        self._radius_label.setText('—')
        self._width_label.setText('—')
        self._fwhm_label.setText('—')
        self._results = []
        self._save_btn.setEnabled(False)
        self._save_curve_btn.setEnabled(False)

        # Clear + hide the object selector (no spurious callback while empty).
        self._object_selector.blockSignals(True)
        self._object_selector.clear()
        self._object_selector.blockSignals(False)
        self._object_row.setVisible(False)

        # Both rows are shown in every mode. The first row is always the raw
        # peak location (a half-quantity per the paper); the second row is
        # exactly twice that — the full geometric size. Erosion: half-thickness
        # / full thickness; dilation: half-spacing / full inter-object spacing.
        mode_key = _MODE_KEY[self._mode_combo.currentText()]
        if mode_key == 'erosion':
            self._radius_row_label.setText(
                'Radius / half-thickness (erosion):'
            )
            self._width_row_label.setText('Width / thickness:')
        else:
            self._radius_row_label.setText('Half-spacing:')
            self._width_row_label.setText('Inter-object spacing:')
        self._width_row_label.setVisible(True)
        self._width_label.setVisible(True)

        self._plot_message('Run analysis to see the count curve')

    def _plot_message(self, text: str) -> None:
        """Clear the plot axes and show a centred grey placeholder message."""
        self._ax.clear()
        self._ax.text(
            0.5,
            0.5,
            text,
            ha='center',
            va='center',
            transform=self._ax.transAxes,
            fontsize=9,
            color='gray',
        )
        self._ax.set_axis_off()
        self._canvas.draw()

    def _on_run_clicked(self) -> None:
        """
        Validate inputs, capture parameters, and launch the worker.

        Validation steps (in order):
        1. A segmentation layer must be selected.
        2. In internal-spacing mode, a container layer must also be selected
           and must differ from the segmentation.
        3. In nm / µm mode all three voxel sizes must be strictly positive.
        4. The segmentation must contain at least one non-zero voxel, and the
           'Label IDs' field must parse and reference labels that are present.
        5. Very large volumes (> 500^3 voxels) display a warning but still
           proceed — the computation is long but legitimate.

        After validation, the current parameter values are snapshotted into
        'self._run_params' (for later use by the CSV writer), a fresh
        '_ProgressEmitter' is created and connected, and a 'thread_worker'
        is started. The worker's 'returned' / 'errored' / 'started' /
        'finished' signals are connected to the corresponding '_on_*' slots.

        The raw (non-binarized) label data is passed to the worker so it can
        either binarize the union of the selected labels (aggregate mode) or
        crop each label into its own sub-volume (per-object mode). Layer data
        is read here on the GUI thread, as napari layer access must be.
        """
        # Always start from a clean slate so a failed Run click cannot leave
        # the user staring at stale radius / FWHM numbers from a previous
        # successful run.
        self._reset_results()

        seg_name = self._seg_combo.currentText()
        if not seg_name:
            self._set_status(
                'Error: No Labels layer found in the viewer.', error=True
            )
            return

        mode_text = self._mode_combo.currentText()
        mode_key = _MODE_KEY[mode_text]

        # Raw container data (cropped / binarized later by the worker), only
        # needed in internal-spacing mode.
        container_data = None
        if mode_key == 'dilation_internal':
            container_name = self._container_combo.currentText()
            if not container_name:
                self._set_status(
                    'Error: No container layer selected.', error=True
                )
                return
            if container_name == seg_name:
                self._set_status(
                    'Error: Segmentation and container must be different '
                    'layers.',
                    error=True,
                )
                return
            container_data = self._viewer.layers[container_name].data

        # Physical-scale validation: in nm / µm mode all three voxel
        # dimensions must be strictly positive, otherwise the voxel →
        # physical conversion is meaningless (0 nm/voxel collapses every
        # length to 0). 'vox' mode is unconstrained — the X/Y/Z values
        # are ignored entirely there.
        unit = self._unit_combo.currentText()
        if unit != 'vox':
            vx = self._vx_spin.value()
            vy = self._vy_spin.value()
            vz = self._vz_spin.value()
            if vx <= 0.0 or vy <= 0.0 or vz <= 0.0:
                self._set_status(
                    'Error: Voxel size is 0 for at least one axis. '
                    'Set X / Y / Z to non-zero values, or switch the '
                    "unit to 'vox'.",
                    error=True,
                )
                return

        # Raw label data (not binarized — the worker needs the label values to
        # extract per-object sub-volumes).
        label_data = self._viewer.layers[seg_name].data
        available = np.unique(label_data)
        available = available[available != 0]
        if available.size == 0:
            self._set_status(
                'Error: Segmentation layer is empty (all zero voxels).',
                error=True,
            )
            return

        # Internal mode crops the container to each object's bounding box, which
        # only makes sense if the two volumes are voxel-aligned. Catch a shape
        # mismatch here with a clear message instead of letting it surface as a
        # cryptic indexing/broadcast error deep inside the worker.
        if (
            container_data is not None
            and container_data.shape != label_data.shape
        ):
            self._set_status(
                f'Error: Container shape {tuple(container_data.shape)} does '
                f'not match segmentation shape {tuple(label_data.shape)}.',
                error=True,
            )
            return

        # Parse the 'Label IDs' field against the labels actually present.
        analyze_each = self._analyze_combo.currentText() == _ANALYZE_EACH
        try:
            label_ids = parse_label_ids(
                self._label_ids_edit.text(), available.tolist()
            )
        except ValueError as exc:
            self._set_status(f'Error: {exc}', error=True)
            return

        # One job per label in per-object mode; a single combined job
        # otherwise. Used to size the progress bar over N * max_steps.
        n_objects = len(label_ids) if analyze_each else 1

        if label_data.size > 500**3:
            self._set_status(
                f'Warning: Volume has {label_data.size:,} voxels — '
                'computation may take a very long time.'
            )

        Lambda = self._lambda_spin.value()
        max_steps = self._max_steps_spin.value()
        connectivity = int(self._connectivity_combo.currentText())
        offset = self._offset_spin.value()
        rank_peaks_by_smoothed = self._rank_peaks_by_smoothed_check.isChecked()
        # 'unit' was already read above for the physical-scale check

        self._run_params = {
            'mode': mode_text,
            'analyze_each': analyze_each,
            'label_ids': label_ids,
            'n_objects': n_objects,
            'seg_name': seg_name,
            'Lambda': Lambda,
            'max_steps': max_steps,
            'connectivity': connectivity,
            'offset': offset,
            'rank_peaks_by_smoothed': rank_peaks_by_smoothed,
            'unit': unit,
            'vx': self._vx_spin.value(),
            'vy': self._vy_spin.value(),
            'vz': self._vz_spin.value(),
        }

        # Thread-safe progress emitter — created fresh each run
        self._progress_emitter = _ProgressEmitter()
        self._progress_emitter.progress_changed.connect(self._on_progress)

        worker = _run_analysis(
            label_data,
            container_data,
            mode_key,
            analyze_each,
            label_ids,
            Lambda,
            max_steps,
            connectivity,
            offset,
            rank_peaks_by_smoothed,
            self._progress_emitter.step,
        )
        worker.returned.connect(self._on_result)
        worker.errored.connect(self._on_error)
        worker.started.connect(self._on_started)
        worker.finished.connect(self._on_finished)
        worker.start()

    def _set_status(self, text: str, *, error: bool = False) -> None:
        """Update the status label; render in bold red when 'error' is True.

        Using 'setStyleSheet' (rather than HTML-rich text) means the message
        string is rendered verbatim — no risk of stray '<' / '&' in an
        exception message being interpreted as markup.
        """
        if error:
            self._status_label.setStyleSheet(
                'color: #d63333; font-weight: bold;'
            )
        else:
            self._status_label.setStyleSheet('')
        self._status_label.setText(text)

    def _on_started(self) -> None:
        """Disable the Run button and reset the progress bar.

        The range spans the whole run: 'n_objects * max_steps' steps, so the
        single bar fills smoothly across a per-object sweep of N objects.
        """
        self._run_btn.setEnabled(False)
        total = self._run_params['n_objects'] * self._run_params['max_steps']
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(0)
        self._set_status('Computing…')

    def _on_progress(self, step: int, total: int) -> None:
        """Advance the progress bar; called once per morphology step."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(step)

    def _on_finished(self) -> None:
        """
        Re-enable the Run button and snap the bar to 100%.

        Snapping is a safety measure for the case where the final
        'step_callback' arrives slightly after the worker's 'finished'
        signal — without it the bar might be left at 99%.
        """
        self._run_btn.setEnabled(True)
        self._progress_bar.setValue(self._progress_bar.maximum())

    def _on_error(self, exc: Exception) -> None:
        """Display the worker exception in the status label.

        Capitalises the first character of the exception message so the
        text after 'Error: ' starts with a capital letter, matching the
        other status messages.
        """
        msg = str(exc)
        if msg:
            msg = msg[0].upper() + msg[1:]
        self._set_status(f'Error: {msg}', error=True)

    @staticmethod
    def _mode_display(mode_key: str) -> dict:
        """Return the mode-dependent display strings for a run.

        The peak label names the raw-peak quantity (a half-quantity per the
        paper):
          erosion             → object-count curve, peak ≈ radius
          dilation            → hole-count curve, peak ≈ half-spacing
          dilation_internal   → hole-count curve, peak ≈ half-spacing
        """
        if mode_key == 'erosion':
            return {
                'analysis_name': 'Object radius / half-thickness',
                'peak_label': 'Max / radius',
                'x_axis_label': 'Erosion round',
                'y_axis_label': 'Object count',
            }
        if mode_key == 'dilation':
            return {
                'analysis_name': 'Object spacing',
                'peak_label': 'Max / half-spacing',
                'x_axis_label': 'Dilation round',
                'y_axis_label': 'Hole count',
            }
        return {
            'analysis_name': 'Internal spacing',
            'peak_label': 'Max / half-spacing',
            'x_axis_label': 'Dilation round',
            'y_axis_label': 'Hole count',
        }

    def _process_object(self, obj: dict, p: dict) -> dict:
        """Turn one raw worker result into a display record.

        Adds the smoothed curve, the peak step index, an 'ok' flag (False for
        degenerate curves), and the pre-formatted radius / width / FWHM
        strings. Voxel → physical conversion uses the arithmetic mean of the
        X/Y/Z voxel pitches when a non-'vox' unit is selected and all three
        voxel sizes are set.
        """
        series = obj['series']
        radius_vox = obj['radius_vox']
        fwhm_vox = obj['fwhm_vox']

        # For plot rendering: the smoothed curve mirrors what v2 uses
        # internally when computing FWHM (moving average over one full
        # voxel round). The peak step index is derived from 'radius_vox'
        # by dividing by 'Lambda' — v2 returned it multiplied by Lambda
        # to get voxel units.
        smooth_window = max(1, int(round(1.0 / p['Lambda'])))
        smoothed = moving_average(series, w=smooth_window)
        max_location_step = int(round(radius_vox / p['Lambda']))

        # Degenerate curve: v2 falls back to '0' on empty / flat curves.
        # Detect either that fallback or a peak step that lies outside a
        # meaningful part of the curve — such an object is flagged 'not ok'
        # (shown as "—" / "no peak") instead of displaying misleading zeros.
        ok = not (
            (radius_vox == 0.0 and fwhm_vox == 0.0)
            or max_location_step >= len(smoothed)
            or smoothed[max_location_step] == 0
        )

        # The second result row is always twice the raw peak. In erosion that
        # is the full diameter / thickness; in dilation it is the full
        # inter-object spacing (the peak itself is a half-distance — see the
        # paper's Fig. 3 caption and the Persistent Homology methods section).
        width_vox = 2.0 * radius_vox
        unit = p['unit']

        if not ok:
            radius_str = width_str = fwhm_str = '—'
        elif unit != 'vox' and min(p['vx'], p['vy'], p['vz']) > 0:
            # Physical units are the primary display; the voxel value
            # appears in brackets afterwards. Arithmetic mean of voxel
            # dimensions — appropriate for length quantities. Assumes
            # near-isotropic voxels; results are approximate for strongly
            # anisotropic data.
            mean_vox = (p['vx'] + p['vy'] + p['vz']) / 3.0
            radius_str = (
                f'{radius_vox * mean_vox:.2f} {unit}  ({radius_vox:.2f} vox)'
            )
            width_str = (
                f'{width_vox * mean_vox:.2f} {unit}  ({width_vox:.2f} vox)'
            )
            fwhm_str = (
                f'{fwhm_vox * mean_vox:.2f} {unit}  ({fwhm_vox:.2f} vox)'
            )
        else:
            # Either unit == 'vox' or the voxel pitch is partially unset:
            # show voxel-only results without brackets.
            radius_str = f'{radius_vox:.2f} vox'
            width_str = f'{width_vox:.2f} vox'
            fwhm_str = f'{fwhm_vox:.2f} vox'

        return {
            'label_id': obj['label_id'],
            'series': series,
            'smoothed': smoothed,
            'radius_vox': radius_vox,
            'fwhm_vox': fwhm_vox,
            'width_vox': width_vox,
            'max_location_step': max_location_step,
            'ok': ok,
            'radius_str': radius_str,
            'width_str': width_str,
            'fwhm_str': fwhm_str,
        }

    def _on_result(self, result: dict) -> None:
        """
        Process the worker result, populate the object selector, show object 0.

        The worker returns one raw record per analysed object (one in
        aggregate mode, N in per-object mode). Each is formatted via
        '_process_object'. If *every* object is degenerate the whole run is
        reported as 'No peak detected'; otherwise the object selector is
        populated and the first object is displayed.
        """
        p = self._run_params
        mode_key = result['mode_key']
        self._curve_label = result['curve_label']

        records = [self._process_object(obj, p) for obj in result['objects']]
        self._results = records

        # All objects degenerate → nothing meaningful to show.
        if not any(r['ok'] for r in records):
            self._reset_results()
            self._set_status(
                'Error: No peak detected — count curve is empty.',
                error=True,
            )
            return

        disp = self._mode_display(mode_key)
        self._set_status(f'{disp["analysis_name"]} analysis completed.')

        self._populate_object_selector(records, p['analyze_each'])
        if p['analyze_each']:
            # setCurrentIndex fires _on_object_selected, which draws object 0.
            self._object_selector.blockSignals(True)
            self._object_selector.setCurrentIndex(0)
            self._object_selector.blockSignals(False)
            self._on_object_selected(0)
        else:
            self._show_combined(records[0])

        self._save_btn.setEnabled(True)
        self._save_curve_btn.setEnabled(True)

    def _populate_object_selector(
        self, records: list, analyze_each: bool
    ) -> None:
        """Fill the 'Show object' combo, or hide the row in aggregate mode.

        Each object entry ('Object <label>') carries its index as user-data;
        two overlay entries ('overlay_smoothed' / 'overlay_raw') are added
        only for 2..'_MAX_OVERLAY' objects — beyond that the overlay is too
        cluttered to read.
        """
        combo = self._object_selector
        combo.blockSignals(True)
        combo.clear()
        if analyze_each:
            for i, rec in enumerate(records):
                suffix = '' if rec['ok'] else '  — no peak'
                combo.addItem(f'Object {rec["label_id"]}{suffix}', i)
            if 1 < len(records) <= _MAX_OVERLAY:
                combo.addItem('All (overlay – smoothed)', 'overlay_smoothed')
                combo.addItem('All (overlay – raw)', 'overlay_raw')
            self._object_row.setVisible(True)
        else:
            self._object_row.setVisible(False)
        combo.blockSignals(False)

    def _on_object_selected(self, index: int) -> None:
        """Show the selected object's curve + metrics, or the overlay.

        Slot for the 'Show object' combo. Reads the current entry's user-data
        rather than the raw 'index' argument so the 'All (overlay)' entry is
        recognised regardless of its position.
        """
        if not self._results:
            return
        data = self._object_selector.currentData()
        if data in ('overlay_smoothed', 'overlay_raw'):
            self._clear_highlight()
            self._object_caption.setText(
                'All objects — select one for metrics'
            )
            self._radius_label.setText('—')
            self._width_label.setText('—')
            self._fwhm_label.setText('—')
            self._update_plot_overlay(
                self._results,
                kind='raw' if data == 'overlay_raw' else 'smoothed',
            )
            return

        rec = self._results[int(data)]
        self._object_caption.setText(f'Object {rec["label_id"]}')
        self._show_object_record(rec)
        self._apply_highlight(rec['label_id'])

    def _show_combined(self, rec: dict) -> None:
        """Display the single aggregate ('All combined') result record."""
        self._object_caption.setText('All combined')
        self._show_object_record(rec)
        self._clear_highlight()  # nothing single to highlight in aggregate mode

    def _show_object_record(self, rec: dict) -> None:
        """Update the three metric rows + plot for one object record."""
        self._radius_label.setText(rec['radius_str'])
        self._width_label.setText(rec['width_str'])
        self._fwhm_label.setText(rec['fwhm_str'])
        if not rec['ok']:
            self._plot_message('No peak detected for this object.')
            return
        self._update_plot(rec)

    # ── Viewer highlight ────────────────────────────────────────────────────

    def _seg_layer(self):
        """Return the run's segmentation Labels layer, or None if unavailable.

        The layer may have been renamed or removed since the run started, so
        every highlight operation goes through this guarded lookup.
        """
        seg_name = self._run_params.get('seg_name')
        if not seg_name or seg_name not in self._viewer.layers:
            return None
        layer = self._viewer.layers[seg_name]
        if not isinstance(layer, napari.layers.Labels):
            return None
        return layer

    def _apply_highlight(self, label_id) -> None:
        """Highlight one label in the viewer via napari's native mechanism.

        Sets 'selected_label' + 'show_selected_label' on the segmentation
        Labels layer so only the picked object is shown — the same
        regionprops-style isolation napari offers natively. No-op when the
        highlight checkbox is off, the label is the aggregate 'None', or the
        layer is gone.
        """
        layer = self._seg_layer()
        if layer is None:
            return
        if not self._highlight_check.isChecked() or label_id is None:
            layer.show_selected_label = False
            return
        layer.selected_label = int(label_id)
        layer.show_selected_label = True

    def _clear_highlight(self) -> None:
        """Restore the full Labels view (turn off single-label isolation)."""
        layer = self._seg_layer()
        if layer is not None:
            layer.show_selected_label = False

    def _on_highlight_toggled(self, checked: bool) -> None:
        """Re-apply or clear the highlight when the checkbox is toggled."""
        if not checked:
            self._clear_highlight()
            return
        # Re-apply for the currently-selected single object, if any.
        # isHidden (not isVisible) so this works before the dock is shown.
        if self._results and not self._object_row.isHidden():
            data = self._object_selector.currentData()
            if data is not None and data not in (
                'overlay_smoothed',
                'overlay_raw',
            ):
                rec = self._results[int(data)]
                self._apply_highlight(rec['label_id'])

    def _object_colors(self, records: list) -> list:
        """Return one plot colour per object, matched to the Labels layer.

        Uses the colour napari assigns each label ('layer.get_color') so the
        overlay curves match the segmentation Labels layer (and the
        single-label highlight). Falls back to the default matplotlib colour
        cycle when the layer is gone or a colour can't be read.

        The viewer highlight ('show_selected_label') isolates one label by
        making 'get_color' return a *transparent* colour for every other
        label. When that isolation is active (e.g. an individual object was
        selected before the user clicked Save), reading colours here would
        come back transparent for all but the highlighted object, so the
        saved overlay dropped every other curve. We therefore lift the
        isolation for the duration of the batch read and restore it after,
        so the colours always reflect the true per-label palette regardless
        of the current highlight state. (The on-screen overlay already clears
        the highlight before drawing, so this toggle is a no-op there.)
        """
        layer = self._seg_layer()
        fallback = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        restore_highlight = layer is not None and getattr(
            layer, 'show_selected_label', False
        )
        if restore_highlight:
            layer.show_selected_label = False
        try:
            colors = []
            for i, rec in enumerate(records):
                color = fallback[i % len(fallback)]
                lid = rec['label_id']
                if layer is not None and lid is not None:
                    try:
                        rgba = layer.get_color(int(lid))
                    except Exception:  # noqa: BLE001 - defensive: 3rd-party API
                        rgba = None
                    if rgba is not None:
                        color = tuple(float(c) for c in rgba)
                colors.append(color)
        finally:
            if restore_highlight:
                layer.show_selected_label = True
        return colors

    def _draw_single_on_ax(self, ax, rec: dict, disp: dict) -> None:
        """Draw one object's raw + smoothed curve, peak line and FWHM bar on 'ax'.

        Shared by the on-screen plot ('_update_plot') and the save-all-plots
        renderer, so a saved PNG matches exactly what the widget shows. The
        x-axis is the morphology-step index, not voxels.

        Shows the raw count curve (light) and its moving-average-smoothed
        version (dark), a dashed vertical line at the detected peak (labelled
        as radius or spacing depending on mode), and a dashed horizontal bar
        at half-peak height spanning the FWHM.
        """
        series = rec['series']
        smoothed = rec['smoothed']
        max_location_step = rec['max_location_step']
        x = list(range(len(series)))
        ax.plot(
            x,
            series,
            color='lightsteelblue',
            linewidth=1.0,
            alpha=0.85,
            label=f'{self._curve_label} (raw data)',
        )
        ax.plot(
            x,
            smoothed,
            color='steelblue',
            linewidth=1.5,
            label=f'{self._curve_label} (smoothed)',
        )
        ax.axvline(
            max_location_step,
            color='crimson',
            linestyle='--',
            linewidth=1.0,
            label=f'{disp["peak_label"]} = {rec["radius_vox"]:.2f} vox',
        )

        # FWHM bar: dashed horizontal line at half-peak height, spanning
        # the contiguous block of smoothed samples that lie above half-max
        # and contain the peak. Same dash style as the peak line for
        # visual consistency.
        peak_value = float(smoothed[max_location_step])
        half_max = peak_value / 2.0
        above = smoothed >= half_max
        left = max_location_step
        while left > 0 and above[left - 1]:
            left -= 1
        right = max_location_step
        while right < len(smoothed) - 1 and above[right + 1]:
            right += 1
        ax.hlines(
            half_max,
            xmin=left,
            xmax=right,
            color='darkorange',
            linestyle='--',
            linewidth=1.5,
            label=f'FWHM = {rec["fwhm_vox"]:.2f} vox',
        )

        ax.set_xlabel(disp['x_axis_label'], fontsize=8)
        ax.set_ylabel(disp['y_axis_label'], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.set_axis_on()

    def _draw_overlay_on_ax(
        self, ax, records: list, kind: str, disp: dict
    ) -> None:
        """Draw the per-object overlay ('smoothed' or 'raw') on 'ax'.

        Each object is drawn in the colour napari assigns its label (see
        '_object_colors'), with a faint dashed peak marker in the same colour;
        degenerate objects are labelled '(no peak)' and drawn without a marker.
        """
        colors = self._object_colors(records)
        key = 'series' if kind == 'raw' else 'smoothed'
        for rec, color in zip(records, colors, strict=True):
            suffix = '' if rec['ok'] else ' (no peak)'
            ax.plot(
                range(len(rec[key])),
                rec[key],
                color=color,
                linewidth=1.3,
                label=f'Object {rec["label_id"]}{suffix}',
            )
            if rec['ok']:
                ax.axvline(
                    rec['max_location_step'],
                    color=color,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.6,
                )
        ax.set_xlabel(disp['x_axis_label'], fontsize=8)
        ax.set_ylabel(f'{disp["y_axis_label"]} ({kind})', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.set_axis_on()

    def _update_plot(self, rec: dict) -> None:
        """Redraw the embedded on-screen plot for one object record."""
        disp = self._mode_display(_MODE_KEY[self._run_params['mode']])
        self._ax.clear()
        self._draw_single_on_ax(self._ax, rec, disp)
        self._figure.tight_layout()
        self._canvas.draw()

    def _update_plot_overlay(
        self, records: list, kind: str = 'smoothed'
    ) -> None:
        """Redraw the embedded on-screen plot as a per-object overlay.

        'kind' selects the smoothed ('smoothed') or raw ('raw') curve — the
        selector offers a separate overlay entry for each. Only reached for
        2..'_MAX_OVERLAY' objects.
        """
        disp = self._mode_display(_MODE_KEY[self._run_params['mode']])
        self._ax.clear()
        self._draw_overlay_on_ax(self._ax, records, kind, disp)
        self._figure.tight_layout()
        self._canvas.draw()

    def _save_all_plots(self, base: str) -> list:
        """Render and save every count-curve plot as PNG(s).

        Single-object runs (aggregate mode, or a per-object run with one
        object) save one PNG directly next to the CSV ('<base>.png').
        Multi-object per-object runs collect all their PNGs in a subfolder
        named exactly after the CSV stem ('<base>/'), i.e. whatever the user
        typed (or accepted from the structured default) in the save dialog —
        nothing extra is appended, so the folder name stays as short as the
        chosen filename. Inside, files are named by what the curve counts:
        'object_count_obj_<label>.png' (erosion) or 'hole_count_obj_<label>.png'
        (dilation) per non-degenerate object, plus
        '<prefix>_overlay_smoothed.png' / '<prefix>_overlay_raw.png' when
        N <= _MAX_OVERLAY. Each plot is rendered on a throwaway Figure so the
        on-screen canvas is left untouched. Returns the list of written paths.
        """
        mode_key = _MODE_KEY[self._run_params['mode']]
        disp = self._mode_display(mode_key)
        prefix = _COUNT_FILE_PREFIX[mode_key]
        saved: list = []

        def _render_and_save(path: str, draw) -> None:
            fig = Figure(figsize=(4, 2.5), tight_layout=True)
            ax = fig.add_subplot(111)
            draw(ax)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            saved.append(path)

        # Single object → one PNG directly in the chosen directory.
        if len(self._results) <= 1:
            rec = self._results[0]
            if rec['ok']:
                _render_and_save(
                    f'{base}.png',
                    lambda ax: self._draw_single_on_ax(ax, rec, disp),
                )
            return saved

        # Multiple objects → a subfolder named exactly like the CSV stem (the
        # deduplicated base), so it never overwrites an earlier run's folder.
        plot_dir = base
        os.makedirs(plot_dir, exist_ok=True)
        for rec in self._results:
            if not rec['ok']:
                continue  # no curve to draw for a degenerate object
            _render_and_save(
                os.path.join(plot_dir, f'{prefix}_obj_{rec["label_id"]}.png'),
                lambda ax, rec=rec: self._draw_single_on_ax(ax, rec, disp),
            )
        if len(self._results) <= _MAX_OVERLAY:
            for kind in ('smoothed', 'raw'):
                _render_and_save(
                    os.path.join(plot_dir, f'{prefix}_overlay_{kind}.png'),
                    lambda ax, kind=kind: self._draw_overlay_on_ax(
                        ax, self._results, kind, disp
                    ),
                )
        return saved

    def _default_save_name(self, kind: str) -> str:
        """Build a structured default filename to prefill a Save dialog.

        'kind' is the trailing token — 'measurements' or 'count_curve_data'.
        The name encodes the analysis mode and the combined/per-object choice
        so runs get distinctive names out of the box, e.g.
        'Object_radius_per_object_measurements.csv' or
        'Object_spacing_combined_objects_count_curve_data.csv'. The user may
        edit it freely; the chosen stem is reused verbatim for the CSV, the
        single-object PNG and the multi-object plot subfolder.
        """
        p = self._run_params
        mode_token = _MODE_NAME_TOKEN[_MODE_KEY[p['mode']]]
        analyze_token = _ANALYZE_NAME_TOKEN[p['analyze_each']]
        return f'{mode_token}_{analyze_token}_{kind}.csv'

    @staticmethod
    def _strip_save_ext(path: str) -> str:
        """Strip a trailing '.csv' or '.png' (case-insensitive) from a chosen
        save path, returning the bare stem.

        Both save handlers share this so the CSV, the single-object PNG and the
        plot folder always derive from one stem regardless of which of the two
        extensions the user happened to type.
        """
        for ext in ('.csv', '.png'):
            if path.lower().endswith(ext):
                return path[: -len(ext)]
        return path

    @staticmethod
    def _dedup_base(base: str, suffixes) -> str:
        """Return 'base' (or 'base_2', 'base_3', …) for which every
        'base + suffix' is free on disk, so a save never clobbers an earlier one.

        'suffixes' lists what this save will create for the stem: e.g.
        ['.csv', '.png'] for a single-object curve save, or ['.csv', ''] for a
        multi-object save (the empty suffix is the plot folder itself).
        """
        candidate = base
        n = 2
        while any(os.path.exists(candidate + s) for s in suffixes):
            candidate = f'{base}_{n}'
            n += 1
        return candidate

    def _write_metadata_header(
        self, w, p, *, note_voxel_size_when_unset: bool = True
    ) -> bool:
        """Write the shared comment header (mode + parameters + voxel size)
        to the CSV writer 'w'. Returns True if a physical voxel size was
        set (so the caller can include a physical-value column).

        When voxel size is not set:
        - With 'note_voxel_size_when_unset=True' (the summary file) a
          '# Voxel size: not set ...' note is written.
        - With 'note_voxel_size_when_unset=False' (the curve file) the
          voxel-size line is omitted entirely — it adds nothing because
          the count curve is in step / count units regardless.
        """
        unit = p['unit']
        w.writerow(['#'])  # visual spacer below the title line
        w.writerow([f'# Mode: {p["mode"]}'])
        w.writerow(
            [
                f'# Parameters: lambda={p["Lambda"]}, '
                f'max_steps={p["max_steps"]}, '
                f'connectivity={p["connectivity"]}, '
                f'offset={p["offset"]}, '
                f'rank_peaks_by_smoothed={p["rank_peaks_by_smoothed"]}, '
                f'analyze={"each" if p["analyze_each"] else "combined"}, '
                f'label_ids={p["label_ids"]}'
            ]
        )
        if unit != 'vox' and min(p['vx'], p['vy'], p['vz']) > 0:
            w.writerow(
                [
                    f'# Voxel size: x={p["vx"]} {unit}, '
                    f'y={p["vy"]} {unit}, z={p["vz"]} {unit}'
                ]
            )
            return True
        if note_voxel_size_when_unset:
            w.writerow(['# Voxel size: not set (results in voxels only)'])
        return False

    @staticmethod
    def _metric_bases(mode_key: str) -> list:
        """Column-name bases for the three summary metrics, per mode.

        Mirror the GUI row labels but drop the '(erosion)' qualifier (the
        mode is already on the '# Mode:' header line) and use CSV-friendly
        underscores. Both modes report the raw peak (a half-quantity) and
        twice that (the full geometric size), matching the two GUI rows.
        """
        if mode_key == 'erosion':
            return ['Radius_half_thickness', 'Width_thickness', 'FWHM']
        return ['Half_spacing', 'Inter_object_spacing', 'FWHM']

    def _on_save_clicked(self) -> None:
        """Save the summary results — exactly what the Results section shows.

        Writes one row per analysed object (a single 'all' row in aggregate
        mode, N rows in per-object mode). File contents:
        - Comment header: mode, parameters (incl. analyze mode + label IDs),
          voxel-size info.
        - Stats table: 'Label_ID' plus the three mode-appropriate metrics in
          voxels and (if a physical voxel size is set) in the chosen physical
          unit. Objects with no detected peak are written as 'NaN'.

        The dialog is prefilled with a structured default name
        ('<Mode>_<analyze>_measurements.csv' — see '_default_save_name'); the
        user can edit it. A '.csv' extension is appended automatically if the
        user does not supply one, and if that file already exists the name is
        auto-incremented ('_2', '_3', …) rather than overwritten. Does nothing
        if no successful run has been performed yet ('self._results' empty).
        """
        if not self._results:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Measurement Results',
            self._default_save_name('measurements'),
            'CSV files (*.csv)',
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not path:
            return
        base = self._strip_save_ext(path)
        # Never overwrite an existing summary CSV — bump to '_2', '_3', ….
        path = self._dedup_base(base, ['.csv']) + '.csv'

        p = self._run_params
        unit = p['unit']
        bases = self._metric_bases(_MODE_KEY[p['mode']])

        # utf-8-sig writes a BOM so Excel on Windows auto-detects UTF-8 and
        # renders the em dash / 'µm' correctly instead of mojibake ('â€”').
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['# napari-persistent-homology — Summary Results'])
            has_physical = self._write_metadata_header(w, p)
            w.writerow([])

            header = ['Label_ID'] + [f'{b}_vox' for b in bases]
            if has_physical:
                # Arithmetic mean — appropriate for length quantities.
                mean_vox = (p['vx'] + p['vy'] + p['vz']) / 3.0
                # ASCII 'um' instead of 'µm' is friendlier to downstream
                # tooling; the unit is baked into each column name.
                column_unit = {'nm': 'nm', 'µm': 'um'}.get(unit, unit)
                header += [f'{b}_{column_unit}' for b in bases]
            w.writerow(header)

            for rec in self._results:
                # 'all' names the aggregate job (label_id is None there).
                label = 'all' if rec['label_id'] is None else rec['label_id']
                vox_vals = [
                    rec['radius_vox'],
                    rec['width_vox'],
                    rec['fwhm_vox'],
                ]
                if not rec['ok']:
                    row = [label] + ['NaN'] * len(bases)
                    if has_physical:
                        row += ['NaN'] * len(bases)
                    w.writerow(row)
                    continue
                row = [label] + [f'{v:.4f}' for v in vox_vals]
                if has_physical:
                    row += [f'{v * mean_vox:.4f}' for v in vox_vals]
                w.writerow(row)

        self._set_status(f'Results saved to {path}')

    def _on_save_curve_clicked(self) -> None:
        """Save the raw + smoothed count curve(s) as CSV *and* every count
        curve plot as PNG(s), all derived from one user-picked path.

        - If the user picks 'foo.csv' (or any path), the CSV is written to
          'foo.csv' and the plots alongside it. The existing extension is
          stripped first so all siblings share a base.
        - The CSV contains the same metadata header as the summary file
          (mode + parameters + voxel size) plus a tidy/long table with the
          columns 'Label_ID', 'Erosion_round' / 'Dilation_round',
          'Count_raw', 'Count_smoothed' — one block of rows per object
          ('all' in aggregate mode).
        - The PNGs are *all* count curve plots, not just the one on screen:
          in per-object mode they go in a subfolder named exactly like the CSV
          stem ('foo/') — one per object ('object_count_obj_<label>.png' /
          'hole_count_obj_<label>.png') plus both overlays
          ('<prefix>_overlay_smoothed.png' / '<prefix>_overlay_raw.png') when
          applicable; in aggregate mode the single plot ('foo.png').
          See '_save_all_plots'.

        The dialog is prefilled with a structured default name
        ('<Mode>_<analyze>_count_curve_data.csv'); the user can edit it, and if
        the resulting CSV / PNG / plot folder already exists the stem is
        auto-incremented ('_2', '_3', …) rather than overwritten.

        Does nothing if no successful run has been performed yet.
        """
        if not self._results:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Count Curve Data & Figures',
            self._default_save_name('count_curve_data'),
            'CSV files (*.csv)',
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not path:
            return

        # Derive the shared base from whatever the user typed (CSV + PNGs).
        base = self._strip_save_ext(path)
        # Auto-increment ('_2', '_3', …) so a save never clobbers an earlier
        # one. A single-object save also writes '<base>.png'; a multi-object
        # save writes a '<base>/' plot folder — both must be free, not just
        # the CSV, so the stem stays consistent across all outputs of one save.
        extra = '.png' if len(self._results) <= 1 else ''
        base = self._dedup_base(base, ['.csv', extra])
        csv_path = base + '.csv'

        p = self._run_params

        # The x-axis column name reflects what each row means in the
        # active analysis mode (erosion vs dilation rounds), and the
        # title spells out which quantity is being counted (object vs
        # hole) — so the file is self-explanatory without needing the
        # reader to cross-reference the '# Mode:' line.
        mode_key = _MODE_KEY[p['mode']]
        if mode_key == 'erosion':
            curve_title = '# napari-persistent-homology — Object Count Curve'
            round_column = 'Erosion_round'
        else:
            curve_title = '# napari-persistent-homology — Hole Count Curve'
            round_column = 'Dilation_round'

        # utf-8-sig (BOM) so Excel on Windows decodes UTF-8 correctly.
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([curve_title])
            self._write_metadata_header(w, p, note_voxel_size_when_unset=False)
            w.writerow([])
            w.writerow(
                ['Label_ID', round_column, 'Count_raw', 'Count_smoothed']
            )
            for rec in self._results:
                label = 'all' if rec['label_id'] is None else rec['label_id']
                # series and smoothed are always the same length (smoothing
                # preserves it), so strict=True documents that invariant.
                for i, (raw, sm) in enumerate(
                    zip(rec['series'], rec['smoothed'], strict=True)
                ):
                    w.writerow([label, i, int(raw), f'{sm:.4f}'])

        # Save every count curve plot as a PNG alongside the CSV.
        png_paths = self._save_all_plots(base)

        self._set_status(
            f'Curve saved to {csv_path}; {len(png_paths)} plot PNG(s) saved.'
        )
