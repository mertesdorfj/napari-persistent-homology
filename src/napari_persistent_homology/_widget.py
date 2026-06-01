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
from math import ceil

import napari.layers
import numpy as np
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
    compute_homology_stats,
    gaussian_average,
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


@thread_worker
def _run_analysis(
    volume: np.ndarray,
    container_vol,
    mode_key: str,
    Lambda: float,
    max_steps: int,
    connectivity: int,
    SIGMA: float,
    offset: int,
    step_callback,
):
    """
    Run the selected persistent-homology analysis on a background thread.

    Dispatches to one of the three analysis functions in 'ph_functions.py'
    based on 'mode_key', picks the relevant count curve (object-count for
    erosion, hole-count for the two dilation variants), and reduces it to
    two scalar shape descriptors using 'compute_homology_stats'.

    Parameters
    ----------
    volume: ndarray
        Binary 3D segmentation to analyse.
    container_vol: ndarray or None
        Container mask, required when 'mode_key == "dilation_internal"',
        ignored otherwise.
    mode_key: str
        One of 'erosion', 'dilation', 'dilation_internal'.
    Lambda, max_steps, connectivity:
        Forwarded to the underlying persistent-homology function. See
        'ph_functions.persistent_homology_*' for details.
    SIGMA, offset:
        Forwarded to 'compute_homology_stats' for smoothing and peak finding.
    step_callback: callable
        Per-step progress reporter, normally
        '_ProgressEmitter.step'.

    Returns
    -------
    tuple
        '(series, radius_vox, fwhm_vox, curve_label, SIGMA)' where 'series'
        is the raw count curve, 'radius_vox' / 'fwhm_vox' are the peak
        location and full-width-at-half-maximum converted from subpixel-step
        units to voxel units (divided by 'ceil(1 / Lambda)'), and
        'curve_label' is a string for the plot legend.
    """
    from napari.utils import progress

    with progress(total=0):  # indeterminate spinner in napari activity bar
        if mode_key == 'erosion':
            obj_counts, hole_counts = persistent_homology_erosion(
                volume,
                max_steps=max_steps,
                Lambda=Lambda,
                Connectivity=connectivity,
                step_callback=step_callback,
            )
            series = obj_counts
            curve_label = 'Object count'
        elif mode_key == 'dilation':
            obj_counts, hole_counts = persistent_homology_dilation(
                volume,
                max_steps=max_steps,
                Lambda=Lambda,
                Connectivity=connectivity,
                step_callback=step_callback,
            )
            series = hole_counts
            curve_label = 'Hole count'
        else:  # dilation_internal
            obj_counts, hole_counts = (
                persistent_homology_dilation_internal_object(
                    volume,
                    container_vol,
                    max_steps=max_steps,
                    Lambda=Lambda,
                    Connectivity=connectivity,
                    step_callback=step_callback,
                )
            )
            series = hole_counts
            curve_label = 'Hole count (internal)'

    stats = compute_homology_stats([series], offset=offset, SIGMA=SIGMA)
    # stats shape: (3, 1)  ->  [FWHM, max_count, max_location]
    scale = int(ceil(1.0 / Lambda))
    radius_vox = float(stats[2, 0]) / scale
    fwhm_vox = float(stats[0, 0]) / scale
    return series, radius_vox, fwhm_vox, curve_label, SIGMA


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
    - 'self._last_result': cached result tuple from the most recent run,
      used by the "Save Results…" button. 'None' until the first run
      completes successfully.
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
        self._last_result = None  # set after a successful run, enables Save
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

        # Sigma
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.1, 20.0)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setValue(3.0)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setMaximumWidth(_SPIN_WIDTH)
        adv_grid.addWidget(QLabel('Sigma:'), 2, 0)
        adv_grid.addWidget(self._sigma_spin, 2, 1)
        adv_grid.addWidget(
            _desc(
                'Gaussian smoothing of the count curve before peak detection.'
            ),
            3,
            0,
            1,
            3,
        )

        # Offset
        self._offset_spin = QSpinBox()
        self._offset_spin.setRange(0, 20)
        self._offset_spin.setValue(int(ceil(1.0 / self._lambda_spin.value())))
        self._offset_spin.setMaximumWidth(_SPIN_WIDTH)
        adv_grid.addWidget(QLabel('Offset:'), 4, 0)
        adv_grid.addWidget(self._offset_spin, 4, 1)
        adv_grid.addWidget(
            _desc(
                'Auto-set to int(1 / Lambda) (one full voxel layer); '
                'recommended to leave unchanged.'
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

        self._save_btn = QPushButton('Save Results')
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_clicked)
        save_row.addWidget(self._save_btn)

        self._save_curve_btn = QPushButton('Save Curve && Plot')
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
        - Radius / Width / FWHM value labels back to "—"
        - The row labels (and the width-row visibility) to match the
          *current* mode — this is the only place result-row labels are
          allowed to change, so switching modes without clicking Run
          leaves the previous result fully intact.
        - The matplotlib plot back to its grey placeholder
        - The cached '_last_result' tuple to None
        - The Save Results button to disabled
        """
        self._radius_label.setText('—')
        self._width_label.setText('—')
        self._fwhm_label.setText('—')
        self._last_result = None
        self._save_btn.setEnabled(False)
        self._save_curve_btn.setEnabled(False)

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

        self._ax.clear()
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
        self._canvas.draw()

    def _on_run_clicked(self) -> None:
        """
        Validate inputs, capture parameters, and launch the worker.

        Validation steps (in order):
        1. A segmentation layer must be selected.
        2. In internal-spacing mode, a container layer must also be selected.
        3. The segmentation must contain at least one non-zero voxel.
        4. Very large volumes (> 500^3 voxels) display a warning but still
           proceed — the computation is long but legitimate.

        After validation, the current parameter values are snapshotted into
        'self._run_params' (for later use by the CSV writer), a fresh
        '_ProgressEmitter' is created and connected, and a 'thread_worker'
        is started. The worker's 'returned' / 'errored' / 'started' /
        'finished' signals are connected to the corresponding '_on_*' slots.
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

        container_vol = None
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
            container_vol = (
                self._viewer.layers[container_name].data > 0
            ).astype(np.uint8)

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

        volume = (self._viewer.layers[seg_name].data > 0).astype(np.uint8)

        if volume.sum() == 0:
            self._set_status(
                'Error: Segmentation layer is empty (all zero voxels).',
                error=True,
            )
            return

        if volume.size > 500**3:
            self._set_status(
                f'Warning: Volume has {volume.size:,} voxels — '
                'computation may take a very long time.'
            )

        Lambda = self._lambda_spin.value()
        max_steps = self._max_steps_spin.value()
        connectivity = int(self._connectivity_combo.currentText())
        SIGMA = self._sigma_spin.value()
        offset = self._offset_spin.value()
        # 'unit' was already read above for the physical-scale check

        self._run_params = {
            'mode': mode_text,
            'Lambda': Lambda,
            'max_steps': max_steps,
            'connectivity': connectivity,
            'SIGMA': SIGMA,
            'offset': offset,
            'unit': unit,
            'vx': self._vx_spin.value(),
            'vy': self._vy_spin.value(),
            'vz': self._vz_spin.value(),
        }

        # Thread-safe progress emitter — created fresh each run
        self._progress_emitter = _ProgressEmitter()
        self._progress_emitter.progress_changed.connect(self._on_progress)

        worker = _run_analysis(
            volume,
            container_vol,
            mode_key,
            Lambda,
            max_steps,
            connectivity,
            SIGMA,
            offset,
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
        """Disable the Run button and reset the progress bar to 0 / max_steps."""
        self._run_btn.setEnabled(False)
        self._progress_bar.setRange(0, self._run_params['max_steps'])
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

    def _on_result(self, result: tuple) -> None:
        """
        Format the result, update the result labels, refresh the plot.

        Converts the radius / FWHM voxel values to physical units via the
        arithmetic mean of the X/Y/Z voxel pitches when a non-'vox' unit
        is selected and all three voxel sizes are set. If the voxel size
        is unset (still at the '—' placeholder, i.e. 0.0), the physical
        value is shown as '—' as well. The smoothed curve, peak step
        location, and the result tuple are cached on 'self._last_result'
        for later use by 'Save Results…'.

        Degenerate input (count curve never rises above zero — typically
        when the algorithm finds no measurable signal) is detected here
        and reported as 'No peak detected — count curve is empty' rather
        than displaying the misleading offset-fallback values that
        'find_max_location' produces on a flat curve.
        """
        series, radius_vox, fwhm_vox, curve_label, SIGMA = result

        p = self._run_params

        # Degenerate curve: 'find_max_location' falls back to returning the
        # offset value when the post-offset series is all zero, and
        # 'compute_FWHM' returns the full curve width on a flat curve. The
        # reported radius / FWHM are therefore not real measurements — show
        # a clear error and clear the result widgets instead.
        scale = int(ceil(1.0 / p['Lambda']))
        max_location_step = int(round(radius_vox * scale))
        smoothed = gaussian_average(series, sigma=SIGMA)
        if (
            max_location_step >= len(smoothed)
            or smoothed[max_location_step] == 0
        ):
            self._reset_results()
            self._set_status(
                'Error: No peak detected — count curve is empty.',
                error=True,
            )
            return

        unit = p['unit']

        # The second result row is always twice the raw peak. In erosion that
        # is the full diameter / thickness; in dilation it is the full
        # inter-object spacing (the peak itself is a half-distance — see the
        # paper's Fig. 3 caption and the Persistent Homology methods section).
        width_vox = 2.0 * radius_vox

        if unit != 'vox' and min(p['vx'], p['vy'], p['vz']) > 0:
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

        self._radius_label.setText(radius_str)
        self._width_label.setText(width_str)
        self._fwhm_label.setText(fwhm_str)

        # Mode-dependent labels — used both for the completion status
        # message and for the plot's axis / peak annotations. The peak label
        # names the raw-peak quantity (a half-quantity per the paper).
        #   erosion             → object-count curve, peak ≈ radius
        #   dilation            → hole-count curve, peak ≈ half-spacing
        #   dilation_internal   → hole-count curve, peak ≈ half-spacing
        mode_key = _MODE_KEY[p['mode']]
        if mode_key == 'erosion':
            analysis_name = 'Object radius / half-thickness'
            peak_label = 'Max / radius'
            x_axis_label = 'Erosion round'
            y_axis_label = 'Object count'
        elif mode_key == 'dilation':
            analysis_name = 'Object spacing'
            peak_label = 'Max / half-spacing'
            x_axis_label = 'Dilation round'
            y_axis_label = 'Hole count'
        else:  # "dilation_internal"
            analysis_name = 'Internal spacing'
            peak_label = 'Max / half-spacing'
            x_axis_label = 'Dilation round'
            y_axis_label = 'Hole count'

        self._set_status(f'{analysis_name} analysis completed.')

        # 'scale', 'max_location_step', and 'smoothed' were already computed
        # above for the degenerate-curve check; reuse them here.
        self._last_result = (
            series,
            smoothed,
            radius_vox,
            fwhm_vox,
            curve_label,
            max_location_step,
        )

        self._update_plot(
            series,
            smoothed,
            curve_label,
            max_location_step,
            fwhm_vox,
            radius_vox,
            peak_label,
            x_axis_label,
            y_axis_label,
        )
        self._save_btn.setEnabled(True)
        self._save_curve_btn.setEnabled(True)

    def _update_plot(
        self,
        series: np.ndarray,
        smoothed: np.ndarray,
        curve_label: str,
        max_location_step: int,
        fwhm_vox: float,
        radius_vox: float,
        peak_label: str,
        x_axis_label: str,
        y_axis_label: str,
    ) -> None:
        """
        Redraw the embedded matplotlib plot for the latest run.

        Shows the raw count curve (light) and its Gaussian-smoothed version
        (dark), a dashed vertical line at the detected peak (labelled as
        radius or spacing depending on mode), and a dashed horizontal bar
        at half-peak height spanning the FWHM. The x-axis is the
        morphology-step index, not voxels.
        """
        self._ax.clear()
        x = list(range(len(series)))
        self._ax.plot(
            x,
            series,
            color='lightsteelblue',
            linewidth=1.0,
            alpha=0.85,
            label=f'{curve_label} (raw data)',
        )
        self._ax.plot(
            x,
            smoothed,
            color='steelblue',
            linewidth=1.5,
            label=f'{curve_label} (smoothed)',
        )
        self._ax.axvline(
            max_location_step,
            color='crimson',
            linestyle='--',
            linewidth=1.0,
            label=f'{peak_label} = {radius_vox:.2f} vox',
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
        self._ax.hlines(
            half_max,
            xmin=left,
            xmax=right,
            color='darkorange',
            linestyle='--',
            linewidth=1.5,
            label=f'FWHM = {fwhm_vox:.2f} vox',
        )

        self._ax.set_xlabel(x_axis_label, fontsize=8)
        self._ax.set_ylabel(y_axis_label, fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._ax.legend(fontsize=7)
        self._ax.set_axis_on()
        self._figure.tight_layout()
        self._canvas.draw()

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
                f'sigma={p["SIGMA"]}, '
                f'offset={p["offset"]}'
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

    def _on_save_clicked(self) -> None:
        """Save only the summary results — i.e. exactly what appears in
        the Results section of the widget (radius, width, FWHM).

        File contents:
        - Comment header: mode, parameters, voxel-size info.
        - Stats table: 'radius', 'width', 'fwhm' with their values in
          voxels and (if a physical voxel size is set) in the chosen
          physical unit.

        A '.csv' extension is appended automatically if the user does
        not supply one. Does nothing if no successful run has been
        performed yet ('self._last_result is None').
        """
        if self._last_result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Results', '', 'CSV files (*.csv)'
        )
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'

        _, _, radius_vox, fwhm_vox, _, _ = self._last_result
        p = self._run_params
        unit = p['unit']
        width_vox = 2.0 * radius_vox

        # Metric labels mirror the GUI row labels but drop the trailing
        # '(erosion)' qualifier — the mode is already on the '# Mode:'
        # line in the header, so repeating it here would be redundant.
        # Both modes report the raw peak (a half-quantity) and twice that
        # (the full geometric size), matching the two GUI result rows.
        mode_key = _MODE_KEY[p['mode']]
        if mode_key == 'erosion':
            metrics = [
                ('Radius / half-thickness', radius_vox),
                ('Width / thickness', width_vox),
                ('Full-width at half-maximum', fwhm_vox),
            ]
        else:
            metrics = [
                ('Half-spacing', radius_vox),
                ('Inter-object spacing', width_vox),
                ('Full-width at half-maximum', fwhm_vox),
            ]

        # utf-8-sig writes a BOM so Excel on Windows auto-detects UTF-8 and
        # renders the em dash / 'µm' correctly instead of mojibake ('â€”').
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['# napari-persistent-homology — Summary Results'])
            has_physical = self._write_metadata_header(w, p)
            w.writerow([])

            if has_physical:
                # Arithmetic mean — appropriate for length quantities.
                mean_vox = (p['vx'] + p['vy'] + p['vz']) / 3.0
                # Column name bakes the unit in (e.g. 'Value_nm'), so we
                # drop the separate 'unit' column that used to repeat
                # the same string on every row. ASCII 'um' instead of
                # 'µm' is friendlier to downstream tooling.
                column_unit = {'nm': 'nm', 'µm': 'um'}.get(unit, unit)
                w.writerow(
                    [
                        'Metric',
                        'Value_vox',
                        f'Value_{column_unit}',
                    ]
                )
                for label, vox in metrics:
                    w.writerow(
                        [
                            label,
                            f'{vox:.4f}',
                            f'{vox * mean_vox:.4f}',
                        ]
                    )
            else:
                w.writerow(['Metric', 'Value_vox'])
                for label, vox in metrics:
                    w.writerow([label, f'{vox:.4f}'])

        self._set_status(f'Results saved to {path}')

    def _on_save_curve_clicked(self) -> None:
        """Save the raw + smoothed count curve as CSV *and* the embedded
        matplotlib plot as a PNG, both derived from one user-picked path.

        - If the user picks 'foo.csv' (or any path), the CSV is written
          to 'foo.csv' and the figure to 'foo.png'. The existing
          extension is stripped first so the two siblings share a base.
        - The CSV contains the same metadata header as the summary file
          (mode + parameters + voxel size) plus three columns:
          'Erosion_round' / 'Dilation_round', 'Count_raw', 'Count_smoothed'.
        - The PNG is the current matplotlib figure rendered at 150 DPI
          with tight bounding box.

        Does nothing if no successful run has been performed yet.
        """
        if self._last_result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Curve & Plot', '', 'CSV files (*.csv)'
        )
        if not path:
            return

        # Derive paired CSV / PNG paths from whatever the user typed.
        base = path
        for ext in ('.csv', '.png'):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
                break
        csv_path = base + '.csv'
        png_path = base + '.png'

        series, smoothed, _, _, _, _ = self._last_result
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
            w.writerow([round_column, 'Count_raw', 'Count_smoothed'])
            # series and smoothed are always the same length (smoothing
            # preserves it), so strict=True documents that invariant.
            for i, (raw, sm) in enumerate(zip(series, smoothed, strict=True)):
                w.writerow([i, int(raw), f'{sm:.4f}'])

        # Save the current plot as a PNG. 'bbox_inches="tight"' trims the
        # surrounding whitespace so the legend isn't clipped.
        self._figure.savefig(png_path, dpi=150, bbox_inches='tight')

        self._set_status(f'Curve saved to {csv_path}, plot to {png_path}')
