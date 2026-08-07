import numpy as np
import pytest

from napari_persistent_homology._widget import (
    _CURVE_LABEL,
    _MAX_OVERLAY,
    _MODE_DILATION,
    _MODE_DILATION_INTERNAL,
    _MODE_EROSION,
    _MODE_KEY,
    PersistentHomologyWidget,
    _make_step_callback,
    _run_analysis,
    parse_label_ids,
)


def test_widget_creates(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    assert isinstance(widget, PersistentHomologyWidget)


def test_layer_combo_populated(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # No layers yet — combo is empty
    assert widget._seg_combo.count() == 0

    # Add a Labels layer
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[2:8, 2:8, 2:8] = 1
    viewer.add_labels(data, name='test_labels')

    assert widget._seg_combo.count() == 1
    assert widget._seg_combo.itemText(0) == 'test_labels'


def test_image_layers_not_in_combo(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    viewer.add_image(np.random.random((10, 10, 10)), name='raw_image')

    # Image layers should NOT appear in the Labels combo
    assert widget._seg_combo.count() == 0


def test_mode_change_hides_container(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Default mode is Erosion — container row explicitly hidden
    assert widget._container_combo.isHidden()
    assert widget._container_label.isHidden()

    # Switch to Internal spacing — container row no longer hidden
    internal_idx = widget._mode_combo.findText(_MODE_DILATION_INTERNAL)
    widget._mode_combo.setCurrentIndex(internal_idx)
    assert not widget._container_combo.isHidden()
    assert not widget._container_label.isHidden()

    # Switch back to Erosion — hidden again
    erosion_idx = widget._mode_combo.findText(_MODE_EROSION)
    widget._mode_combo.setCurrentIndex(erosion_idx)
    assert widget._container_combo.isHidden()


def _make_run_params(
    mode_text=_MODE_EROSION,
    *,
    unit='vox',
    vx=0.0,
    vy=0.0,
    vz=0.0,
    analyze_each=False,
    label_ids=(1,),
    n_objects=1,
    seg_name='seg',
):
    """Build a '_run_params' snapshot with all keys _on_run_clicked sets."""
    return {
        'mode': mode_text,
        'analyze_each': analyze_each,
        'label_ids': list(label_ids),
        'n_objects': n_objects,
        'seg_name': seg_name,
        'Lambda': 0.1,
        'max_steps': 100,
        'connectivity': 26,
        'offset': 5,
        'rank_peaks_by_smoothed': False,
        'unit': unit,
        'vx': vx,
        'vy': vy,
        'vz': vz,
    }


def _plateau_series(peak=5.0, length=101):
    """A non-degenerate count curve: zeros with a plateau around the peak."""
    series = np.zeros(length)
    series[40:60] = peak
    return series


def _fake_worker_result(mode_text=_MODE_EROSION, *, objects=None):
    """Build the dict the worker returns (one entry per analysed object)."""
    mode_key = _MODE_KEY[mode_text]
    if objects is None:
        objects = [
            {
                'label_id': None,
                'series': _plateau_series(),
                'radius_vox': 5.0,
                'fwhm_vox': 2.0,
            }
        ]
    analyze_each = any(o['label_id'] is not None for o in objects)
    return {
        'analyze_each': analyze_each,
        'mode_key': mode_key,
        'curve_label': _CURVE_LABEL[mode_key],
        'objects': objects,
    }


def _fake_completed_run(widget, mode_text=None, *, objects=None):
    """Helper: drive widget._on_result with a non-degenerate fake result
    so the two save buttons are enabled and self._results is populated as
    if a real analysis had completed.

    'objects' (a list of raw per-object dicts) drives per-object runs; the
    default is a single aggregate object.
    """
    if mode_text is None:
        mode_text = _MODE_EROSION
    result = _fake_worker_result(mode_text, objects=objects)
    widget._run_params = _make_run_params(
        mode_text,
        analyze_each=result['analyze_each'],
        label_ids=[
            o['label_id']
            for o in result['objects']
            if o['label_id'] is not None
        ]
        or (1,),
        n_objects=len(result['objects']),
    )
    widget._on_result(result)


def _read_csv_rows(path):
    """Read a CSV file written by the widget and return all non-comment
    data rows as lists of strings. Comment lines starting with '#' and
    blank rows are skipped."""
    import csv as _csv

    rows = []
    # utf-8-sig strips the BOM the widget now writes (so Excel decodes it).
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in _csv.reader(f):
            if not row or (row[0].startswith('#')):
                continue
            rows.append(row)
    return rows


def test_save_results_writes_summary_csv_erosion(
    make_napari_viewer,
    tmp_path,
    monkeypatch,
):
    """Save Results in erosion mode writes three metrics. Labels drop
    the '(erosion)' qualifier (already in the '# Mode:' header)."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, mode_text=_MODE_EROSION)

    target = tmp_path / 'summary.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_args, **_kwargs: (str(target), 'CSV files (*.csv)'),
    )

    widget._on_save_clicked()

    assert target.is_file()
    rows = _read_csv_rows(target)
    # Wide table: Label_ID + the three erosion metrics in voxels.
    assert rows[0] == [
        'Label_ID',
        'Radius_half_thickness_vox',
        'Width_thickness_vox',
        'FWHM_vox',
    ]
    # A single aggregate row, labelled 'all'.
    assert len(rows) == 2
    assert rows[1][0] == 'all'
    # Width is exactly twice the radius.
    assert float(rows[1][2]) == pytest.approx(2.0 * float(rows[1][1]))
    # The curve-x-axis columns must NOT appear in the summary
    assert not any(
        r[:1] in (['Erosion_round'], ['Dilation_round']) for r in rows
    )


def test_save_results_writes_summary_csv_dilation(
    make_napari_viewer,
    tmp_path,
    monkeypatch,
):
    """Save Results in dilation mode writes THREE metrics: the raw peak
    ('Half-spacing'), twice that ('Inter-object spacing'), and FWHM —
    mirroring the two GUI result rows plus FWHM."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, mode_text=_MODE_DILATION)

    target = tmp_path / 'summary.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_args, **_kwargs: (str(target), 'CSV files (*.csv)'),
    )

    widget._on_save_clicked()

    rows = _read_csv_rows(target)
    # Wide table: Label_ID + dilation metrics (half-spacing, inter-object, FWHM)
    assert rows[0] == [
        'Label_ID',
        'Half_spacing_vox',
        'Inter_object_spacing_vox',
        'FWHM_vox',
    ]
    assert len(rows) == 2
    assert rows[1][0] == 'all'
    # Inter-object spacing must be exactly twice the half-spacing.
    assert float(rows[1][2]) == pytest.approx(2.0 * float(rows[1][1]))


def test_save_curve_writes_csv_and_png(
    make_napari_viewer, tmp_path, monkeypatch
):
    """The Save Curve & Plot button writes BOTH the count-curve CSV and
    a PNG of the matplotlib figure, derived from one user-picked path."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget)

    # User picks "run1.csv" — code should derive "run1.png" alongside it.
    target = tmp_path / 'run1.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_args, **_kwargs: (str(target), 'CSV files (*.csv)'),
    )

    widget._on_save_curve_clicked()

    csv_path = tmp_path / 'run1.csv'
    png_path = tmp_path / 'run1.png'
    assert csv_path.is_file()
    assert png_path.is_file()
    assert png_path.stat().st_size > 0

    rows = _read_csv_rows(csv_path)
    # Erosion mode → the round column is named 'Erosion_round'; the tidy
    # table leads with a 'Label_ID' column ('all' in aggregate mode).
    assert rows[0] == [
        'Label_ID',
        'Erosion_round',
        'Count_raw',
        'Count_smoothed',
    ]
    assert len(rows) == 102  # 1 header + 101 data rows (rounds 0..100)
    assert all(r[0] == 'all' for r in rows[1:])
    # The summary metric column header must NOT appear in the curve file
    assert not any(r[:1] == ['Metric'] for r in rows)


def test_save_buttons_disabled_initially(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    assert not widget._save_btn.isEnabled()
    assert not widget._save_curve_btn.isEnabled()


def test_advanced_section_collapsed_initially(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    assert widget._advanced_widget.isHidden()
    assert not widget._advanced_toggle.isChecked()


def test_parameter_defaults_and_lambda_offset_linkage(make_napari_viewer):
    """Default Lambda is 0.1 with a hard minimum of 0.1, Sigma defaults to
    3.0, and the offset tracks 'ceil(1 / Lambda)' — including after
    changing Lambda in the widget."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Lambda: default 0.1 and the spinbox refuses values < 0.1.
    assert widget._lambda_spin.value() == 0.1
    assert widget._lambda_spin.minimum() == 0.1

    # Sigma spinbox was removed when the analysis switched to v2 (which
    # smooths internally with a Lambda-derived moving average).
    assert not hasattr(widget, '_sigma_spin')

    # Offset: default = ceil(1 / Lambda) = 10 at the default Lambda.
    assert widget._offset_spin.value() == 10

    # Changing Lambda must auto-update offset to ceil(1 / new_lambda).
    widget._lambda_spin.setValue(0.5)
    assert widget._offset_spin.value() == 2

    widget._lambda_spin.setValue(0.2)
    assert widget._offset_spin.value() == 5


def test_advanced_section_expands_on_toggle(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    widget._advanced_toggle.setChecked(True)
    assert not widget._advanced_widget.isHidden()

    widget._advanced_toggle.setChecked(False)
    assert widget._advanced_widget.isHidden()


def test_unit_combo_enables_xyz_spinboxes(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Default: "vox" — all three voxel size spinboxes disabled
    assert widget._unit_combo.currentText() == 'vox'
    for spin in (widget._vx_spin, widget._vy_spin, widget._vz_spin):
        assert not spin.isEnabled()

    # Switch to "nm" — all three enabled
    widget._unit_combo.setCurrentText('nm')
    for spin in (widget._vx_spin, widget._vy_spin, widget._vz_spin):
        assert spin.isEnabled()

    # Switch back to "vox" — all three disabled again
    widget._unit_combo.setCurrentText('vox')
    for spin in (widget._vx_spin, widget._vy_spin, widget._vz_spin):
        assert not spin.isEnabled()


def test_run_button_disabled_on_empty_viewer(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Clicking Run with no layer should show an error, not crash
    widget._on_run_clicked()
    assert 'Error' in widget._status_label.text()


def test_progress_bar_resets_on_run_started(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Simulate _on_started after a run_params snapshot. The bar spans the
    # whole run: n_objects * max_steps.
    widget._run_params = {'max_steps': 50, 'n_objects': 3}
    widget._on_started()

    assert widget._progress_bar.value() == 0
    assert widget._progress_bar.maximum() == 150
    assert not widget._run_btn.isEnabled()


def test_progress_bar_updates_via_on_progress(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    widget._on_progress(37, 100)

    assert widget._progress_bar.maximum() == 100
    assert widget._progress_bar.value() == 37


def test_mode_switch_does_not_alter_results_section(make_napari_viewer):
    """Switching modes after an analysis must NOT update the result row
    labels, values, or the width-row visibility. The previous result
    stays fully intact (labelled with the mode it was computed in) until
    the user clicks Run Analysis again."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Simulate state after a successful erosion analysis
    widget._radius_label.setText('5.00 vox')
    widget._width_label.setText('10.00 vox')
    widget._fwhm_label.setText('2.00 vox')
    widget._save_btn.setEnabled(True)

    initial_row_text = widget._radius_row_label.text()
    assert 'Radius' in initial_row_text  # erosion-style label
    assert not widget._width_row_label.isHidden()

    # Switch to dilation: everything in the result section must stay
    widget._mode_combo.setCurrentText(_MODE_DILATION)
    assert widget._radius_row_label.text() == initial_row_text
    assert widget._radius_label.text() == '5.00 vox'
    assert widget._width_label.text() == '10.00 vox'
    assert widget._fwhm_label.text() == '2.00 vox'
    assert not widget._width_row_label.isHidden()  # width row NOT hidden
    assert not widget._width_label.isHidden()

    # Switch to internal-spacing: same — no relabelling
    widget._mode_combo.setCurrentText(_MODE_DILATION_INTERNAL)
    assert widget._radius_row_label.text() == initial_row_text
    assert widget._radius_label.text() == '5.00 vox'

    # Switch back to erosion: still no change — frozen until Run is clicked
    widget._mode_combo.setCurrentText(_MODE_EROSION)
    assert widget._radius_row_label.text() == initial_row_text
    assert widget._radius_label.text() == '5.00 vox'


def test_reset_results_applies_current_mode_labels(make_napari_viewer):
    """'_reset_results' (called at the start of every Run click) is the
    one place the result row labels and width-row visibility are allowed
    to update. After a mode switch, calling Run / _reset_results must
    propagate the new mode's terminology into the result section."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Default mode (erosion): labels start in erosion-style
    assert 'Radius' in widget._radius_row_label.text()
    assert '(erosion)' in widget._radius_row_label.text()
    assert not widget._width_row_label.isHidden()

    # Switch to dilation: row labels STILL unchanged (locked in until Run)
    widget._mode_combo.setCurrentText(_MODE_DILATION)
    assert 'Radius' in widget._radius_row_label.text()

    # Trigger _reset_results (as the start of a Run click would): now the
    # new-mode labels take effect. Dilation shows the raw peak as
    # 'Half-spacing' and the doubled value as 'Inter-object spacing'.
    widget._reset_results()
    assert widget._radius_row_label.text() == 'Half-spacing:'
    assert widget._width_row_label.text() == 'Inter-object spacing:'
    assert not widget._width_row_label.isHidden()
    assert not widget._width_label.isHidden()

    # Switch back to erosion and reset again — labels restore
    widget._mode_combo.setCurrentText(_MODE_EROSION)
    widget._reset_results()
    assert 'Radius' in widget._radius_row_label.text()
    assert '(erosion)' in widget._radius_row_label.text()
    assert not widget._width_row_label.isHidden()
    assert not widget._width_label.isHidden()


def test_dilation_run_shows_half_and_full_spacing_in_gui(make_napari_viewer):
    """End-to-end GUI check: a completed dilation run shows the raw peak as
    'Half-spacing' and exactly twice that as 'Inter-object spacing', with
    both rows visible."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Set dilation mode, then run _reset_results (as a Run click would) so
    # the row labels switch to the dilation terminology.
    widget._mode_combo.setCurrentText(_MODE_DILATION)
    widget._reset_results()
    assert widget._radius_row_label.text() == 'Half-spacing:'
    assert widget._width_row_label.text() == 'Inter-object spacing:'
    assert not widget._width_row_label.isHidden()
    assert not widget._width_label.isHidden()

    # Drive a completed run (raw peak = 5.0 vox) and check both values.
    _fake_completed_run(widget, mode_text=_MODE_DILATION)
    assert widget._radius_label.text() == '5.00 vox'  # raw peak (half)
    assert widget._width_label.text() == '10.00 vox'  # 2x = full spacing


def test_run_with_same_layer_for_seg_and_container_shows_error(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    # Add a single Labels layer that will appear in both dropdowns
    viewer.add_labels(np.ones((10, 10, 10), dtype=np.uint8), name='x')

    # Switch to internal-spacing mode and select the same layer in both
    widget._mode_combo.setCurrentText(_MODE_DILATION_INTERNAL)
    widget._seg_combo.setCurrentText('x')
    widget._container_combo.setCurrentText('x')

    widget._on_run_clicked()

    text = widget._status_label.text()
    assert 'Error' in text
    assert 'different layers' in text


def test_run_with_mismatched_container_shape_shows_error(make_napari_viewer):
    """Internal-spacing mode crops the container to each object's bounding box,
    so a container whose shape differs from the segmentation must be rejected
    up front with a clear message rather than failing deep in the worker."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    viewer.add_labels(np.ones((10, 10, 10), dtype=np.uint8), name='seg')
    # A container of a different shape than the segmentation.
    viewer.add_labels(np.ones((8, 10, 10), dtype=np.uint8), name='cont')

    widget._mode_combo.setCurrentText(_MODE_DILATION_INTERNAL)
    widget._seg_combo.setCurrentText('seg')
    widget._container_combo.setCurrentText('cont')

    widget._on_run_clicked()

    text = widget._status_label.text()
    assert 'Error' in text
    assert 'does not match' in text


def test_run_with_zero_voxel_size_in_physical_unit_shows_error(
    make_napari_viewer,
):
    """In 'nm' or 'µm' mode, any of X / Y / Z being 0 must trigger an
    error and block the analysis — a zero voxel pitch would collapse
    every physical length to 0. Each axis is checked independently and
    the validation must fire for both 'nm' and 'µm'."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # A valid Labels layer so the seg-layer validation passes
    viewer.add_labels(np.ones((10, 10, 10), dtype=np.uint8), name='seg')
    widget._seg_combo.setCurrentText('seg')

    # Try each axis in each physical-unit mode. Validation must fire in
    # all six combinations.
    spinboxes = {
        'X': widget._vx_spin,
        'Y': widget._vy_spin,
        'Z': widget._vz_spin,
    }
    for unit_label in ('nm', 'µm'):
        widget._unit_combo.setCurrentText(unit_label)
        for axis, spin in spinboxes.items():
            # Reset all spinboxes to a valid 1.0 baseline
            for s in spinboxes.values():
                s.setValue(1.0)
            # Zero out only the axis under test
            spin.setValue(0.0)

            widget._on_run_clicked()

            text = widget._status_label.text()
            assert 'Error' in text, (
                f'No error fired with {unit_label} mode and {axis}=0; '
                f'status: {text!r}'
            )
            assert 'Voxel size' in text
            # Worker must never have started — Run button still enabled.
            assert widget._run_btn.isEnabled(), (
                f'Run button was disabled with {unit_label} {axis}=0 — '
                'worker should have been blocked by validation'
            )


def test_reset_results_clears_labels_and_disables_save(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Pretend a previous successful run left some state behind
    widget._radius_label.setText('5.00 vox')
    widget._width_label.setText('10.00 vox')
    widget._fwhm_label.setText('3.00 vox')
    widget._save_btn.setEnabled(True)
    widget._save_curve_btn.setEnabled(True)
    widget._results = ['placeholder']

    widget._reset_results()

    assert widget._radius_label.text() == '—'
    assert widget._width_label.text() == '—'
    assert widget._fwhm_label.text() == '—'
    assert widget._save_btn.isEnabled() is False
    assert widget._save_curve_btn.isEnabled() is False
    assert widget._results == []


def test_run_click_clears_stale_results_before_validation(make_napari_viewer):
    """Even when the click fails validation, previous result labels must
    be wiped — otherwise the user sees the error paired with stale numbers
    from the previous successful run."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Simulate state from a successful prior run
    widget._radius_label.setText('5.00 vox')
    widget._width_label.setText('10.00 vox')
    widget._fwhm_label.setText('3.00 vox')
    widget._save_btn.setEnabled(True)
    widget._save_curve_btn.setEnabled(True)
    widget._results = ['placeholder']

    # No Labels layer in the viewer → validation will fail on the first
    # check inside _on_run_clicked
    widget._on_run_clicked()

    # The validation error must fire AND the stale labels must be gone
    assert 'Error' in widget._status_label.text()
    assert widget._radius_label.text() == '—'
    assert widget._width_label.text() == '—'
    assert widget._fwhm_label.text() == '—'
    assert widget._save_btn.isEnabled() is False
    assert widget._save_curve_btn.isEnabled() is False
    assert widget._results == []


def test_switching_to_physical_unit_sets_default_voxel_size(
    make_napari_viewer,
):
    """When the user switches from 'vox' to 'nm' or 'µm', the three voxel-
    size spinboxes should be pre-filled with 1.0 so the user can run the
    analysis immediately without having to set them manually first."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Initial state: all voxel sizes at 0 (the '—' placeholder)
    for spin in (widget._vx_spin, widget._vy_spin, widget._vz_spin):
        assert spin.value() == 0.0

    widget._unit_combo.setCurrentText('nm')

    for spin in (widget._vx_spin, widget._vy_spin, widget._vz_spin):
        assert spin.value() == 1.0


def test_switching_to_physical_unit_preserves_existing_values(
    make_napari_viewer,
):
    """If the user has already set non-zero voxel sizes, switching unit
    away and back must NOT clobber them with the default 1.0."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    widget._unit_combo.setCurrentText('nm')
    widget._vx_spin.setValue(5.0)
    widget._vy_spin.setValue(5.0)
    widget._vz_spin.setValue(5.0)

    # Toggle to 'vox' and back to 'nm'
    widget._unit_combo.setCurrentText('vox')
    widget._unit_combo.setCurrentText('nm')

    # Values should still be 5.0, not reset to 1.0
    assert widget._vx_spin.value() == 5.0
    assert widget._vy_spin.value() == 5.0
    assert widget._vz_spin.value() == 5.0


def test_result_label_shows_physical_first_voxels_in_brackets(
    make_napari_viewer,
):
    """With a physical unit selected and voxel sizes set, the result label
    should lead with the physical value and show voxels in parentheses."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    widget._run_params = _make_run_params(
        _MODE_EROSION, unit='nm', vx=5.0, vy=5.0, vz=5.0
    )
    # Synthetic non-degenerate count curve so the result path runs
    widget._on_result(_fake_worker_result(_MODE_EROSION))

    # Physical value is reported first, voxel value in brackets
    # 5.0 vox × ((5+5+5)/3 = 5 nm/vox) = 25.00 nm
    assert 'nm' in widget._radius_label.text()
    assert '(5.00 vox)' in widget._radius_label.text()
    assert widget._radius_label.text().index(
        'nm'
    ) < widget._radius_label.text().index('vox')


def test_result_label_voxel_only_when_unit_is_vox(make_napari_viewer):
    """With unit = 'vox', the result label should be voxels only — no
    brackets, no physical-unit suffix."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    widget._run_params = _make_run_params(_MODE_EROSION, unit='vox')
    widget._on_result(_fake_worker_result(_MODE_EROSION))

    assert widget._radius_label.text() == '5.00 vox'
    assert '(' not in widget._radius_label.text()


def test_completion_status_includes_analysis_mode(make_napari_viewer):
    """After a successful run the status message must name the specific
    analysis mode that finished — generic 'Analysis completed.' is not
    enough to keep the three modes distinguishable in the UI."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    cases = [
        (_MODE_EROSION, 'Object radius / half-thickness'),
        (_MODE_DILATION, 'Object spacing'),
        (_MODE_DILATION_INTERNAL, 'Internal spacing'),
    ]
    for mode_text, expected_phrase in cases:
        widget._run_params = _make_run_params(mode_text)
        widget._on_result(_fake_worker_result(mode_text))
        text = widget._status_label.text()
        assert expected_phrase in text, (
            f'Status for {mode_text!r} should include {expected_phrase!r}, '
            f'got {text!r}'
        )
        assert 'completed' in text.lower()


def test_degenerate_curve_shows_no_peak_error(make_napari_viewer):
    """A worker result with an all-zero count curve must trigger the
    'No peak detected' error and reset every result widget — not display
    the misleading offset-fallback values (radius=0.5, FWHM=10)."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # Set the run-params snapshot the way _on_run_clicked would have done.
    widget._run_params = _make_run_params(_MODE_EROSION)

    # Simulate a degenerate worker output: all-zero count curve, with the
    # offset-fallback radius (0.5) and full-curve FWHM (10.0) that
    # find_max_location / compute_FWHM produce on a flat curve.
    degenerate = [
        {
            'label_id': None,
            'series': np.zeros(101),
            'radius_vox': 0.5,
            'fwhm_vox': 10.0,
        }
    ]
    widget._on_result(_fake_worker_result(_MODE_EROSION, objects=degenerate))

    text = widget._status_label.text()
    assert 'Error' in text
    assert 'No peak detected' in text
    # All result widgets reset to their empty initial state
    assert widget._radius_label.text() == '—'
    assert widget._width_label.text() == '—'
    assert widget._fwhm_label.text() == '—'
    assert widget._save_btn.isEnabled() is False
    assert widget._save_curve_btn.isEnabled() is False
    assert widget._results == []


# ---------------------------------------------------------------------------
# parse_label_ids
# ---------------------------------------------------------------------------


class TestParseLabelIds:
    def test_all_returns_every_available_sorted(self):
        assert parse_label_ids('all', [3, 1, 2]) == [1, 2, 3]

    def test_empty_string_means_all(self):
        assert parse_label_ids('', [2, 1]) == [1, 2]

    def test_case_insensitive_all(self):
        assert parse_label_ids('ALL', [1, 2]) == [1, 2]

    def test_comma_list(self):
        assert parse_label_ids('1,3', [1, 2, 3]) == [1, 3]

    def test_bracketed_list(self):
        assert parse_label_ids('[1, 3]', [1, 2, 3]) == [1, 3]

    def test_space_separated(self):
        assert parse_label_ids('2 3', [1, 2, 3]) == [2, 3]

    def test_dedup_and_sort(self):
        assert parse_label_ids('3,1,3', [1, 2, 3]) == [1, 3]

    def test_junk_raises(self):
        with pytest.raises(ValueError, match='Could not parse'):
            parse_label_ids('abc', [1, 2, 3])

    def test_missing_label_raises(self):
        with pytest.raises(ValueError, match='not present'):
            parse_label_ids('1,9', [1, 2, 3])


# ---------------------------------------------------------------------------
# Per-object analysis
# ---------------------------------------------------------------------------


def _per_object_objects(n, peak=5.0):
    """Build n per-object worker records (labels 1..n).

    Each object's radius is '5.0 + i' vox, and its count-curve plateau is
    centred on the matching peak step ('round(radius / 0.1)') so the object
    is genuinely non-degenerate (the earlier objects at least — objects whose
    peak step exceeds the 101-sample curve stay flat / degenerate, which the
    large-N tests rely on).
    """
    objects = []
    for i in range(n):
        radius = 5.0 + i
        step = int(round(radius / 0.1))  # peak step for this radius
        series = np.zeros(101)
        series[max(0, step - 10) : min(101, step + 10)] = peak + i
        objects.append(
            {
                'label_id': i + 1,
                'series': series,
                'radius_vox': radius,
                'fwhm_vox': 2.0,
            }
        )
    return objects


def test_per_object_run_populates_selector(make_napari_viewer):
    """A per-object run of 3 objects fills the object selector with 3 entries
    plus the two 'All (overlay)' entries, and shows the selector row."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, objects=_per_object_objects(3))

    # isHidden (not isVisible) is the reliable check in headless tests: the
    # top-level widget is never shown, so isVisible() is always False.
    assert not widget._object_row.isHidden()
    # 3 objects + 2 overlay entries (smoothed + raw)
    assert widget._object_selector.count() == 5
    assert widget._object_selector.itemText(0) == 'Object 1'
    assert widget._object_selector.itemData(3) == 'overlay_smoothed'
    assert widget._object_selector.itemData(4) == 'overlay_raw'
    # Results reflect the first object (label 1, radius 5.0)
    assert widget._object_caption.text() == 'Object 1'
    assert widget._radius_label.text() == '5.00 vox'
    assert len(widget._results) == 3


def test_aggregate_run_hides_selector(make_napari_viewer):
    """Aggregate mode shows a single curve and hides the object selector."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget)  # default aggregate

    assert widget._object_row.isHidden()
    assert widget._object_caption.text() == 'All combined'


def test_overlay_entry_omitted_above_max(make_napari_viewer):
    """With more than _MAX_OVERLAY objects, the overlay entries are not
    offered (the one-at-a-time selector still lists every object)."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    n = _MAX_OVERLAY + 1
    _fake_completed_run(widget, objects=_per_object_objects(n))

    assert widget._object_selector.count() == n  # no overlay entries
    data = [
        widget._object_selector.itemData(i)
        for i in range(widget._object_selector.count())
    ]
    assert 'overlay_smoothed' not in data
    assert 'overlay_raw' not in data


def test_selecting_object_updates_results(make_napari_viewer):
    """Switching the object selector updates the metric rows + caption."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, objects=_per_object_objects(3))

    # Select the 2nd object (index 1, label 2, radius 6.0)
    widget._object_selector.setCurrentIndex(1)
    assert widget._object_caption.text() == 'Object 2'
    assert widget._radius_label.text() == '6.00 vox'

    # Select the smoothed-overlay entry (index 3) → metrics blanked
    widget._object_selector.setCurrentIndex(3)
    assert widget._radius_label.text() == '—'
    assert 'All objects' in widget._object_caption.text()

    # Select the raw-overlay entry (index 4) → still an overlay view
    widget._object_selector.setCurrentIndex(4)
    assert widget._radius_label.text() == '—'
    assert 'All objects' in widget._object_caption.text()


def test_per_object_degenerate_object_kept(make_napari_viewer):
    """A single degenerate object among several must not abort the run: it
    is flagged 'no peak' in the selector while the others stay valid."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    objects = _per_object_objects(2)
    objects.append(
        {
            'label_id': 3,
            'series': np.zeros(101),
            'radius_vox': 0.0,
            'fwhm_vox': 0.0,
        }
    )
    _fake_completed_run(widget, objects=objects)

    assert len(widget._results) == 3
    assert widget._results[2]['ok'] is False
    # The degenerate object is labelled in the selector
    assert 'no peak' in widget._object_selector.itemText(2)
    # The valid first object is displayed
    assert widget._radius_label.text() == '5.00 vox'


def test_highlight_sets_selected_label(make_napari_viewer):
    """With the highlight checkbox on, selecting a per-object entry sets the
    napari Labels layer's selected_label + show_selected_label."""
    viewer = make_napari_viewer()
    # A real multi-label Labels layer for the highlight to target.
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[1:4, 1:4, 1:4] = 1
    data[6:9, 6:9, 6:9] = 2
    layer = viewer.add_labels(data, name='seg')

    widget = PersistentHomologyWidget(viewer)
    widget._highlight_check.setChecked(True)
    # run_params must name the real layer so the highlight can find it.
    result = _fake_worker_result(_MODE_EROSION, objects=_per_object_objects(2))
    widget._run_params = _make_run_params(
        _MODE_EROSION,
        analyze_each=True,
        label_ids=[1, 2],
        n_objects=2,
        seg_name='seg',
    )
    widget._on_result(result)

    # First object (label 1) selected → layer highlights label 1
    assert layer.show_selected_label is True
    assert layer.selected_label == 1

    # Switch to object 2 → label 2
    widget._object_selector.setCurrentIndex(1)
    assert layer.selected_label == 2

    # Turning the checkbox off clears the isolation
    widget._highlight_check.setChecked(False)
    assert layer.show_selected_label is False


def test_object_colors_match_labels_layer(make_napari_viewer):
    """Overlay curve colours are taken from the napari Labels layer, so each
    object's curve matches its colour in the viewer."""
    viewer = make_napari_viewer()
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[1:4, 1:4, 1:4] = 1
    data[6:9, 6:9, 6:9] = 2
    layer = viewer.add_labels(data, name='seg')

    widget = PersistentHomologyWidget(viewer)
    widget._run_params = _make_run_params(
        _MODE_EROSION,
        analyze_each=True,
        label_ids=[1, 2],
        n_objects=2,
        seg_name='seg',
    )
    widget._on_result(
        _fake_worker_result(_MODE_EROSION, objects=_per_object_objects(2))
    )

    colors = widget._object_colors(widget._results)
    assert len(colors) == 2
    # Each colour equals the layer's true palette colour for that label.
    # Read the expected colours with the highlight isolation off, since an
    # active highlight makes get_color return transparent for non-selected
    # labels (see test_object_colors_ignore_active_highlight).
    layer.show_selected_label = False
    for rec, color in zip(widget._results, colors, strict=True):
        expected = tuple(float(c) for c in layer.get_color(rec['label_id']))
        assert color == pytest.approx(expected)


def test_object_colors_ignore_active_highlight(make_napari_viewer):
    """Overlay colours must reflect the true per-label palette even when the
    viewer highlight is active.

    Regression: napari's 'show_selected_label' isolation makes 'get_color'
    return a *transparent* colour for every non-selected label. Saving the
    overlay while an individual object was highlighted therefore dropped every
    other object's curve (it was drawn transparent). '_object_colors' must lift
    the isolation for the read and restore it afterwards.
    """
    viewer = make_napari_viewer()
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[1:4, 1:4, 1:4] = 1
    data[6:9, 6:9, 6:9] = 2
    layer = viewer.add_labels(data, name='seg')

    widget = PersistentHomologyWidget(viewer)
    widget._run_params = _make_run_params(
        _MODE_EROSION,
        analyze_each=True,
        label_ids=[1, 2],
        n_objects=2,
        seg_name='seg',
    )
    widget._on_result(
        _fake_worker_result(_MODE_EROSION, objects=_per_object_objects(2))
    )

    # Reproduce the save-time state: one object is highlighted in the viewer.
    layer.selected_label = 1
    layer.show_selected_label = True

    colors = widget._object_colors(widget._results)
    assert len(colors) == 2
    # No colour may be fully transparent — every curve must be drawable.
    for color in colors:
        assert color[3] > 0.0
    # The two objects get distinct colours (not both the highlighted one).
    assert colors[0] != colors[1]
    # The highlight state is restored after the read.
    assert layer.show_selected_label is True


def test_object_colors_fall_back_without_layer(make_napari_viewer):
    """When the segmentation layer is gone, overlay colours fall back to the
    matplotlib cycle rather than crashing."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    # seg_name points at a non-existent layer.
    _fake_completed_run(widget, objects=_per_object_objects(2))
    colors = widget._object_colors(widget._results)
    assert len(colors) == 2
    assert all(c is not None for c in colors)


def test_per_object_save_results_one_row_per_object(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Save Results in per-object mode writes one row per object, each keyed
    by its Label_ID."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, objects=_per_object_objects(3))

    target = tmp_path / 'per_object.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_clicked()

    rows = _read_csv_rows(target)
    assert rows[0][0] == 'Label_ID'
    # Three data rows, one per label
    assert [r[0] for r in rows[1:]] == ['1', '2', '3']


def test_per_object_save_curve_long_format(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Save Curve in per-object mode writes a tidy/long table: every object's
    curve rows carry that object's Label_ID."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, objects=_per_object_objects(2))

    target = tmp_path / 'curves.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_curve_clicked()

    rows = _read_csv_rows(target)
    assert rows[0] == [
        'Label_ID',
        'Erosion_round',
        'Count_raw',
        'Count_smoothed',
    ]
    label_ids = {r[0] for r in rows[1:]}
    assert label_ids == {'1', '2'}
    # 2 objects × 101 rounds + 1 header
    assert len(rows) == 1 + 2 * 101


def test_per_object_save_writes_all_plots_in_subfolder(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Multi-object per-object runs collect one PNG per object plus both
    overlay PNGs in a subfolder named exactly after the chosen CSV stem — not
    just the currently-shown plot, and not loose in the chosen directory."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, objects=_per_object_objects(3))

    target = tmp_path / 'run.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_curve_clicked()

    # Folder is the CSV stem itself; erosion → 'object_count' file prefix.
    plot_dir = tmp_path / 'run'
    assert plot_dir.is_dir()
    # One PNG per object (labels 1, 2, 3) + both overlays, all in the subfolder.
    for label in (1, 2, 3):
        assert (plot_dir / f'object_count_obj_{label}.png').is_file()
    assert (plot_dir / 'object_count_overlay_smoothed.png').is_file()
    assert (plot_dir / 'object_count_overlay_raw.png').is_file()
    # The CSV sits beside the folder, and no loose single-object PNG.
    assert (tmp_path / 'run.csv').is_file()
    assert not (tmp_path / 'run.png').is_file()


def test_per_object_save_dilation_uses_hole_count_prefix(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Dilation runs name the per-object PNGs 'hole_count_obj_<label>.png' (the
    curve counts holes, not objects), inside the CSV-stem folder."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(
        widget, mode_text=_MODE_DILATION, objects=_per_object_objects(2)
    )

    target = tmp_path / 'run.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_curve_clicked()

    plot_dir = tmp_path / 'run'
    assert plot_dir.is_dir()
    assert (plot_dir / 'hole_count_obj_1.png').is_file()
    assert (plot_dir / 'hole_count_obj_2.png').is_file()
    assert (plot_dir / 'hole_count_overlay_smoothed.png').is_file()
    assert (plot_dir / 'hole_count_overlay_raw.png').is_file()


def test_single_object_save_writes_png_directly(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Single-object analysis (aggregate / binary case) saves the one plot
    directly next to the CSV — no subfolder."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget)  # aggregate = single object

    target = tmp_path / 'run.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_curve_clicked()

    assert (tmp_path / 'run.png').is_file()
    # Single object → no plot subfolder is created.
    assert not (tmp_path / 'run').exists()


def test_per_object_save_skips_degenerate_object_png(
    make_napari_viewer, tmp_path, monkeypatch
):
    """A degenerate object has no curve, so no per-object PNG is written for
    it (the overlays are still written, drawing it as a flat 'no peak' line)."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    objects = _per_object_objects(2)
    objects.append(
        {
            'label_id': 3,
            'series': np.zeros(101),
            'radius_vox': 0.0,
            'fwhm_vox': 0.0,
        }
    )
    _fake_completed_run(widget, objects=objects)

    target = tmp_path / 'run.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_curve_clicked()

    plot_dir = tmp_path / 'run'
    assert (plot_dir / 'object_count_obj_1.png').is_file()
    assert (plot_dir / 'object_count_obj_2.png').is_file()
    # degenerate object 3 → skipped
    assert not (plot_dir / 'object_count_obj_3.png').is_file()
    assert (plot_dir / 'object_count_overlay_smoothed.png').is_file()


@pytest.mark.parametrize(
    ('mode', 'analyze_each', 'kind', 'expected'),
    [
        (
            _MODE_EROSION,
            True,
            'measurements',
            'Object_radius_per_object_measurements.csv',
        ),
        (
            _MODE_EROSION,
            False,
            'measurements',
            'Object_radius_combined_objects_measurements.csv',
        ),
        (
            _MODE_DILATION,
            True,
            'count_curve_data',
            'Object_spacing_per_object_count_curve_data.csv',
        ),
        (
            _MODE_DILATION_INTERNAL,
            True,
            'count_curve_data',
            'Internal_spacing_per_object_count_curve_data.csv',
        ),
    ],
)
def test_default_save_name_encodes_mode_and_analyze(
    make_napari_viewer, mode, analyze_each, kind, expected
):
    """The prefilled Save-dialog default name encodes the analysis mode and the
    combined/per-object choice, so runs get distinctive names out of the box."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    widget._run_params = _make_run_params(mode, analyze_each=analyze_each)
    assert widget._default_save_name(kind) == expected


def test_save_measurements_dedup_increments_on_collision(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Saving Measurement Results over an existing file does not overwrite it —
    the name is auto-incremented to '_2'."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget)  # aggregate single object

    existing = tmp_path / 'foo.csv'
    existing.write_text('DO NOT OVERWRITE', encoding='utf-8')
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(existing), 'CSV files (*.csv)'),
    )
    widget._on_save_clicked()

    # Original untouched; the new save landed on the '_2' name.
    assert existing.read_text(encoding='utf-8') == 'DO NOT OVERWRITE'
    assert (tmp_path / 'foo_2.csv').is_file()


def test_save_curve_dedup_increments_on_folder_collision(
    make_napari_viewer, tmp_path, monkeypatch
):
    """A multi-object curve save whose plot folder name already exists bumps the
    whole stem to '_2' (CSV + folder), leaving the earlier folder intact."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, objects=_per_object_objects(2))

    # An earlier run already produced the 'run' plot folder.
    (tmp_path / 'run').mkdir()
    target = tmp_path / 'run.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_curve_clicked()

    # New outputs use the '_2' stem; the pre-existing folder is left empty.
    assert (tmp_path / 'run_2.csv').is_file()
    new_dir = tmp_path / 'run_2'
    assert (new_dir / 'object_count_obj_1.png').is_file()
    assert (new_dir / 'object_count_obj_2.png').is_file()
    assert not any((tmp_path / 'run').iterdir())


def test_save_measurements_dedup_increments_to_third(
    make_napari_viewer, tmp_path, monkeypatch
):
    """Two pre-existing summary CSVs → the save lands on '_3', not '_2'."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget)

    (tmp_path / 'foo.csv').write_text('first', encoding='utf-8')
    (tmp_path / 'foo_2.csv').write_text('second', encoding='utf-8')
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(tmp_path / 'foo.csv'), 'CSV files (*.csv)'),
    )
    widget._on_save_clicked()

    assert (tmp_path / 'foo_3.csv').is_file()
    assert (tmp_path / 'foo.csv').read_text(encoding='utf-8') == 'first'
    assert (tmp_path / 'foo_2.csv').read_text(encoding='utf-8') == 'second'


def test_save_results_writes_physical_unit_columns(
    make_napari_viewer, tmp_path, monkeypatch
):
    """A physical-unit run adds '<base>_<unit>' columns with values scaled by
    the mean voxel pitch; 'µm' is written as the ASCII column suffix 'um'."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    _fake_completed_run(widget, mode_text=_MODE_EROSION)  # radius_vox = 5.0
    # Re-snapshot the run params as a physical (µm) run with a 2.0 mean pitch.
    widget._run_params = _make_run_params(
        _MODE_EROSION, unit='µm', vx=2.0, vy=2.0, vz=2.0
    )

    target = tmp_path / 'phys.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_clicked()

    text = target.read_text(encoding='utf-8-sig')
    # µm → 'um' column suffix; voxel columns are still present.
    assert 'Radius_half_thickness_um' in text
    assert 'Width_thickness_um' in text
    assert 'FWHM_um' in text
    assert 'Radius_half_thickness_vox' in text
    # Physical radius = radius_vox (5.0) * mean pitch (2.0) = 10.0.
    assert '10.0000' in text


def test_save_results_degenerate_object_writes_nan(
    make_napari_viewer, tmp_path, monkeypatch
):
    """An object with no detected peak is written as a NaN row in the summary."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    objects = _per_object_objects(2)
    objects.append(
        {
            'label_id': 3,
            'series': np.zeros(101),
            'radius_vox': 0.0,
            'fwhm_vox': 0.0,
        }
    )
    _fake_completed_run(widget, objects=objects)

    target = tmp_path / 'deg.csv'
    monkeypatch.setattr(
        'napari_persistent_homology._widget.QFileDialog.getSaveFileName',
        lambda *_a, **_k: (str(target), 'CSV files (*.csv)'),
    )
    widget._on_save_clicked()

    text = target.read_text(encoding='utf-8-sig')
    # Voxel-only run (unit='vox'), so the degenerate label-3 row is all NaN.
    assert '3,NaN,NaN,NaN' in text


# ── Worker (_run_analysis) round-trips — anchor the fakes against the real
#    worker so drift in the returned dict's keys/units is caught ─────────────


def test_run_analysis_worker_aggregate_erosion_dict_shape():
    """The aggregate worker returns exactly the dict '_on_result' consumes."""
    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    vol[4:12, 4:12, 4:12] = 1
    result = _run_analysis(
        vol, None, 'erosion', False, [1], 0.5, 6, 26, 2, False, None
    ).work()

    assert result['analyze_each'] is False
    assert result['mode_key'] == 'erosion'
    assert result['curve_label'] == _CURVE_LABEL['erosion']
    assert len(result['objects']) == 1
    obj = result['objects'][0]
    assert set(obj) == {'label_id', 'series', 'radius_vox', 'fwhm_vox'}
    assert obj['label_id'] is None
    assert isinstance(obj['radius_vox'], float)
    assert isinstance(obj['fwhm_vox'], float)
    assert len(obj['series']) > 0


def test_run_analysis_worker_per_object_two_labels():
    """Per-object mode yields one job per label, each keyed by its label ID."""
    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    vol[2:6, 2:6, 2:6] = 1
    vol[9:14, 9:14, 9:14] = 2
    result = _run_analysis(
        vol, None, 'erosion', True, [1, 2], 0.5, 6, 26, 2, False, None
    ).work()

    assert result['analyze_each'] is True
    assert [o['label_id'] for o in result['objects']] == [1, 2]
    for o in result['objects']:
        assert set(o) == {'label_id', 'series', 'radius_vox', 'fwhm_vox'}


def test_run_analysis_worker_aggregate_dilation():
    """The dilation branch of _compute_series returns a hole-count curve."""
    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    vol[3:6, 3:6, 3:6] = 1
    vol[10:13, 10:13, 10:13] = 1
    result = _run_analysis(
        vol, None, 'dilation', False, [1], 0.5, 6, 26, 2, False, None
    ).work()

    assert result['mode_key'] == 'dilation'
    assert result['curve_label'] == _CURVE_LABEL['dilation']
    assert len(result['objects'][0]['series']) > 0


def test_run_analysis_internal_with_container_runs():
    """The dilation_internal branch runs with a valid container mask."""
    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    vol[6:10, 6:10, 6:10] = 1
    container = np.zeros((16, 16, 16), dtype=np.uint8)
    container[3:13, 3:13, 3:13] = 1  # container encloses the object
    result = _run_analysis(
        vol,
        container,
        'dilation_internal',
        False,
        [1],
        0.5,
        6,
        26,
        2,
        False,
        None,
    ).work()

    assert result['mode_key'] == 'dilation_internal'
    assert result['curve_label'] == _CURVE_LABEL['dilation_internal']
    assert len(result['objects'][0]['series']) > 0


def test_run_analysis_internal_empty_container_is_degenerate():
    """An object whose cropped container is empty is flagged 'no peak' (radius/
    FWHM = 0, a zeros curve) and the progress bar jumps past its slice — the
    worker must not crash."""
    vol = np.zeros((16, 16, 16), dtype=np.uint8)
    vol[2:6, 2:6, 2:6] = 1  # object in one corner
    container = np.zeros((16, 16, 16), dtype=np.uint8)
    container[10:15, 10:15, 10:15] = 1  # container elsewhere — misses the bbox

    steps = []
    result = _run_analysis(
        vol,
        container,
        'dilation_internal',
        True,
        [1],
        0.5,
        6,
        26,
        2,
        False,
        lambda s, t: steps.append((s, t)),
    ).work()

    obj = result['objects'][0]
    assert obj['label_id'] == 1
    assert obj['radius_vox'] == 0.0
    assert obj['fwhm_vox'] == 0.0
    assert len(obj['series']) == 6 + 1
    assert np.all(obj['series'] == 0)
    # Progress advanced past the object's slice (base 0 + max_steps 6).
    assert steps[-1] == (6, 6)


def test_make_step_callback_offsets_step_and_reports_total():
    """_make_step_callback shifts the local step by 'base' and reports the
    shared total, so one progress bar spans an N-object sweep."""
    calls = []
    cb = _make_step_callback(
        base=6, total_steps=12, step_callback=lambda s, t: calls.append((s, t))
    )
    cb(3, 6)
    assert calls == [(9, 12)]


def test_make_step_callback_none_is_noop():
    """A None step_callback is tolerated (no-op)."""
    cb = _make_step_callback(0, 6, None)
    cb(2, 6)  # must not raise


# ── Worker signal slots ─────────────────────────────────────────────────────


def test_on_error_capitalizes_message(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    widget._on_error(ValueError('boom happened'))
    assert 'Error: Boom happened' in widget._status_label.text()


def test_on_error_empty_message_does_not_crash(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    widget._on_error(ValueError(''))
    assert 'Error' in widget._status_label.text()


def test_on_finished_reenables_run_and_fills_bar(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    widget._run_btn.setEnabled(False)
    widget._progress_bar.setRange(0, 50)
    widget._progress_bar.setValue(10)

    widget._on_finished()

    assert widget._run_btn.isEnabled()
    assert widget._progress_bar.value() == widget._progress_bar.maximum() == 50


# ── Run-time validation branches ────────────────────────────────────────────


def test_run_internal_no_container_selected_shows_error(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.ones((10, 10, 10), dtype=np.uint8), name='seg')
    widget = PersistentHomologyWidget(viewer)
    widget._mode_combo.setCurrentText(_MODE_DILATION_INTERNAL)
    widget._seg_combo.setCurrentText('seg')
    widget._container_combo.clear()  # no container available/selected

    widget._on_run_clicked()

    text = widget._status_label.text()
    assert 'Error' in text
    assert 'container' in text.lower()


def test_run_empty_segmentation_shows_error(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.zeros((10, 10, 10), dtype=np.uint8), name='seg')
    widget = PersistentHomologyWidget(viewer)
    widget._seg_combo.setCurrentText('seg')

    widget._on_run_clicked()

    assert 'empty' in widget._status_label.text().lower()


def test_run_with_absent_label_id_shows_error(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[2:5, 2:5, 2:5] = 1
    viewer.add_labels(data, name='seg')
    widget = PersistentHomologyWidget(viewer)
    widget._seg_combo.setCurrentText('seg')
    widget._analyze_combo.setCurrentText('Each object')
    widget._label_ids_edit.setText('99')  # not present → error before any run

    widget._on_run_clicked()

    assert 'Error' in widget._status_label.text()


# ── parse_label_ids malformed inputs ────────────────────────────────────────


@pytest.mark.parametrize('text', ['abc', '1,abc', '1 2 x', '[]'])
def test_parse_label_ids_rejects_junk(text):
    with pytest.raises(ValueError):
        parse_label_ids(text, [1, 2, 3])


@pytest.mark.parametrize('text', ['99', '-1'])
def test_parse_label_ids_absent_label_raises(text):
    with pytest.raises(ValueError):
        parse_label_ids(text, [1, 2, 3])


def test_parse_label_ids_whitespace_is_all():
    assert parse_label_ids('   ', [3, 1, 2]) == [1, 2, 3]


# ── _process_object display branches ────────────────────────────────────────


def test_process_object_partial_pitch_falls_back_to_voxels(make_napari_viewer):
    """A physical unit with one voxel dimension unset (0) shows voxel-only
    strings (no physical brackets)."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    p = _make_run_params(_MODE_EROSION, unit='nm', vx=5.0, vy=5.0, vz=0.0)
    rec = widget._process_object(
        {
            'label_id': 1,
            'series': _plateau_series(),
            'radius_vox': 5.0,
            'fwhm_vox': 2.0,
        },
        p,
    )
    assert rec['ok'] is True
    assert rec['radius_str'] == '5.00 vox'  # voxel-only, no '(… vox)' bracket


def test_process_object_zero_at_peak_is_not_ok(make_napari_viewer):
    """A non-zero radius whose peak step lands on a zero of the smoothed curve
    is flagged degenerate ('—')."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    p = _make_run_params(_MODE_EROSION)  # Lambda 0.1 → peak step = 50
    rec = widget._process_object(
        {
            'label_id': 1,
            'series': np.zeros(101),  # flat, so smoothed[50] == 0
            'radius_vox': 5.0,
            'fwhm_vox': 2.0,
        },
        p,
    )
    assert rec['ok'] is False
    assert rec['radius_str'] == '—'


def test_selecting_degenerate_object_shows_no_peak_message(make_napari_viewer):
    """Selecting a degenerate object in the plot selector blanks the metric
    rows to '—' (its curve has no peak to display)."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    objects = _per_object_objects(1)  # label 1 valid
    objects.append(
        {
            'label_id': 2,
            'series': np.zeros(101),
            'radius_vox': 0.0,
            'fwhm_vox': 0.0,
        }
    )
    _fake_completed_run(widget, objects=objects)

    # combo index 1 = object 2 (the degenerate one), before the overlay entries.
    widget._object_selector.setCurrentIndex(1)
    assert widget._radius_label.text() == '—'


# ── Small helpers ───────────────────────────────────────────────────────────


def test_strip_save_ext_handles_csv_png_and_bare():
    f = PersistentHomologyWidget._strip_save_ext
    assert f('a/b/run.csv') == 'a/b/run'
    assert f('a/b/run.PNG') == 'a/b/run'  # case-insensitive
    assert f('a/b/run') == 'a/b/run'  # no known extension → unchanged
    assert (
        f('a/b/run.csv.png') == 'a/b/run.csv'
    )  # strips only the trailing ext


def test_seg_layer_returns_none_for_non_labels(make_napari_viewer):
    """_seg_layer guards against the run's layer having become a non-Labels
    layer (e.g. renamed onto an Image)."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)
    viewer.add_image(np.zeros((5, 5, 5)), name='img')
    widget._run_params = _make_run_params(_MODE_EROSION, seg_name='img')
    assert widget._seg_layer() is None
