import numpy as np

from napari_persistent_homology._widget import (
    _MODE_DILATION,
    _MODE_DILATION_INTERNAL,
    _MODE_EROSION,
    PersistentHomologyWidget,
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


def _fake_completed_run(widget, mode_text=None):
    """Helper: drive widget._on_result with a non-degenerate fake result
    so the two save buttons are enabled and self._last_result is
    populated as if a real analysis had completed."""
    if mode_text is None:
        mode_text = _MODE_EROSION
    widget._run_params = {
        'mode': mode_text,
        'Lambda': 0.1,
        'max_steps': 100,
        'connectivity': 26,
        'SIGMA': 5.0,
        'offset': 5,
        'unit': 'vox',
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
    }
    series = np.zeros(101)
    series[40:60] = 5.0
    widget._on_result((series, 5.0, 2.0, 'Object count', 5.0))


def _read_csv_rows(path):
    """Read a CSV file written by the widget and return all non-comment
    data rows as lists of strings. Comment lines starting with '#' and
    blank rows are skipped."""
    import csv as _csv

    rows = []
    with open(path, encoding='utf-8', newline='') as f:
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
    # Column header — 'metric' (not 'stat')
    assert rows[0] == ['Metric', 'Value_vox']
    metric_labels = [r[0] for r in rows[1:]]
    assert metric_labels == [
        'Radius / half-thickness',
        'Width / thickness',
        'Full-width at half-maximum',
    ]
    # The curve-x-axis columns must NOT appear in the summary
    assert not any(
        r[:1] in (['Erosion_round'], ['Dilation_round']) for r in rows
    )


def test_save_results_writes_summary_csv_dilation(
    make_napari_viewer,
    tmp_path,
    monkeypatch,
):
    """Save Results in dilation mode writes only TWO metrics ('Inter-
    object spacing' + FWHM) — the width row is hidden in the GUI for
    these modes, and the CSV mirrors that."""
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
    assert rows[0] == ['Metric', 'Value_vox']
    metric_labels = [r[0] for r in rows[1:]]
    assert metric_labels == [
        'Inter-object spacing',
        'Full-width at half-maximum',
    ]
    # No 'Width / thickness' anywhere
    assert not any('Width' in r[0] for r in rows[1:])


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
    # Erosion mode → the round column is named 'Erosion_round'
    assert rows[0] == ['Erosion_round', 'Count_raw', 'Count_smoothed']
    assert len(rows) == 102  # 1 header + 101 data rows (rounds 0..100)
    # The summary metric rows must NOT appear in the curve file
    assert not any(r[:1] == ['Metric'] for r in rows)
    metric_labels = {
        'Radius / half-thickness',
        'Width / thickness',
        'Full-width at half-maximum',
        'Inter-object spacing',
    }
    assert not any(r and r[0] in metric_labels for r in rows)


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

    # Simulate _on_started after a run_params snapshot
    widget._run_params = {'max_steps': 50}
    widget._on_started()

    assert widget._progress_bar.value() == 0
    assert widget._progress_bar.maximum() == 50
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
    # new-mode labels take effect.
    widget._reset_results()
    assert widget._radius_row_label.text() == 'Inter-object spacing:'
    assert widget._width_row_label.isHidden()
    assert widget._width_label.isHidden()

    # Switch back to erosion and reset again — labels restore
    widget._mode_combo.setCurrentText(_MODE_EROSION)
    widget._reset_results()
    assert 'Radius' in widget._radius_row_label.text()
    assert '(erosion)' in widget._radius_row_label.text()
    assert not widget._width_row_label.isHidden()
    assert not widget._width_label.isHidden()


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
    widget._last_result = ('placeholder',)

    widget._reset_results()

    assert widget._radius_label.text() == '—'
    assert widget._width_label.text() == '—'
    assert widget._fwhm_label.text() == '—'
    assert widget._save_btn.isEnabled() is False
    assert widget._save_curve_btn.isEnabled() is False
    assert widget._last_result is None


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
    widget._last_result = ('placeholder',)

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
    assert widget._last_result is None


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

    widget._run_params = {
        'mode': _MODE_EROSION,
        'Lambda': 0.1,
        'max_steps': 100,
        'connectivity': 26,
        'SIGMA': 5.0,
        'offset': 5,
        'unit': 'nm',
        'vx': 5.0,
        'vy': 5.0,
        'vz': 5.0,
    }
    # Synthetic non-degenerate count curve so the result path runs
    series = np.zeros(101)
    series[40:60] = 5.0
    fake_result = (series, 5.0, 2.0, 'Object count', 5.0)
    widget._on_result(fake_result)

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

    widget._run_params = {
        'mode': _MODE_EROSION,
        'Lambda': 0.1,
        'max_steps': 100,
        'connectivity': 26,
        'SIGMA': 5.0,
        'offset': 5,
        'unit': 'vox',
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
    }
    series = np.zeros(101)
    series[40:60] = 5.0
    fake_result = (series, 5.0, 2.0, 'Object count', 5.0)
    widget._on_result(fake_result)

    assert widget._radius_label.text() == '5.00 vox'
    assert '(' not in widget._radius_label.text()


def test_completion_status_includes_analysis_mode(make_napari_viewer):
    """After a successful run the status message must name the specific
    analysis mode that finished — generic 'Analysis completed.' is not
    enough to keep the three modes distinguishable in the UI."""
    viewer = make_napari_viewer()
    widget = PersistentHomologyWidget(viewer)

    # A non-degenerate count curve: zeros with a 5.0 plateau around the
    # peak step, so the degenerate-curve check passes and we reach the
    # completion path.
    series = np.zeros(101)
    series[40:60] = 5.0
    fake_result = (series, 5.0, 2.0, 'Object count', 5.0)

    cases = [
        (_MODE_EROSION, 'Object radius / half-thickness'),
        (_MODE_DILATION, 'Object spacing'),
        (_MODE_DILATION_INTERNAL, 'Internal spacing'),
    ]
    for mode_text, expected_phrase in cases:
        widget._run_params = {
            'mode': mode_text,
            'Lambda': 0.1,
            'max_steps': 100,
            'connectivity': 26,
            'SIGMA': 5.0,
            'offset': 5,
            'unit': 'vox',
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
        }
        widget._on_result(fake_result)
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
    widget._run_params = {
        'mode': _MODE_EROSION,
        'Lambda': 0.1,
        'max_steps': 100,
        'connectivity': 26,
        'SIGMA': 5.0,
        'offset': 5,
        'unit': 'vox',
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
    }

    # Simulate a degenerate worker output: all-zero count curve, with the
    # offset-fallback radius (0.5) and full-curve FWHM (10.0) that
    # find_max_location / compute_FWHM produce on a flat curve.
    series = np.zeros(101)
    fake_result = (series, 0.5, 10.0, 'Object count', 5.0)
    widget._on_result(fake_result)

    text = widget._status_label.text()
    assert 'Error' in text
    assert 'No peak detected' in text
    # All result widgets reset to their empty initial state
    assert widget._radius_label.text() == '—'
    assert widget._width_label.text() == '—'
    assert widget._fwhm_label.text() == '—'
    assert widget._save_btn.isEnabled() is False
    assert widget._save_curve_btn.isEnabled() is False
    assert widget._last_result is None
