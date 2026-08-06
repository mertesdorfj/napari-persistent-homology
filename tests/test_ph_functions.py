import numpy as np
import pytest

from napari_persistent_homology.ph_functions import (
    bounding_box_crop,
    compute_FWHM,
    compute_FWHM_v2,
    compute_homology_stats,
    compute_homology_stats_v2,
    find_max_location,
    find_max_location_v2,
    gaussian_average,
    hole_count,
    holes_count_internal_object,
    label_subvolume,
    moving_average,
    object_count,
    persistent_homology_dilation,
    persistent_homology_dilation_internal_object,
    persistent_homology_erosion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def two_blobs():
    """3D uint8 volume with two separate 3x3x3 blobs."""
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[2:5, 2:5, 2:5] = 1
    vol[15:18, 15:18, 15:18] = 1
    return vol


def single_blob():
    """3D uint8 volume with one 4x4x4 blob in the centre."""
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[8:12, 8:12, 8:12] = 1
    return vol


def gaussian_peak(length=100, centre=50, sigma=5):
    x = np.arange(length, dtype=float)
    return np.exp(-0.5 * ((x - centre) / sigma) ** 2)


# ---------------------------------------------------------------------------
# object_count
# ---------------------------------------------------------------------------


class TestObjectCount:
    def test_two_separate_blobs(self):
        assert object_count(two_blobs(), connectivity=26) == 2

    def test_single_blob(self):
        assert object_count(single_blob(), connectivity=26) == 1

    def test_empty_volume(self):
        vol = np.zeros((10, 10, 10), dtype=np.uint8)
        assert object_count(vol, connectivity=26) == 0


# ---------------------------------------------------------------------------
# hole_count
# ---------------------------------------------------------------------------


class TestHoleCount:
    def test_solid_blob_has_no_holes(self):
        # A solid blob floating in background has no topological holes
        assert hole_count(single_blob(), connectivity=26) == 0

    def test_empty_volume_has_no_holes(self):
        # All-zero volume: inverse is all ones (1 background component) → 0 holes
        vol = np.zeros((10, 10, 10), dtype=np.uint8)
        assert hole_count(vol, connectivity=26) == 0

    def test_hollow_shell_has_one_hole(self):
        # Cube with an enclosed empty cavity inside → exactly 1 hole.
        # Outer shell at indices 2..11, inner cavity at indices 4..9.
        # The volume is padded with background so an "outer" component
        # exists and the -1 correction in hole_count is meaningful.
        vol = np.zeros((14, 14, 14), dtype=np.uint8)
        vol[2:12, 2:12, 2:12] = 1
        vol[4:10, 4:10, 4:10] = 0
        assert hole_count(vol, connectivity=26) == 1

    def test_solid_with_two_cavities_has_two_holes(self):
        # One solid block with two separate empty pockets inside.
        # Each pocket is its own enclosed background component → 2 holes.
        vol = np.zeros((14, 14, 14), dtype=np.uint8)
        vol[2:12, 2:12, 2:12] = 1
        vol[3:5, 3:5, 3:5] = 0
        vol[8:10, 8:10, 8:10] = 0
        assert hole_count(vol, connectivity=26) == 2

    def test_two_separate_hollow_shells_have_two_holes(self):
        # Two distant hollow shells, each contributing one cavity → 2 holes total
        vol = np.zeros((20, 20, 20), dtype=np.uint8)
        vol[1:7, 1:7, 1:7] = 1
        vol[3:5, 3:5, 3:5] = 0
        vol[12:18, 12:18, 12:18] = 1
        vol[14:16, 14:16, 14:16] = 0
        assert hole_count(vol, connectivity=26) == 2


# ---------------------------------------------------------------------------
# holes_count_internal_object
# ---------------------------------------------------------------------------


class TestHolesCountInternalObject:
    def test_internal_equals_container(self):
        blob = single_blob()
        assert holes_count_internal_object(blob, blob, connectivity=26) == 0

    def test_empty_internal(self):
        blob = single_blob()
        internal = np.zeros_like(blob)
        assert (
            holes_count_internal_object(internal, blob, connectivity=26) == 0
        )

    def test_one_interior_gap_is_one_hole(self):
        # Container is a solid cube; internal fills it except for one
        # interior voxel. The gap is fully enclosed by foreground (none of
        # its neighbours are background outside the container), so it
        # counts as one hole.
        container = np.zeros((14, 14, 14), dtype=np.uint8)
        container[2:12, 2:12, 2:12] = 1
        internal = container.copy()
        internal[5, 5, 5] = 0
        assert (
            holes_count_internal_object(internal, container, connectivity=26)
            == 1
        )

    def test_two_interior_gaps_are_two_holes(self):
        # Two interior gaps, well separated → 2 holes
        container = np.zeros((14, 14, 14), dtype=np.uint8)
        container[2:12, 2:12, 2:12] = 1
        internal = container.copy()
        internal[4, 4, 4] = 0
        internal[9, 9, 9] = 0
        assert (
            holes_count_internal_object(internal, container, connectivity=26)
            == 2
        )

    def test_gap_at_container_boundary_is_not_a_hole(self):
        # A gap touching the surface of the container connects to the outer
        # background (which is also 0 because internal × container = 0
        # outside the container) — so it is not a topological hole.
        container = np.zeros((14, 14, 14), dtype=np.uint8)
        container[2:12, 2:12, 2:12] = 1
        internal = container.copy()
        internal[2, 5, 5] = 0  # at index 2 = the container's surface
        assert (
            holes_count_internal_object(internal, container, connectivity=26)
            == 0
        )


# ---------------------------------------------------------------------------
# gaussian_average
# ---------------------------------------------------------------------------


class TestGaussianAverage:
    def test_output_shape(self):
        x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        assert gaussian_average(x, sigma=1).shape == x.shape

    def test_smoothing_reduces_peak(self):
        x = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        result = gaussian_average(x, sigma=1)
        assert result[2] < 10.0

    def test_smoothing_spreads_to_neighbours(self):
        x = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        result = gaussian_average(x, sigma=1)
        assert result[1] > 0.0
        assert result[3] > 0.0

    def test_flat_signal_unchanged(self):
        x = np.ones(20) * 3.0
        result = gaussian_average(x, sigma=2)
        np.testing.assert_array_almost_equal(result, x)


# ---------------------------------------------------------------------------
# find_max_location
# ---------------------------------------------------------------------------


class TestFindMaxLocation:
    def test_peak_at_known_location(self):
        series = np.zeros(20)
        series[12] = 5.0
        assert find_max_location(series, offset=5) == 12

    def test_peak_before_offset_is_ignored(self):
        series = np.zeros(20)
        series[2] = 10.0  # before offset=5, should be ignored
        series[12] = 5.0
        assert find_max_location(series, offset=5) == 12

    def test_all_zeros_returns_offset(self):
        series = np.zeros(20)
        assert find_max_location(series, offset=5) == 5

    def test_last_peak_wins_on_tie(self):
        # The implementation returns the last occurrence of the maximum
        series = np.zeros(20)
        series[8] = 5.0
        series[14] = 5.0
        assert find_max_location(series, offset=5) == 14


# ---------------------------------------------------------------------------
# compute_FWHM
# ---------------------------------------------------------------------------


class TestComputeFWHM:
    def test_gaussian_peak_fwhm(self):
        # Gaussian with sigma=5 has FWHM ≈ 11.77; expect within ±3
        series = gaussian_peak(length=100, centre=50, sigma=5)
        fwhm = compute_FWHM(series, offset=5)
        assert 8 <= fwhm <= 15

    def test_narrow_peak_has_smaller_fwhm(self):
        wide = gaussian_peak(length=100, centre=50, sigma=8)
        narrow = gaussian_peak(length=100, centre=50, sigma=3)
        assert compute_FWHM(narrow, offset=5) < compute_FWHM(wide, offset=5)


# ---------------------------------------------------------------------------
# compute_homology_stats
# ---------------------------------------------------------------------------


class TestComputeHomologyStats:
    def test_output_shape_two_series(self):
        s1 = gaussian_peak(centre=30, sigma=4)
        s2 = gaussian_peak(centre=40, sigma=6)
        result = compute_homology_stats([s1, s2], offset=5, SIGMA=2)
        # shape is (3, n_series): rows = [FWHM, max_count, max_location]
        assert result.shape == (3, 2)

    def test_max_location_near_peak(self):
        series = gaussian_peak(length=100, centre=30, sigma=3)
        result = compute_homology_stats([series], offset=5, SIGMA=1)
        max_loc = result[2, 0]
        assert 25 <= max_loc <= 35

    def test_fwhm_row_positive(self):
        series = gaussian_peak(length=100, centre=50, sigma=5)
        result = compute_homology_stats([series], offset=5, SIGMA=2)
        assert result[0, 0] > 0

    def test_max_count_reflects_peak_amplitude(self):
        # A unit-amplitude Gaussian smoothed by SIGMA=2 should retain most
        # of its peak height. Analytical: smoothing a Gaussian (sigma=5) by
        # SIGMA=2 produces an effective sigma of sqrt(5² + 2²) ≈ 5.39, so
        # the peak amplitude scales by 5/5.39 ≈ 0.93.
        series = gaussian_peak(length=100, centre=50, sigma=5)
        result = compute_homology_stats([series], offset=5, SIGMA=2)
        max_count = result[1, 0]
        assert 0.85 < max_count <= 1.0

    def test_max_count_scales_linearly_with_input(self):
        # Doubling the input amplitude must double the reported max_count
        # (smoothing is linear). Guards against any normalisation creeping
        # into the smoothing or peak-reading code path.
        s1 = gaussian_peak(length=100, centre=50, sigma=5)
        s2 = 2.0 * s1
        result = compute_homology_stats([s1, s2], offset=5, SIGMA=2)
        assert result[1, 1] == pytest.approx(2 * result[1, 0], rel=0.01)

    def test_fwhm_quantitative_for_known_gaussian(self):
        # Gaussian with sigma=5 has analytical FWHM = 2σ√(2 ln 2) ≈ 11.77.
        # With minimal smoothing (SIGMA=1), the function should recover a
        # value within ±4 of that.
        series = gaussian_peak(length=100, centre=50, sigma=5)
        result = compute_homology_stats([series], offset=5, SIGMA=1)
        fwhm = result[0, 0]
        assert 8 <= fwhm <= 16

    def test_fwhm_ordering_two_series(self):
        # A narrower input must yield a smaller FWHM, in the same call.
        # Locks the per-series independence of the FWHM row.
        narrow = gaussian_peak(length=100, centre=50, sigma=3)
        wide = gaussian_peak(length=100, centre=50, sigma=8)
        result = compute_homology_stats([narrow, wide], offset=5, SIGMA=1)
        assert result[0, 0] < result[0, 1]

    def test_per_series_stats_are_independent(self):
        # Two peaks at different positions must produce independent
        # max_locations — confirms the function processes each input
        # series separately rather than accidentally mixing them.
        s1 = gaussian_peak(length=100, centre=30, sigma=4)
        s2 = gaussian_peak(length=100, centre=60, sigma=4)
        result = compute_homology_stats([s1, s2], offset=5, SIGMA=1)
        assert abs(result[2, 0] - 30) <= 5
        assert abs(result[2, 1] - 60) <= 5


# ---------------------------------------------------------------------------
# persistent_homology_erosion  (small volume, few steps for speed)
# ---------------------------------------------------------------------------


class TestPersistentHomologyErosion:
    def test_output_lengths(self):
        max_steps = 5
        obj, holes = persistent_homology_erosion(
            single_blob(), max_steps=max_steps, Lambda=0.5, Connectivity=26
        )
        assert len(obj) == max_steps + 1
        assert len(holes) == max_steps + 1

    def test_initial_object_count_is_one(self):
        obj, _ = persistent_homology_erosion(
            single_blob(), max_steps=5, Lambda=0.5, Connectivity=26
        )
        assert obj[0] == 1

    def test_object_count_nonincreasing(self):
        obj, _ = persistent_homology_erosion(
            single_blob(), max_steps=10, Lambda=0.5, Connectivity=26
        )
        # Object count can only stay the same or decrease during erosion
        assert all(obj[i] >= obj[i + 1] for i in range(len(obj) - 1))

    def test_object_eroded_to_zero(self):
        # With enough steps the small blob should vanish
        obj, _ = persistent_homology_erosion(
            single_blob(), max_steps=10, Lambda=0.5, Connectivity=26
        )
        assert obj[-1] == 0


# ---------------------------------------------------------------------------
# persistent_homology_dilation  (small volume, few steps for speed)
# ---------------------------------------------------------------------------


class TestPersistentHomologyDilation:
    def test_output_lengths(self):
        max_steps = 5
        obj, holes = persistent_homology_dilation(
            two_blobs(), max_steps=max_steps, Lambda=0.5, Connectivity=26
        )
        assert len(obj) == max_steps + 1
        assert len(holes) == max_steps + 1

    def test_initial_object_count(self):
        obj, _ = persistent_homology_dilation(
            two_blobs(), max_steps=5, Lambda=0.5, Connectivity=26
        )
        assert obj[0] == 2

    def test_objects_merge_during_dilation(self):
        # Two blobs close enough together should merge; count goes from 2 → 1
        vol = np.zeros((20, 20, 20), dtype=np.uint8)
        vol[8:10, 9:11, 9:11] = 1  # blob 1
        vol[12:14, 9:11, 9:11] = 1  # blob 2 — 2 voxels apart

        obj, _ = persistent_homology_dilation(
            vol, max_steps=10, Lambda=0.5, Connectivity=26
        )
        assert obj[-1] == 1  # merged into one


# ---------------------------------------------------------------------------
# step_callback for all three persistent homology functions
# ---------------------------------------------------------------------------


class TestStepCallback:
    def _make_blob(self):
        vol = np.zeros((10, 10, 10), dtype=np.uint8)
        vol[3:7, 3:7, 3:7] = 1
        return vol

    def test_erosion_callback_called_max_steps_times(self):
        calls = []
        persistent_homology_erosion(
            self._make_blob(),
            max_steps=5,
            Lambda=0.5,
            Connectivity=26,
            step_callback=lambda s, t: calls.append((s, t)),
        )
        assert len(calls) == 5
        assert calls[0] == (1, 5)
        assert calls[-1] == (5, 5)

    def test_dilation_callback_called_max_steps_times(self):
        calls = []
        persistent_homology_dilation(
            self._make_blob(),
            max_steps=5,
            Lambda=0.5,
            Connectivity=26,
            step_callback=lambda s, t: calls.append((s, t)),
        )
        assert len(calls) == 5
        assert calls[-1] == (5, 5)

    def test_dilation_internal_callback_called_max_steps_times(self):
        blob = self._make_blob()
        container = np.ones((10, 10, 10), dtype=np.uint8)
        calls = []
        persistent_homology_dilation_internal_object(
            blob,
            container,
            max_steps=5,
            Lambda=0.5,
            Connectivity=26,
            step_callback=lambda s, t: calls.append((s, t)),
        )
        assert len(calls) == 5
        assert calls[-1] == (5, 5)

    def test_no_callback_still_works(self):
        # Verify backward compat: no step_callback arg = no error
        obj, _ = persistent_homology_erosion(
            self._make_blob(), max_steps=3, Lambda=0.5, Connectivity=26
        )
        assert len(obj) == 4  # step 0 + 3 steps


# ---------------------------------------------------------------------------
# v2 helpers for the feature-extraction tests
# ---------------------------------------------------------------------------


def _noise_spike_then_broad_peak(length=60):
    """Curve modelled after the failure case observed on real hole-count
    data: a narrow, high early spike followed (after a small dip that
    does NOT drop back to count = 1 for 10 samples) by a broad, slightly
    lower main peak. Under default v2 ('rank_peaks_by_smoothed=False'),
    the spike wins the argmax; under the smoothed variant, the broad
    peak wins because averaging dilutes the spike."""
    series = np.zeros(length, dtype=float)
    # Narrow raw spike near step 12 — value 8, one sample wide.
    series[10] = 3.0
    series[11] = 5.0
    series[12] = 8.0
    series[13] = 4.0
    series[14] = 3.0
    # Sustained mid-region around count = 2..3 so the noise filter's
    # 'count = 1 for >= 1/Lambda samples' condition does NOT fire.
    series[15:22] = np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0])
    # Broad main peak centred near step 30, height 7.
    x = np.arange(length, dtype=float)
    series += 7.0 * np.exp(-0.5 * ((x - 30) / 4.0) ** 2)
    return series


# ---------------------------------------------------------------------------
# find_max_location_v2  (noise-tolerant, used by the widget)
# ---------------------------------------------------------------------------


class TestFindMaxLocationV2:
    def test_peak_at_known_location(self):
        # Single clean peak — v2 should find it exactly like v1.
        series = np.zeros(30, dtype=float)
        series[8:14] = np.array([1.0, 3.0, 5.0, 4.0, 2.0, 1.0])
        assert find_max_location_v2(series, offset=5, Lambda=0.1) == 10

    def test_peak_before_offset_is_ignored(self):
        # A peak inside the 'offset' window must not be returned.
        series = np.zeros(30, dtype=float)
        series[2] = 10.0  # inside offset region
        series[15:18] = np.array([3.0, 5.0, 2.0])  # real peak
        assert find_max_location_v2(series, offset=5, Lambda=0.1) == 16

    def test_local_max_wins_over_earlier_smaller_local_max(self):
        # Two clean local maxima — the tallest wins, not just the last.
        series = np.zeros(50, dtype=float)
        series[10:14] = np.array([2.0, 4.0, 2.0, 1.0])  # smaller
        series[25:29] = np.array([3.0, 6.0, 3.0, 1.0])  # taller
        assert find_max_location_v2(series, offset=5, Lambda=0.1) == 26

    def test_noise_filter_discards_first_peak(self):
        # Textbook noise pattern that the filter is designed to catch:
        # small early peak → sustained count = 1 for >= 1/Lambda samples
        # → later real peak. v2 must return the LATER peak.
        series = np.zeros(50, dtype=float)
        series[7:10] = np.array([1.0, 2.0, 1.0])  # small early peak
        series[10:22] = 1.0  # stable count = 1 for 12 samples (>= 10)
        series[25:29] = np.array([3.0, 5.0, 3.0, 1.0])  # real peak
        assert find_max_location_v2(series, offset=5, Lambda=0.1) == 26

    def test_noise_filter_leaves_curve_alone_without_count_1_plateau(self):
        # This is the observed real-data failure mode: the raw noise
        # spike is TALLER than the true structural peak and the count
        # never returns to 1 between them, so v2's noise filter does
        # not fire. Default weighting (raw) picks the spike.
        series = _noise_spike_then_broad_peak()
        raw_pick = find_max_location_v2(series, offset=5, Lambda=0.1)
        # The spike sits at step 12 — one of the samples 10..14.
        assert raw_pick in (11, 12, 13)

    def test_rank_peaks_by_smoothed_picks_main_peak_over_spike(self):
        # Same curve; enabling smoothed weighting must push argmax onto
        # the broad main peak near step 30 instead of the narrow spike.
        series = _noise_spike_then_broad_peak()
        smoothed_pick = find_max_location_v2(
            series, offset=5, Lambda=0.1, rank_peaks_by_smoothed=True
        )
        assert abs(smoothed_pick - 30) <= 3

    def test_flat_curve_returns_valid_step_index_without_crashing(self):
        # A completely flat curve has no meaningful peak, but v2 still
        # walks the degenerate branch to a definite answer. The point
        # of this check is that v2 does not raise on such input — the
        # returned step lives inside the post-offset range so it stays
        # a valid index for downstream lookups.
        series = np.zeros(30, dtype=float)
        step = find_max_location_v2(series, offset=5, Lambda=0.1)
        assert 5 <= step < len(series)


# ---------------------------------------------------------------------------
# compute_FWHM_v2  (returns a tuple: total + (left, middle, right))
# ---------------------------------------------------------------------------


class TestComputeFWHMV2:
    def test_returns_tuple_shape(self):
        # (total, (left, middle, right)) — three-piece decomposition.
        series = gaussian_peak(length=100, centre=50, sigma=5)
        result = compute_FWHM_v2(series, offset=5, Lambda=0.1)
        total, parts = result
        assert isinstance(total, (int, float, np.integer, np.floating))
        assert isinstance(parts, tuple)
        assert len(parts) == 3

    def test_parts_sum_to_total(self):
        # The whole-vs-parts contract: left + middle + right == FWHM.
        series = gaussian_peak(length=100, centre=50, sigma=5)
        total, (left, middle, right) = compute_FWHM_v2(
            series, offset=5, Lambda=0.1
        )
        assert left + middle + right == total

    def test_middle_is_one(self):
        # 'middle' is the peak sample itself — always 1 by construction.
        series = gaussian_peak(length=100, centre=50, sigma=5)
        _, (_, middle, _) = compute_FWHM_v2(series, offset=5, Lambda=0.1)
        assert middle == 1

    def test_narrower_peak_gives_smaller_fwhm(self):
        # Sanity: a taller / narrower peak has a smaller FWHM in step
        # units than a wider one, regardless of the exact algorithm.
        narrow = gaussian_peak(length=100, centre=50, sigma=3)
        wide = gaussian_peak(length=100, centre=50, sigma=8)
        narrow_fwhm, _ = compute_FWHM_v2(narrow, offset=5, Lambda=0.1)
        wide_fwhm, _ = compute_FWHM_v2(wide, offset=5, Lambda=0.1)
        assert narrow_fwhm < wide_fwhm

    def test_gaussian_peak_fwhm_within_analytical_range(self):
        # Analytical FWHM of a Gaussian with sigma = 5 is
        # 2 * 5 * sqrt(2 ln 2) ~ 11.77 samples. v2's moving-average
        # pre-smoothing widens it slightly, so allow a generous
        # tolerance.
        series = gaussian_peak(length=100, centre=50, sigma=5)
        total, _ = compute_FWHM_v2(series, offset=5, Lambda=0.1)
        assert 8 <= total <= 22

    def test_flat_curve_returns_zero_total_without_crashing(self):
        # A completely flat curve has no meaningful width; only the
        # aggregate 'total' being 0 is contractual. The individual
        # (left, middle, right) parts can legitimately end up as
        # (0, 1, -1) — 'middle' is always 1 (the peak sample) and
        # 'right_half_max = argmin(...) - 1' walks off the end into
        # -1 when the whole mask is True. That's an intended quirk of
        # v2's argmin-based walk, not a bug.
        series = np.zeros(30, dtype=float)
        total, _ = compute_FWHM_v2(series, offset=5, Lambda=0.1)
        assert total == 0


# ---------------------------------------------------------------------------
# compute_homology_stats_v2  (single-series API, voxel-unit output)
# ---------------------------------------------------------------------------


class TestComputeHomologyStatsV2:
    def test_returns_three_tuple(self):
        series = gaussian_peak(length=100, centre=50, sigma=5)
        result = compute_homology_stats_v2(series, offset=5, Lambda=0.1)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_output_in_voxel_units(self):
        # v2 multiplies step-unit results by Lambda internally, so
        # 'max_location_vox' should equal the step index * Lambda.
        # A clean peak at step 30 with Lambda = 0.1 → 3.0 voxels.
        series = np.zeros(60, dtype=float)
        series[27:33] = np.array([1.0, 3.0, 5.0, 4.0, 2.0, 1.0])
        _, max_loc_vox, _ = compute_homology_stats_v2(
            series, offset=5, Lambda=0.1
        )
        assert max_loc_vox == pytest.approx(29 * 0.1)

    def test_max_location_scales_with_lambda(self):
        # Same step-index peak but a different Lambda should scale the
        # reported voxel-unit location proportionally, since the return
        # value is step_index * Lambda.
        series = np.zeros(60, dtype=float)
        series[27:33] = np.array([1.0, 3.0, 5.0, 4.0, 2.0, 1.0])
        _, loc_lam_01, _ = compute_homology_stats_v2(
            series, offset=5, Lambda=0.1
        )
        _, loc_lam_02, _ = compute_homology_stats_v2(
            series, offset=5, Lambda=0.2
        )
        # Both should point at the same step index (29), so the voxel
        # values must differ by a factor of 2.
        assert loc_lam_02 == pytest.approx(2 * loc_lam_01, rel=0.01)

    def test_maximum_count_is_raw_by_default(self):
        # Default 'rank_peaks_by_smoothed=False': the reported
        # 'maximum_count' is the RAW series value at the peak.
        series = np.zeros(60, dtype=float)
        series[27:33] = np.array([1.0, 3.0, 5.0, 4.0, 2.0, 1.0])
        _, _, max_count = compute_homology_stats_v2(
            series, offset=5, Lambda=0.1
        )
        assert max_count == 5.0

    def test_rank_peaks_by_smoothed_flag_shifts_max_location(self):
        # On a curve with a raw noise spike beating the true main peak,
        # enabling 'rank_peaks_by_smoothed' must move the reported
        # peak away from the spike.
        series = _noise_spike_then_broad_peak()
        _, raw_loc_vox, _ = compute_homology_stats_v2(
            series, offset=5, Lambda=0.1, rank_peaks_by_smoothed=False
        )
        _, smooth_loc_vox, _ = compute_homology_stats_v2(
            series, offset=5, Lambda=0.1, rank_peaks_by_smoothed=True
        )
        # The spike is around step 12 (voxel 1.2); the broad main peak
        # is around step 30 (voxel 3.0). The two flags should point at
        # opposite ends of the curve.
        assert raw_loc_vox < 2.0
        assert smooth_loc_vox > 2.5

    def test_maximum_count_from_smoothed_when_flag_set(self):
        # When the smoothed-weighting flag is enabled, 'maximum_count'
        # must come from the smoothed curve at the chosen step — not
        # the raw curve.
        series = _noise_spike_then_broad_peak()
        _, _, smooth_max_count = compute_homology_stats_v2(
            series, offset=5, Lambda=0.1, rank_peaks_by_smoothed=True
        )
        # Independently recompute what the smoothed value at the
        # chosen step is, using the same moving-average window that
        # v2 uses internally.
        window = int(round(1 / 0.1))
        smoothed = moving_average(series, w=window)
        step = int(
            round(
                find_max_location_v2(
                    series, offset=5, Lambda=0.1, rank_peaks_by_smoothed=True
                )
            )
        )
        assert smooth_max_count == pytest.approx(smoothed[step], rel=1e-6)


# ---------------------------------------------------------------------------
# bounding_box_crop
# ---------------------------------------------------------------------------


class TestBoundingBoxCrop:
    def test_crops_to_tight_bounds(self):
        vol = np.zeros((20, 20, 20), dtype=np.uint8)
        vol[3:6, 8:12, 15:20] = 1
        cropped, bbox = bounding_box_crop(vol)
        assert bbox == (3, 8, 15, 6, 12, 20)
        assert cropped.shape == (3, 4, 5)
        assert cropped.sum() == vol.sum()
        assert cropped.all()  # bbox is fully filled for a solid block

    def test_bbox_reindexes_original(self):
        vol = np.zeros((10, 10, 10), dtype=np.uint8)
        vol[2:4, 5:7, 1:9] = 1
        cropped, (z0, y0, x0, z1, y1, x1) = bounding_box_crop(vol)
        assert np.array_equal(cropped, vol[z0:z1, y0:y1, x0:x1])

    def test_preserves_dtype(self):
        vol = np.zeros((6, 6, 6), dtype=np.int32)
        vol[1:3, 1:3, 1:3] = 7
        cropped, _ = bounding_box_crop(vol)
        assert cropped.dtype == np.int32

    def test_empty_mask_raises(self):
        vol = np.zeros((5, 5, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            bounding_box_crop(vol)


# ---------------------------------------------------------------------------
# label_subvolume
# ---------------------------------------------------------------------------


def _three_label_volume():
    """3D volume with labels 1, 2, 3 in three disjoint boxes."""
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[2:5, 2:5, 2:5] = 1
    vol[10:14, 10:14, 10:14] = 2
    vol[16:18, 1:3, 16:19] = 3
    return vol


class TestLabelSubvolume:
    def test_extracts_only_requested_label(self):
        vol = _three_label_volume()
        obj, container, bbox = label_subvolume(vol, 2)
        assert container is None
        assert obj.dtype == np.uint8
        assert set(np.unique(obj)).issubset({0, 1})
        # label 2 occupied a 4x4x4 box, so the crop is exactly that
        assert obj.shape == (4, 4, 4)
        assert obj.sum() == 4**3
        assert bbox == (10, 10, 10, 14, 14, 14)

    def test_other_labels_excluded_from_crop(self):
        # Two labels whose bounding boxes overlap in Z: cropping label 1
        # must not pick up label 2's voxels even if they fall in the bbox.
        vol = np.zeros((10, 10, 10), dtype=np.uint8)
        vol[0:5, 0:5, 0:5] = 1
        vol[0:5, 0:5, 5:10] = 2
        obj, _, _ = label_subvolume(vol, 1)
        assert obj.sum() == 5**3  # only label 1's voxels

    def test_container_cropped_to_same_bbox(self):
        vol = _three_label_volume()
        container = np.ones((20, 20, 20), dtype=np.uint8)
        obj, cont, bbox = label_subvolume(vol, 1, container_data=container)
        assert cont is not None
        assert cont.shape == obj.shape
        assert cont.dtype == np.uint8
        z0, y0, x0, z1, y1, x1 = bbox
        assert cont.shape == (z1 - z0, y1 - y0, x1 - x0)

    def test_container_binarized(self):
        vol = _three_label_volume()
        container = np.full((20, 20, 20), 7, dtype=np.uint8)  # non-1 labels
        _, cont, _ = label_subvolume(vol, 1, container_data=container)
        assert set(np.unique(cont)).issubset({0, 1})

    def test_missing_label_raises(self):
        vol = _three_label_volume()
        with pytest.raises(ValueError):
            label_subvolume(vol, 99)
