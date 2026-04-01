import numpy as np

from napari_persistent_homology.ph_functions import (
    compute_FWHM,
    compute_homology_stats,
    find_max_location,
    gaussian_average,
    hole_count,
    holes_count_internal_object,
    object_count,
    persistent_homology_dilation,
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
        assert 8 <= fwhm <= 16

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
        vol[2:5, 9:11, 9:11] = 1  # blob 1
        vol[15:18, 9:11, 9:11] = (
            1  # blob 2 — far apart, won't merge in few steps
        )

        # Use blobs that ARE close so they do merge
        vol2 = np.zeros((20, 20, 20), dtype=np.uint8)
        vol2[8:10, 9:11, 9:11] = 1  # blob 1
        vol2[12:14, 9:11, 9:11] = 1  # blob 2 — 2 voxels apart

        obj, _ = persistent_homology_dilation(
            vol2, max_steps=10, Lambda=0.5, Connectivity=26
        )
        assert obj[-1] == 1  # merged into one
