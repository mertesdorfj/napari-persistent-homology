import numpy as np
import pytest

from napari_persistent_homology.subpixel_morphology import (
    subpixel_dilation_2D,
    subpixel_dilation_3D,
    subpixel_erosion_2D,
    subpixel_erosion_3D,
)


def solid_square(size=10, blob_slice=slice(3, 7)):
    """2D float32 field with a solid square of 1s in the centre."""
    vol = np.zeros((size, size), dtype=np.float32)
    vol[blob_slice, blob_slice] = 1.0
    return vol


class TestSubpixelDilation2D:
    def test_output_shape_and_dtype(self):
        vol = solid_square()
        result = subpixel_dilation_2D(vol, t=0.1, Lambda=0.1)
        assert result.shape == vol.shape
        assert result.dtype == np.float32

    def test_values_in_range(self):
        result = subpixel_dilation_2D(solid_square(), t=0.5, Lambda=0.1)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_expands_into_neighbours(self):
        vol = np.zeros((10, 10), dtype=np.float32)
        vol[5, 5] = 1.0
        result = subpixel_dilation_2D(vol, t=0.5, Lambda=0.1)
        assert result[5, 6] > 0.0
        assert result[4, 5] > 0.0

    def test_all_ones_stays_saturated(self):
        result = subpixel_dilation_2D(np.ones((6, 6), np.float32), 0.5, 0.1)
        assert np.allclose(result, 1.0)

    def test_all_zeros_stays_zero(self):
        result = subpixel_dilation_2D(np.zeros((6, 6), np.float32), 0.5, 0.1)
        assert np.allclose(result, 0.0)


class TestSubpixelErosion2D:
    def test_output_shape_and_dtype(self):
        vol = solid_square()
        result = subpixel_erosion_2D(vol, t=0.1, Lambda=0.1)
        assert result.shape == vol.shape
        assert result.dtype == np.float32

    def test_values_in_range(self):
        result = subpixel_erosion_2D(solid_square(), t=0.5, Lambda=0.1)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shrinks_at_the_boundary(self):
        # Erosion retreats the front: an edge voxel of the square loses value.
        vol = solid_square()
        result = subpixel_erosion_2D(vol, t=0.5, Lambda=0.1)
        assert result[3, 3] < 1.0  # corner of the square

    def test_all_zeros_stays_zero(self):
        result = subpixel_erosion_2D(np.zeros((6, 6), np.float32), 0.5, 0.1)
        assert np.allclose(result, 0.0)


def solid_cube(size=10, blob_slice=slice(3, 7)):
    """3D float32 volume with a solid cube of 1s in the centre."""
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[blob_slice, blob_slice, blob_slice] = 1.0
    return vol


class TestSubpixelDilation3D:
    def test_output_shape(self):
        vol = solid_cube()
        result = subpixel_dilation_3D(vol, t=0.1, Lambda=0.1)
        assert result.shape == vol.shape

    def test_output_dtype_float32(self):
        vol = solid_cube()
        result = subpixel_dilation_3D(vol, t=0.1, Lambda=0.1)
        assert result.dtype == np.float32

    def test_values_in_range(self):
        vol = solid_cube()
        result = subpixel_dilation_3D(vol, t=0.5, Lambda=0.1)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_expands_into_neighbours(self):
        # A single foreground voxel: its neighbours should gain value after dilation
        vol = np.zeros((10, 10, 10), dtype=np.float32)
        vol[5, 5, 5] = 1.0
        result = subpixel_dilation_3D(vol, t=0.5, Lambda=0.1)
        assert result[5, 5, 6] > 0.0
        assert result[5, 5, 4] > 0.0
        assert result[5, 6, 5] > 0.0

    def test_foreground_stays_at_one(self):
        # Interior foreground voxels should remain at 1 after dilation
        vol = solid_cube()
        result = subpixel_dilation_3D(vol, t=0.5, Lambda=0.1)
        assert result[5, 5, 5] == pytest.approx(1.0)

    def test_no_change_on_all_ones(self):
        # Dilation of a fully-foreground volume should be a no-op
        vol = np.ones((8, 8, 8), dtype=np.float32)
        result = subpixel_dilation_3D(vol, t=0.5, Lambda=0.1)
        np.testing.assert_array_almost_equal(result, vol)

    def test_no_change_on_all_zeros(self):
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        result = subpixel_dilation_3D(vol, t=0.5, Lambda=0.1)
        np.testing.assert_array_almost_equal(result, vol)


class TestSubpixelErosion3D:
    def test_output_shape(self):
        vol = solid_cube()
        result = subpixel_erosion_3D(vol, t=0.1, Lambda=0.1)
        assert result.shape == vol.shape

    def test_output_dtype_float32(self):
        vol = solid_cube()
        result = subpixel_erosion_3D(vol, t=0.1, Lambda=0.1)
        assert result.dtype == np.float32

    def test_values_in_range(self):
        vol = solid_cube()
        result = subpixel_erosion_3D(vol, t=0.5, Lambda=0.1)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shrinks_boundary(self):
        # Boundary voxels of the object should lose value after erosion
        vol = solid_cube()
        result = subpixel_erosion_3D(vol, t=0.5, Lambda=0.1)
        assert result[3, 3, 3] < 1.0  # corner of blob

    def test_interior_preserved(self):
        # Deep interior voxels should remain at 1 after a small erosion step
        vol = solid_cube(size=20, blob_slice=slice(4, 16))
        result = subpixel_erosion_3D(vol, t=0.1, Lambda=0.1)
        assert result[10, 10, 10] == pytest.approx(1.0)

    def test_no_change_on_all_zeros(self):
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        result = subpixel_erosion_3D(vol, t=0.5, Lambda=0.1)
        np.testing.assert_array_almost_equal(result, vol)

    def test_no_change_on_all_ones(self):
        # Erosion of a fully-foreground volume should be a no-op
        vol = np.ones((8, 8, 8), dtype=np.float32)
        result = subpixel_erosion_3D(vol, t=0.5, Lambda=0.1)
        np.testing.assert_array_almost_equal(result, vol)


class TestDilationErosionInverse:
    """
    Tests for the morphological closing: dilate then erode by the same t.

    For a solid blob with no holes, closing should approximately recover the
    original shape. Because this is a finite-difference upwind PDE scheme
    (not exact set-theoretic morphology), the recovery is approximate —
    boundary-zone voxels retain some numerical diffusion. The tests below
    check the *direction* of the effect at multiple sample points rather
    than insisting on a perfect identity.
    """

    # Sample points grouped by which region they occupy.
    # Cube spans indices 5..14 on every axis (slice(5, 15)).
    DEEP_INTERIOR = [(10, 10, 10), (8, 11, 9), (9, 9, 9)]
    ORIG_BOUNDARY_FACE = [(5, 10, 10), (14, 10, 10), (10, 5, 10), (10, 10, 14)]
    JUST_OUTSIDE_FACE = [(4, 10, 10), (15, 10, 10), (10, 4, 10), (10, 10, 15)]
    FAR_BACKGROUND = [(0, 0, 0), (19, 19, 19), (0, 19, 0)]

    def test_dilation_intermediate_state(self):
        """
        After dilation only: interior stays 1, just-outside voxels
        gain value, and far-background voxels stay 0.
        """
        vol = solid_cube(size=20, blob_slice=slice(5, 15))
        dilated = subpixel_dilation_3D(vol, t=0.5, Lambda=0.1)

        for p in self.DEEP_INTERIOR:
            assert dilated[p] == pytest.approx(1.0, abs=0.05), (
                f'interior {p} should remain ~1.0 after dilation '
                f'(got {dilated[p]:.3f})'
            )
        for p in self.JUST_OUTSIDE_FACE:
            assert dilated[p] > 0.2, (
                f'dilation should activate just-outside voxel {p} '
                f'(got {dilated[p]:.3f})'
            )
        for p in self.FAR_BACKGROUND:
            assert dilated[p] == pytest.approx(0.0, abs=1e-3), (
                f'far-background {p} should stay ~0 after dilation '
                f'(got {dilated[p]:.3f})'
            )

    def test_closing_recovers_shape_approximately(self):
        """
        After dilation + erosion (closing): interior stays high,
        far-background stays zero, and the just-outside voxels that
        dilation activated come back down toward zero.
        """
        vol = solid_cube(size=20, blob_slice=slice(5, 15))
        dilated = subpixel_dilation_3D(vol, t=0.5, Lambda=0.1)
        recovered = subpixel_erosion_3D(dilated, t=0.5, Lambda=0.1)

        # Deep interior should remain near 1.0
        for p in self.DEEP_INTERIOR:
            assert recovered[p] > 0.9, (
                f'closing eroded interior {p} too much '
                f'(got {recovered[p]:.3f})'
            )

        # Original boundary loses some value due to numerical diffusion,
        # but should still be clearly in the foreground (> 0.5)
        for p in self.ORIG_BOUNDARY_FACE:
            assert recovered[p] > 0.5, (
                f'closing eroded original boundary {p} below 0.5 '
                f'(got {recovered[p]:.3f})'
            )

        # Just-outside voxels: erosion must reduce them (recovery direction).
        # They don't quite return to 0 due to numerical diffusion — but they
        # must clearly be lower than the post-dilation value.
        for p in self.JUST_OUTSIDE_FACE:
            assert recovered[p] < dilated[p] - 0.1, (
                f'erosion should reduce just-outside {p} '
                f'(dilated={dilated[p]:.3f}, recovered={recovered[p]:.3f})'
            )

        # Far background must stay essentially zero throughout
        for p in self.FAR_BACKGROUND:
            assert recovered[p] == pytest.approx(0.0, abs=1e-3), (
                f'far-background {p} drifted from 0 after closing '
                f'(got {recovered[p]:.3f})'
            )
