import numpy as np
import pytest

from napari_persistent_homology.subpixel_morphology import (
    subpixel_dilation_3D,
    subpixel_erosion_3D,
)


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
    def test_dilate_then_erode_preserves_interior(self):
        # After dilation + erosion with the same t, the deep interior should
        # still be close to 1 (not a perfect inverse but should hold for interior)
        vol = solid_cube(size=20, blob_slice=slice(5, 15))
        dilated = subpixel_dilation_3D(vol, t=0.5, Lambda=0.1)
        recovered = subpixel_erosion_3D(dilated, t=0.5, Lambda=0.1)
        assert recovered[10, 10, 10] > 0.9
