"""
Sample data for napari-persistent-homology.

Ships a small 3D binary segmentation of mitochondrial cristae (from FIB-SEM
imaging) to be able to try the persistent-homology analysis without supplying
ones own data first. Registered in napari.yaml under contributions.sample_data
and appears in napari under File > Open Sample > Persistent Homology 3D.
"""

from __future__ import annotations

from importlib.resources import files


def load_cristae_binary_mask_3d():
    """
    Load the bundled 3D cristae binary mask as a Labels layer.

    The dataset is a uint8 array of shape (114, 163, 234) with values in
    {0, 1} — a binary segmentation of mitochondrial cristae from a 3D FIB-SEM
    volume, taken from the original research code by Wang et al.
    """
    import numpy as np

    data_path = files('napari_persistent_homology.data').joinpath(
        'cristae_binary_mask_3d.npy'
    )
    with data_path.open('rb') as f:
        data = np.load(f)
    return [(data, {'name': 'Cristae binary mask 3D'}, 'labels')]
