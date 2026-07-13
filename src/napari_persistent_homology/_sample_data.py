"""
Sample data for napari-persistent-homology.

Ships two bundled 3D FIB-SEM samples of mitochondrial cristae so users can
try the persistent-homology analysis without supplying their own data:

* Cristae binary mask 3D — a single Labels layer (uint8 0/1 segmentation)
  taken from the original research code by Wang et al.
* Cristae multi-label mask 3D — an Image + Labels layer pair with 5 manually
  labelled cristae (IDs 1–5), useful for per-object workflows.

Both are registered in napari.yaml under contributions.sample_data and appear
under File > Open Sample > Persistent Homology 3D.
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


def load_cristae_multi_label_mask_3d():
    """
    Load the bundled 3D EM image + labelled cristae mask as two layers.

    Returns an Image layer (uint8, (311, 183, 195) FIB-SEM crop) and a
    matching Labels layer with 5 manually-labelled cristae (IDs 1-5 plus
    background 0). Useful for exercising the plugin on per-object analyses
    (e.g. selecting a single crista from the Labels layer before running).
    """
    import numpy as np

    root = files('napari_persistent_homology.data')
    image_path = root.joinpath('cristae_image_3d.npy')
    label_path = root.joinpath('cristae_multi_label_mask_3d.npy')
    with image_path.open('rb') as f:
        image = np.load(f)
    with label_path.open('rb') as f:
        labels = np.load(f)
    return [
        (image, {'name': 'Cristae image 3D'}, 'image'),
        (labels, {'name': 'Cristae multi-label mask 3D'}, 'labels'),
    ]
