import numpy as np

from napari_persistent_homology._sample_data import (
    load_cristae_binary_mask_3d,
)


def test_load_cristae_binary_mask_3d_returns_labels_layer_data():
    layer_data_list = load_cristae_binary_mask_3d()
    assert isinstance(layer_data_list, list)
    assert len(layer_data_list) == 1

    data, meta, layer_type = layer_data_list[0]
    assert isinstance(data, np.ndarray)
    assert data.ndim == 3
    assert set(np.unique(data)).issubset({0, 1})
    assert layer_type == 'labels'
    assert meta['name'] == 'Cristae binary mask 3D'


def test_load_cristae_binary_mask_3d_adds_to_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    ((data, meta, _),) = load_cristae_binary_mask_3d()
    viewer.add_labels(data, **meta)
    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == 'Cristae binary mask 3D'
