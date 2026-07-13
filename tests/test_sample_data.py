import numpy as np

from napari_persistent_homology._sample_data import (
    load_cristae_binary_mask_3d,
    load_cristae_multi_label_mask_3d,
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


def test_load_cristae_multi_label_mask_3d_returns_image_and_labels():
    layer_data_list = load_cristae_multi_label_mask_3d()
    assert isinstance(layer_data_list, list)
    assert len(layer_data_list) == 2

    (image, image_meta, image_type), (labels, labels_meta, labels_type) = (
        layer_data_list
    )
    assert image_type == 'image'
    assert labels_type == 'labels'
    assert image.shape == labels.shape
    assert image.ndim == 3
    assert image.dtype == np.uint8
    assert labels.dtype == np.uint8
    # 5 labelled objects + background
    assert set(np.unique(labels)) == {0, 1, 2, 3, 4, 5}
    assert image_meta['name'] == 'Cristae image 3D'
    assert labels_meta['name'] == 'Cristae multi-label mask 3D'


def test_load_cristae_multi_label_mask_3d_adds_both_layers_to_viewer(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    (image, image_meta, _), (labels, labels_meta, _) = (
        load_cristae_multi_label_mask_3d()
    )
    viewer.add_image(image, **image_meta)
    viewer.add_labels(labels, **labels_meta)
    assert len(viewer.layers) == 2
    assert viewer.layers[0].name == 'Cristae image 3D'
    assert viewer.layers[1].name == 'Cristae multi-label mask 3D'
