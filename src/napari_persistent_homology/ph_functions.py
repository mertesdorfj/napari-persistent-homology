"""
Extracting Mitochondrial Cristae Characteristics from 3D Focused Ion Beam Scanning Electron Microscopy Data

Chenhao Wang, Leif Østergaard, Stine Hasselholt, Jon Sporring

https://doi.org/10.1101/2022.11.08.515664
"""

# Imports
import os

import cc3d
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
from tqdm import tqdm

from .subpixel_morphology import (
    subpixel_dilation_3D,
    subpixel_erosion_3D,
)

###########################################################################################
# counting algorithm for objects and holes within a binary image volume


def object_count(volume_to_count, connectivity=8):
    object_num = cc3d.connected_components(
        volume_to_count, connectivity=connectivity
    ).max()
    return object_num


def hole_count(volume_to_count, connectivity=8):
    """
    always subtracts by 1 at the end because the background makes up 1 connected_component,
    after inversing the volume with holes.
    """

    total_holes = np.ones(volume_to_count.shape).astype(
        np.uint8
    ) - volume_to_count.astype(np.uint8)
    hole_num = cc3d.connected_components(
        total_holes, connectivity=connectivity
    ).max()

    return hole_num - 1


def holes_count_internal_object(
    internal_object, container_object, connectivity=8
):
    """
    for use with measurement of internal objects
    """
    internal_object_filtered = internal_object * container_object
    total_holes = np.ones(internal_object_filtered.shape).astype(
        np.uint8
    ) - internal_object_filtered.astype(np.uint8)
    hole_num = cc3d.connected_components(
        total_holes, connectivity=connectivity
    ).max()

    return hole_num - 1


###########################################################################################
# persistent homology functions


def persistent_homology_dilation(
    segmented_object_volume, max_steps=100, Lambda=0.1, Connectivity=26
):
    """
    Dilation version, returns the count curves for both objects and holes,
    but usually we only use the count curve for holes.
    """

    t = Lambda
    full_round = int(np.round(np.ceil(1 / Lambda)))
    object_count_result = []
    hole_count_result = []

    segmented_object_volume = segmented_object_volume.astype(np.uint8)

    # adds padding to avoid boundary effects, where padding size is (dist_limit + 1) on
    # 6 sides to make sure it is never touched. Therefore 2 * (dist_limit + 1) for each dimension
    dist_limit = int(np.round(np.ceil(max_steps * Lambda)))

    segmented_object_volume_padded = np.zeros(
        [
            segmented_object_volume.shape[0] + 2 * (dist_limit + 1),
            segmented_object_volume.shape[1] + 2 * (dist_limit + 1),
            segmented_object_volume.shape[2] + 2 * (dist_limit + 1),
        ],
        dtype=np.uint8,
    )

    segmented_object_volume_padded[
        (dist_limit + 1) : -(dist_limit + 1),
        (dist_limit + 1) : -(dist_limit + 1),
        (dist_limit + 1) : -(dist_limit + 1),
    ] = segmented_object_volume

    segmented_object_volume_padded = segmented_object_volume_padded.astype(
        np.uint8
    )

    # initial count at step 0
    object_num_current = object_count(
        segmented_object_volume_padded, connectivity=Connectivity
    )
    hole_num_current = hole_count(
        segmented_object_volume_padded, connectivity=Connectivity
    )
    object_count_result.append(object_num_current)
    hole_count_result.append(hole_num_current)

    segmented_object_volume_padded_dilated = segmented_object_volume_padded

    for step in tqdm(range(max_steps)):
        # subpixel dilation
        segmented_object_volume_padded_dilated = subpixel_dilation_3D(
            segmented_object_volume_padded_dilated, t, Lambda
        )

        # binarization for counting
        segmented_object_volume_padded_dilated_binarized = (
            segmented_object_volume_padded_dilated > 0.5
        )
        segmented_object_volume_padded_dilated_binarized = (
            segmented_object_volume_padded_dilated_binarized.astype(np.uint8)
        )

        # counting results
        object_num_current = object_count(
            segmented_object_volume_padded_dilated_binarized,
            connectivity=Connectivity,
        )
        hole_num_current = hole_count(
            segmented_object_volume_padded_dilated_binarized,
            connectivity=Connectivity,
        )
        object_count_result.append(object_num_current)
        hole_count_result.append(hole_num_current)

        # reset the subpixel dilation to start from binary image
        if (step + 1) % (5 * full_round) == 0:
            segmented_object_volume_padded_dilated = (
                segmented_object_volume_padded_dilated_binarized
            )

    return np.array(object_count_result), np.array(hole_count_result)


def persistent_homology_dilation_internal_object(
    segmented_object_volume,
    container_object_volume,
    max_steps=100,
    Lambda=0.1,
    Connectivity=26,
):
    """
    Dilation version for measurement of internal objects contained within other objects,
    returns the count curves for both objects and holes,
    but usually we only use the count curve for holes.
    """

    t = Lambda
    full_round = int(np.round(np.ceil(1 / Lambda)))
    object_count_result = []
    hole_count_result = []

    segmented_object_volume = segmented_object_volume.astype(np.uint8)
    container_object_volume = container_object_volume.astype(np.uint8)

    # adds padding to avoid boundary effects, where padding size is (dist_limit + 1) on
    # 6 sides to make sure it is never touched. Therefore 2 * (dist_limit + 1) for each dimension
    dist_limit = int(np.round(np.ceil(max_steps * Lambda)))

    segmented_object_volume_padded = np.zeros(
        [
            segmented_object_volume.shape[0] + 2 * (dist_limit + 1),
            segmented_object_volume.shape[1] + 2 * (dist_limit + 1),
            segmented_object_volume.shape[2] + 2 * (dist_limit + 1),
        ],
        dtype=np.uint8,
    )
    segmented_object_volume_padded[
        (dist_limit + 1) : -(dist_limit + 1),
        (dist_limit + 1) : -(dist_limit + 1),
        (dist_limit + 1) : -(dist_limit + 1),
    ] = segmented_object_volume
    segmented_object_volume_padded = segmented_object_volume_padded.astype(
        np.uint8
    )

    container_object_volume_padded = np.zeros(
        [
            container_object_volume.shape[0] + 2 * (dist_limit + 1),
            container_object_volume.shape[1] + 2 * (dist_limit + 1),
            container_object_volume.shape[2] + 2 * (dist_limit + 1),
        ],
        dtype=np.uint8,
    )
    container_object_volume_padded[
        (dist_limit + 1) : -(dist_limit + 1),
        (dist_limit + 1) : -(dist_limit + 1),
        (dist_limit + 1) : -(dist_limit + 1),
    ] = container_object_volume
    container_object_volume_padded = container_object_volume_padded.astype(
        np.uint8
    )

    # initial count at step 0
    object_num_current = object_count(
        segmented_object_volume_padded, connectivity=Connectivity
    )
    hole_num_current = holes_count_internal_object(
        segmented_object_volume_padded,
        container_object_volume_padded,
        connectivity=Connectivity,
    )
    object_count_result.append(object_num_current)
    hole_count_result.append(hole_num_current)

    segmented_object_volume_padded_dilated = segmented_object_volume_padded

    for step in tqdm(range(max_steps)):
        # subpixel dilation
        segmented_object_volume_padded_dilated = subpixel_dilation_3D(
            segmented_object_volume_padded_dilated, t, Lambda
        )

        # binarization for counting
        segmented_object_volume_padded_dilated_binarized = (
            segmented_object_volume_padded_dilated > 0.5
        )
        segmented_object_volume_padded_dilated_binarized = (
            segmented_object_volume_padded_dilated_binarized.astype(np.uint8)
        )

        # counting results
        object_num_current = object_count(
            segmented_object_volume_padded_dilated_binarized,
            connectivity=Connectivity,
        )
        hole_num_current = holes_count_internal_object(
            segmented_object_volume_padded_dilated_binarized,
            container_object_volume_padded,
            connectivity=Connectivity,
        )
        object_count_result.append(object_num_current)
        hole_count_result.append(hole_num_current)

        # reset the subpixel dilation to start from binary image
        if (step + 1) % (5 * full_round) == 0:
            segmented_object_volume_padded_dilated = (
                segmented_object_volume_padded_dilated_binarized
            )

    return np.array(object_count_result), np.array(hole_count_result)


def persistent_homology_erosion(
    segmented_object_volume, max_steps=100, Lambda=0.1, Connectivity=26
):
    """
    Erosion version, returns the count curves for both objects and holes,
    but usually we only use the count curve for objects.
    """

    t = Lambda
    full_round = int(np.round(np.ceil(1 / Lambda)))
    object_count_result = []
    hole_count_result = []

    segmented_object_volume = segmented_object_volume.astype(np.uint8)

    # adds padding to avoid boundary effects, where padding size is (dist_limit + 1) on
    # 6 sides to make sure it is never touched. Therefore 2 * (dist_limit + 1) for each dimension
    dist_limit = int(np.round(np.ceil(max_steps * Lambda)))

    segmented_object_volume_padded = np.zeros(
        [
            segmented_object_volume.shape[0] + 2 * (dist_limit + 1),
            segmented_object_volume.shape[1] + 2 * (dist_limit + 1),
            segmented_object_volume.shape[2] + 2 * (dist_limit + 1),
        ],
        dtype=np.uint8,
    )

    segmented_object_volume_padded[
        (dist_limit + 1) : -(dist_limit + 1),
        (dist_limit + 1) : -(dist_limit + 1),
        (dist_limit + 1) : -(dist_limit + 1),
    ] = segmented_object_volume

    segmented_object_volume_padded = segmented_object_volume_padded.astype(
        np.uint8
    )

    # initial count at step zero
    object_num_current = object_count(
        segmented_object_volume_padded, connectivity=Connectivity
    )
    hole_num_current = hole_count(
        segmented_object_volume_padded, connectivity=Connectivity
    )
    object_count_result.append(object_num_current)
    hole_count_result.append(hole_num_current)

    segmented_object_volume_padded_eroded = segmented_object_volume_padded

    for step in tqdm(range(max_steps)):
        # subpixel erosion
        segmented_object_volume_padded_eroded = subpixel_erosion_3D(
            segmented_object_volume_padded_eroded, t, Lambda
        )

        # binarization for counting
        segmented_object_volume_padded_eroded_binarized = (
            segmented_object_volume_padded_eroded > 0.5
        )
        segmented_object_volume_padded_eroded_binarized = (
            segmented_object_volume_padded_eroded_binarized.astype(np.uint8)
        )

        # counting results
        object_num_current = object_count(
            segmented_object_volume_padded_eroded_binarized,
            connectivity=Connectivity,
        )
        hole_num_current = hole_count(
            segmented_object_volume_padded_eroded_binarized,
            connectivity=Connectivity,
        )
        object_count_result.append(object_num_current)
        hole_count_result.append(hole_num_current)

        # reset the subpixel erosion to start from binary image
        if (step + 1) % (5 * full_round) == 0:
            segmented_object_volume_padded_eroded = (
                segmented_object_volume_padded_eroded_binarized
            )

    return np.array(object_count_result), np.array(hole_count_result)


###########################################################################################
# feature extraction from count curves and performing statistics


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def gaussian_average(x, sigma=1):
    return scipy.ndimage.gaussian_filter1d(x, sigma=sigma)


def find_max_location(series, offset=5):
    # finds argmax considering the offset
    if np.max(series[offset:]) == 0:
        max_loc = 0 + 5
    else:
        max_loc = (
            len(series[offset:][::-1])
            - np.argmax(series[offset:][::-1])
            - 1
            + offset
        )

    return max_loc


def compute_FWHM(series, offset=5):

    max_location = find_max_location(series, offset=offset)
    maximum_count = series[max_location]
    half_maximum = maximum_count / 2

    indices_larger_than_half_max = np.sort(np.where(series >= half_maximum)[0])

    left_half = indices_larger_than_half_max[
        np.where(max_location >= indices_larger_than_half_max)
    ]
    right_half = indices_larger_than_half_max[
        np.where(max_location <= indices_larger_than_half_max)
    ]

    try:
        left_index = np.split(
            left_half, np.where(np.diff(left_half) != 1)[0] + 1
        )[-1][0]
    except IndexError:
        left_index = left_index[0]
    try:
        right_index = np.split(
            right_half, np.where(np.diff(right_half) != 1)[0] + 1
        )[0][-1]
    except IndexError:
        right_index = right_index[0]
    full_width_half_maximum = right_index - left_index

    return full_width_half_maximum


def compute_homology_stats(series_of_series, offset=5, SIGMA=5):
    """
    window = 0 or 1 is no filter
    window > 1 is moving average with the specified window size.

    output is a list containing [FWHM_collection,
                                 maximum_count_collection,
                                 max_location_collection]
    """
    FWHM_collection = []
    max_location_collection = []
    maximum_count_collection = []
    for index in range(len(series_of_series)):
        series = series_of_series[index]
        series = gaussian_average(series, sigma=SIGMA)
        FWHM = compute_FWHM(series, offset)
        max_location = find_max_location(series, offset)
        maximum_count = series[max_location]

        FWHM_collection.append(FWHM)
        max_location_collection.append(max_location)
        maximum_count_collection.append(maximum_count)

    FWHM_collection = np.array(FWHM_collection)
    max_location_collection = np.array(max_location_collection)
    maximum_count_collection = np.array(maximum_count_collection)
    output = np.array(
        [FWHM_collection, maximum_count_collection, max_location_collection]
    )

    return output


def plot_hist_fixed_width_save(
    x_data,
    drop_outlier_percentile_lower,
    drop_outlier_percentile_upper,
    color,
    plot_title,
    x_title,
    bin_count,
    binwidth,
    bin_min,
    bin_max,
    save_folder,
    save_name='reference.png',
    horizontal_pos=0.925,
    vertical_pos=0.8,
    title_font=20,
    axis_font=18,
    legend_font=16,
):

    x_mean = np.round(np.mean(x_data), 1)
    x_median = np.round(np.median(x_data), 1)
    x_std = np.round(np.std(x_data), 1)
    x_MAD = np.round(stats.median_absolute_deviation(x_data, scale=1), 1)

    stats_text = (
        'x_mean = '
        + str(x_mean)
        + '\n'
        + 'x_std = '
        + str(x_std)
        + '\n'
        + 'x_median = '
        + str(x_median)
        + '\n'
        + 'x_mad = '
        + str(x_MAD)
    )

    x_data_sorted = np.sort(x_data)

    lower_bound = int(len(x_data_sorted) * drop_outlier_percentile_lower)
    upper_bound = len(x_data_sorted) - int(
        len(x_data_sorted) * (1 - drop_outlier_percentile_upper)
    )

    x_data_to_plot = x_data_sorted[lower_bound:upper_bound]

    fig, ax = plt.subplots(figsize=(9, 6))

    if (
        binwidth is None
        or bin_min is None
        or bin_max is None
        or bin_count is not None
    ):
        ax.hist(
            x_data_to_plot,
            bins=bin_count,
            color=color,
            edgecolor='black',
            linewidth=1.2,
        )
    else:
        ax.hist(
            x_data_to_plot,
            bins=np.arange(bin_min, bin_max + binwidth, binwidth),
            color=color,
            edgecolor='black',
            linewidth=1.2,
        )
    stats_text = (
        stats_text + '\n' + 'sample_size = ' + str(len(x_data_to_plot))
    )

    ax.set_ylabel('Count', fontsize=axis_font)
    ax.set_xlabel(x_title, fontsize=axis_font)
    ax.tick_params(labelsize=axis_font)
    fig.figtext(
        horizontal_pos, vertical_pos, s=stats_text, fontsize=legend_font
    )
    fig.savefig(os.path.join(save_folder, save_name), bbox_inches='tight')
    return fig


def plot_scatter(
    x_data,
    y_data,
    alpha,
    x_label,
    y_label,
    main_label,
    save_folder,
    horizontal_pos=0.925,
    vertical_pos=0.8,
    title_font=18,
    axis_font=18,
    legend_font=16,
):

    pearson_score, p_val = np.round(scipy.stats.pearsonr(x_data, y_data), 4)
    text = 'pearson = ' + str(pearson_score) + '\n' + 'p_value = ' + str(p_val)
    text = text + '\n' + 'sample_size = ' + str(len(x_data))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x_data, y_data, alpha=alpha)
    ax.set_xlabel(x_label, fontsize=axis_font)
    ax.set_ylabel(y_label, fontsize=axis_font)
    ax.tick_params(labelsize=axis_font)
    fig.figtext(horizontal_pos, vertical_pos, s=text, fontsize=legend_font)
    fig.savefig(
        os.path.join(save_folder, main_label + '.png'), bbox_inches='tight'
    )
    return fig


def find_boundary_mitochondria_id(img_volume, seg_volume):

    out_of_bounds_volume = img_volume == 0

    out_of_bounds_volume = scipy.ndimage.morphology.binary_dilation(
        out_of_bounds_volume, iterations=3
    )

    boundary_volume = out_of_bounds_volume * (img_volume > 0)
    boundary_coors = np.where(boundary_volume == 1)

    del boundary_volume
    del out_of_bounds_volume

    seg_cc = cc3d.connected_components(seg_volume)

    boundary_mito_id = seg_cc[boundary_coors]

    return np.unique(boundary_mito_id)


def plot_all_curves(results, sigma, destination_path):

    for i in tqdm(range(len(results))):
        curve = results[i]
        curve_smooth = gaussian_average(curve, sigma=sigma)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(curve)
        ax.plot(curve_smooth)

        FWHM = np.round(compute_FWHM(curve_smooth, offset=5), 4)
        max_location = np.round(find_max_location(curve_smooth, offset=5), 4)

        maximum_count = curve_smooth[max_location]

        half_maximum = maximum_count / 2
        indices_larger_than_half_max = np.sort(
            np.where(curve_smooth >= half_maximum)[0]
        )
        left_half = indices_larger_than_half_max[
            np.where(max_location >= indices_larger_than_half_max)
        ]
        right_half = indices_larger_than_half_max[
            np.where(max_location <= indices_larger_than_half_max)
        ]
        try:
            left_index = np.split(
                left_half, np.where(np.diff(left_half) != 1)[0] + 1
            )[-1][0]
        except IndexError:
            left_index = left_index[0]
        try:
            right_index = np.split(
                right_half, np.where(np.diff(right_half) != 1)[0] + 1
            )[0][-1]
        except IndexError:
            right_index = right_index[0]
        left_count = curve_smooth[left_index]
        right_count = curve_smooth[right_index]

        ax.scatter(
            [max_location, left_index, right_index],
            [maximum_count, left_count, right_count],
            c='red',
        )

        text = (
            'max_location = '
            + str(max_location)
            + '\n'
            + 'FWHM = '
            + str(FWHM)
        )
        ax.set_xlabel('rounds of morphology', fontsize=16)
        ax.set_ylabel('object count', fontsize=16)
        ax.tick_params(labelsize=16)
        ax.set_title('mito id = ' + str(i), fontsize=20)
        fig.figtext(0.65, 0.8, s=text, fontsize=16)
        fig.savefig(
            os.path.join(destination_path, 'mito id = ' + str(i) + '.png'),
            bbox_inches='tight',
        )
        plt.close(fig)
