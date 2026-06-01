"""
Persistent-homology analysis of 3D binary segmentations.

This module is the analysis core of the plugin. It iteratively erodes or
dilates a binary volume by subpixel amounts (using
'subpixel_morphology.subpixel_*_3D'), counts how many connected components
and topological holes survive at each step, and extracts shape descriptors
(radius, FWHM) from the resulting count curves.

Three analysis modes:
* Function 'persistent_homology_erosion' — repeatedly shrinks the object;
  the object-count curve peaks at the typical half-thickness / radius.
* Function 'persistent_homology_dilation' — repeatedly grows the object;
  the hole-count curve peaks where most surfaces make contact.
* Function 'persistent_homology_dilation_internal_object' — same as dilation
  but the count is restricted to a container mask, for measuring spacing of
  objects inside a parent compartment (e.g. cristae inside a mitochondrion).
  This is the variant used in the paper, where the dilated mask is multiplied
  by the mitochondrion segmentation before hole counting.

Per the paper, the count-curve peak location can be interpreted as *half* of
the average minimum distance across the region being eroded or dilated:
half-thickness/radius for erosion, half the inter-object gap for dilation
(at the peak each surface has grown by the peak amount, so the gap is twice
that). The FWHM measures surface roughness / curvature of the same region —
rougher, more curved surfaces keep holes/objects alive over more rounds.

Each routine returns '(object_count_curve, hole_count_curve)'. The relevant
curve is then passed through function 'compute_homology_stats', which Gaussian-
smooths it and reports the peak location and full-width-at-half-maximum. Raw
peak/FWHM values are in subpixel-step units — divide by 'ceil(1 / Lambda)'
to convert to voxel units.

Source: Wang, Østergaard, Hasselholt, Sporring,
"A semi-automatic method for extracting mitochondrial cristae characteristics
from 3D focused ion beam scanning electron microscopy data",
Communications Biology 7:377 (2024),
https://doi.org/10.1038/s42003-024-06045-4
"""

import os

import cc3d
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
from tqdm import tqdm

from .subpixel_morphology import subpixel_dilation_3D, subpixel_erosion_3D

##############################################################################
# Counting components and holes
#
# Wrappers around 'cc3d.connected_components' that count either foreground
# components ("objects") or background components inside the volume ("holes").
# All three are called inside the persistent-homology loop after every
# subpixel morphology step.
##############################################################################


def object_count(volume_to_count, connectivity=8):
    """
    Count connected foreground components in a 3D binary volume.

    Parameters
    ----------
    volume_to_count: ndarray
        Binary 3D array (0 = background, non-zero = foreground).
    connectivity: int, optional
        Connectivity for 3D component labelling (6, 18, or 26). Default 8 is
        the 2D value and is silently treated as 6 in 3D by 'cc3d'; the
        plugin always passes 26 (full 3D neighbourhood).

    Returns
    -------
    int
        Number of distinct foreground components.
    """
    object_num = cc3d.connected_components(
        volume_to_count, connectivity=connectivity
    ).max()
    return object_num


def hole_count(volume_to_count, connectivity=8):
    """
    Count connected background components ("holes") in a 3D binary volume.

    Inverts the binary volume and counts connected components of the
    background. The outer background is one such component, so the final
    count is decremented by 1 — what is returned is therefore the number of
    enclosed holes only.

    Parameters
    ----------
    volume_to_count: ndarray
        Binary 3D array (0 = background, non-zero = foreground).
    connectivity: int, optional
        See function 'object_count'.

    Returns
    -------
    int
        Number of enclosed holes (background components minus the outer one).
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
    Count holes of an internal object, restricted to a container mask.

    Used by the "internal-spacing" analysis mode. The internal object is
    first masked by the container (voxels outside the container are dropped),
    then the holes of the resulting volume are counted as in function
    'hole_count'. The decrement by 1 again removes the outer background component.

    Parameters
    ----------
    internal_object: ndarray
        Binary 3D mask of the internal structures.
    container_object: ndarray
        Binary 3D mask of the parent compartment. Same shape as 'internal_object'.
    connectivity: int, optional
        See function 'object_count'.

    Returns
    -------
    int
        Number of enclosed holes inside the container.
    """
    internal_object_filtered = internal_object * container_object
    total_holes = np.ones(internal_object_filtered.shape).astype(
        np.uint8
    ) - internal_object_filtered.astype(np.uint8)
    hole_num = cc3d.connected_components(
        total_holes, connectivity=connectivity
    ).max()

    return hole_num - 1


##############################################################################
# Persistent-homology analysis functions
#
# All three analysis functions share the same skeleton:
#   1. Pad the input volume by 'max_steps * Lambda' voxels on each side so
#      morphological evolution never touches the boundary.
#   2. Record the initial object/hole counts at step 0.
#   3. Repeat 'max_steps' times: apply one subpixel dilation/erosion step
#      of size 'Lambda' voxels, binarise the result at 0.5, and count
#      objects and holes on the binarised volume.
#   4. Every '5 * ceil(1/Lambda)' steps, reset the working volume to its
#      binarised form to prevent numerical drift in the float field.
#
# Parameters (common to all three analysis functions):
#   segmented_object_volume: ndarray
#       Input 3D binary mask.
#   max_steps: int
#       Total number of subpixel morphology steps. Maximum measurable
#       distance is 'max_steps * Lambda' voxels.
#   Lambda: float
#       Subpixel step size in voxel-length units (0.1 = 10 steps per voxel).
#       Smaller = more accurate, but slower.
#   Connectivity: int
#       3D neighbour connectivity used by the component counters (6/18/26).
#   step_callback: callable, optional
#       If given, called as 'step_callback(current_step, max_steps)' after
#       each iteration — used by the widget to drive a progress bar.
#
# Returns:
#   (object_count_curve, hole_count_curve) — both 'ndarray' of length
#   'max_steps + 1' (index 0 is the initial state).
##############################################################################


def persistent_homology_dilation(
    segmented_object_volume,
    max_steps=100,
    Lambda=0.1,
    Connectivity=26,
    step_callback=None,
):
    """
    Iteratively dilate a binary volume and track object/hole counts.

    Used for object-spacing analysis: as the object grows, neighbouring
    regions merge and the surrounding holes close up. The peak of the
    returned hole-count curve corresponds to the characteristic spacing
    between objects.

    See the section header above for parameter and return-value details.
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

        if step_callback is not None:
            step_callback(step + 1, max_steps)

    return np.array(object_count_result), np.array(hole_count_result)


def persistent_homology_dilation_internal_object(
    segmented_object_volume,
    container_object_volume,
    max_steps=100,
    Lambda=0.1,
    Connectivity=26,
    step_callback=None,
):
    """
    Dilation with hole counting restricted to a container mask.

    Variant of function 'persistent_homology_dilation' for the internal-spacing
    mode: useful when the structures of interest are contained inside a parent
    compartment (e.g. cristae inside a mitochondrion). At every step the hole
    count is computed only inside 'container_object_volume' so that background
    voxels outside the container do not contribute.

    Parameters
    ----------
    segmented_object_volume: ndarray
        Binary 3D mask of the internal structures.
    container_object_volume: ndarray
        Binary 3D mask of the parent compartment. Same shape as 'segmented_object_volume'.

    Other parameters and return value are as in function 'persistent_homology_dilation'.
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

        if step_callback is not None:
            step_callback(step + 1, max_steps)

    return np.array(object_count_result), np.array(hole_count_result)


def persistent_homology_erosion(
    segmented_object_volume,
    max_steps=100,
    Lambda=0.1,
    Connectivity=26,
    step_callback=None,
):
    """
    Iteratively erode a binary volume and track object/hole counts.

    Used for object-radius / half-thickness analysis: as the object shrinks,
    thinner regions disappear first and the foreground splits into more
    components before vanishing. The peak of the returned object-count curve
    corresponds to the characteristic half-thickness (radius) of the
    structure; its FWHM measures surface roughness / curvature.

    See the section header above for parameter and return-value details.
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

        if step_callback is not None:
            step_callback(step + 1, max_steps)

    return np.array(object_count_result), np.array(hole_count_result)


##############################################################################
# Feature extraction from count curves
#
# Given a count curve produced by one of the persistent-homology analysis functions
# above, these routines smooth the curve and extract the peak location and
# full-width-at-half-maximum (FWHM). Peak location is interpreted as
# radius / spacing (in subpixel-step units); FWHM measures the spread of
# the distribution of feature sizes.
##############################################################################


def gaussian_average(x, sigma=1):
    """1D Gaussian smoothing of a curve with the given 'sigma'."""
    return scipy.ndimage.gaussian_filter1d(x, sigma=sigma)


def find_max_location(series, offset=5):
    """
    Index of the last occurrence of the maximum of 'series[offset:]'.

    The initial 'offset' samples are skipped because the count curve is most
    susceptible to noise there. The paper excludes the first five subpixel
    rounds — half of a full morphology round at 'Lambda = 0.1' — for this
    reason, which is the source of the default 'offset = 5'. When several
    samples share the maximum value, the right-most one is returned. If the
    whole post-offset series is zero, the function returns 'offset' as a
    fallback (the caller then treats the curve as degenerate).
    """
    if np.max(series[offset:]) == 0:
        max_loc = offset
    else:
        max_loc = (
            len(series[offset:][::-1])
            - np.argmax(series[offset:][::-1])
            - 1
            + offset
        )

    return max_loc


def compute_FWHM(series, offset=5):
    """
    Full-width at half-maximum of a count curve, around its peak.

    Finds the peak via function 'find_max_location', then walks outward to the
    contiguous block of samples that lie at or above half-maximum and
    returns 'right_index - left_index'. Result is in step units; divide by
    'ceil(1 / Lambda)' to convert to voxels.
    """
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

    # Keep only the contiguous run of above-half-max samples that contains the
    # peak (np.split breaks the index list wherever consecutive indices are
    # not adjacent).
    try:
        left_index = np.split(
            left_half, np.where(np.diff(left_half) != 1)[0] + 1
        )[-1][0]
    except IndexError:
        left_index = max_location
    try:
        right_index = np.split(
            right_half, np.where(np.diff(right_half) != 1)[0] + 1
        )[0][-1]
    except IndexError:
        right_index = max_location
    full_width_half_maximum = right_index - left_index

    return full_width_half_maximum


def compute_homology_stats(series_of_series, offset=5, SIGMA=3):
    """
    Smooth one or more count curves and extract peak + FWHM statistics.

    Parameters
    ----------
    series_of_series: sequence of ndarray
        One or more count curves (typically one per analysed object).
    offset: int, optional
        Initial samples to skip when locating the peak (passes through to
        functions 'find_max_location' / 'compute_FWHM'). Default = 5.
    SIGMA: float, optional
        Gaussian smoothing sigma applied to each curve before feature
        extraction. Default = 3.

    Returns
    -------
    ndarray, shape (3, N)
        Stacked statistics in the order [FWHM, maximum_count, max_location],
        with one column per input curve. Values for FWHM and max_location
        are in step units.
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


##############################################################################
# Plotting and batch-analysis helpers (from the original research code)
#
# These functions are not used by the napari widget itself — they are kept
# for reproducing the figures in the paper or run the pipeline as a script
# on many segmentations at once.
##############################################################################


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
    """
    Plot a histogram with mean/median/std/MAD annotation and save to disk.

    Trims the lower 'drop_outlier_percentile_lower' and the upper
    '1 - drop_outlier_percentile_upper' tails of 'x_data' before
    plotting. If 'bin_count' is given it is used directly; otherwise the
    bin edges are built from 'bin_min', 'bin_max', and 'binwidth'.
    The figure is saved as 'save_folder/save_name' and also returned.
    """
    x_mean = np.round(np.mean(x_data), 1)
    x_median = np.round(np.median(x_data), 1)
    x_std = np.round(np.std(x_data), 1)
    x_MAD = np.round(stats.median_abs_deviation(x_data, scale=1), 1)

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
    """
    Scatter plot 'y_data' vs 'x_data' with Pearson correlation annotation.

    The Pearson coefficient, its p-value, and the sample size are written into
    the figure. The image is saved as 'save_folder/(main_label + '.png')'
    and the figure object is also returned.
    """
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
    """
    Identify segmented objects that touch the image-volume boundary.

    Builds a thin mask near the edge of the imaged region (voxels where
    'img_volume == 0' dilated by 3) and looks up which connected
    components of 'seg_volume' overlap it. Used to exclude
    boundary-truncated mitochondria from batch shape statistics, since
    their measured shape descriptors would be unreliable.

    Returns
    -------
    ndarray
        Unique segmentation IDs that touch the imaged-region boundary.
    """
    out_of_bounds_volume = img_volume == 0

    out_of_bounds_volume = scipy.ndimage.binary_dilation(
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
    """
    Save one annotated count-curve figure per entry in 'results'.

    Iterates over the count curves in 'results', smooths each with the
    given 'sigma', marks the peak and the FWHM half-maximum endpoints in
    red, and saves the figure as 'destination_path / "mito id = i.png"'.
    Intended for batch inspection of many segmentations after a scripted
    analysis run.
    """
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
