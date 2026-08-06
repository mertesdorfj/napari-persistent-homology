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
curve is then passed through function 'compute_homology_stats_v2', which
smooths it with a Lambda-sized moving average and reports the peak
location and full-width-at-half-maximum ALREADY converted to voxel units.
The napari widget uses this v2 pipeline.

Legacy v1 helpers ('find_max_location', 'compute_FWHM',
'compute_homology_stats') remain in this module for backwards
compatibility and regression coverage — they Gaussian-smooth the curve
externally, return step-unit results, and are noise-sensitive on real
segmentation data. See the docstring of each v2 function for a
side-by-side comparison with its v1 predecessor.

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
# Per-object sub-volume extraction
#
# Helpers for the widget's per-object analysis mode: isolate one label from a
# multi-label mask and crop it (and, optionally, an aligned container mask) to
# its bounding box before running the persistent-homology pipeline. The
# analysis functions pad their input internally, so a tight foreground crop is
# safe — enclosed holes lie inside the bounding box, while the outer
# background that 'hole_count' subtracts is outside it and irrelevant.
##############################################################################


def bounding_box_crop(mask):
    """Crop a mask to the bounding box of its nonzero voxels.

    Parameters
    ----------
    mask: ndarray
        3D array; nonzero voxels are treated as foreground.

    Returns
    -------
    cropped: ndarray
        The smallest sub-array containing every nonzero voxel of 'mask',
        with the same dtype.
    bbox: tuple of int
        '(z0, y0, x0, z1, y1, x1)' — the half-open bounds such that
        'mask[z0:z1, y0:y1, x0:x1]' equals 'cropped'.

    Raises
    ------
    ValueError
        If 'mask' has no nonzero voxels (an empty bounding box).
    """
    nonzero = np.argwhere(mask)
    if nonzero.size == 0:
        raise ValueError('Cannot crop an empty mask (no nonzero voxels).')
    z0, y0, x0 = nonzero.min(axis=0)
    z1, y1, x1 = nonzero.max(axis=0) + 1
    bbox = (int(z0), int(y0), int(x0), int(z1), int(y1), int(x1))
    return mask[z0:z1, y0:y1, x0:x1], bbox


def label_subvolume(label_data, label_id, container_data=None):
    """Extract a single label as a cropped binary sub-volume.

    Builds the binary mask '(label_data == label_id)', crops it to its
    bounding box, and — when 'container_data' is given — crops the container
    to the *same* bounding box so the two stay voxel-aligned for the
    internal-spacing analysis.

    Parameters
    ----------
    label_data: ndarray
        3D integer label volume.
    label_id: int
        The label value to isolate.
    container_data: ndarray or None
        Optional container / parent mask of identical shape. Cropped to the
        same bounding box and binarized when provided.

    Returns
    -------
    obj_mask: ndarray of uint8
        The isolated label, cropped, values in {0, 1}.
    container_mask: ndarray of uint8 or None
        The container cropped to the same bbox (binarized), or None when no
        container was supplied.
    bbox: tuple of int
        '(z0, y0, x0, z1, y1, x1)' bounding box in the original volume.

    Raises
    ------
    ValueError
        If 'label_id' is absent from 'label_data'.
    """
    obj_mask, bbox = bounding_box_crop(label_data == label_id)
    obj_mask = obj_mask.astype(np.uint8)

    container_mask = None
    if container_data is not None:
        z0, y0, x0, z1, y1, x1 = bbox
        container_mask = (container_data[z0:z1, y0:y1, x0:x1] > 0).astype(
            np.uint8
        )
    return obj_mask, container_mask, bbox


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

    .. deprecated::
        Legacy v1 function — no longer used by the napari widget.
        Kept for backwards compatibility with external callers and as
        regression coverage for the noise-tolerant replacement,
        'find_max_location_v2'. Prefer v2 for new code.

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


def find_max_location_v2(
    series, offset=5, Lambda=0.1, rank_peaks_by_smoothed=False
):
    """
    Locate the peak of a count curve — robust, noise-tolerant version.

    Runs the structural analysis (local-maximum detection from the
    discrete first difference + the first-peak noise filter) entirely
    on the RAW input curve — a pre-smoothing pass would blur the very
    count-drop-to-1 pattern the noise filter looks for.

    When several local maxima survive the noise filter, the "tallest"
    one is selected by weighting each candidate by its count value.
    The parameter 'rank_peaks_by_smoothed' controls which count curve is
    used for THAT weighting step (structure detection is always on
    the raw curve):

    * 'rank_peaks_by_smoothed = False' (default, original Chenhao v2
      behaviour) — weight by the raw counts. Fastest and cleanest on
      well-behaved segmentation data.
    * 'rank_peaks_by_smoothed = True' — weight by a moving-average
      smoothed version of the curve (window 'round(1 / Lambda)').
      Useful when the raw curve carries individual noise samples
      that spike above the true peak — with raw weighting, argmax
      would pick that spike; with smoothed weighting, the tall but
      wide real peak wins instead.

    Parameters
    ----------
    series: ndarray
        1D count curve — object or hole counts as a function of
        subpixel-step. Cast to 'float' internally.
    offset: int, optional
        Number of initial samples to skip when locating the peak,
        matching the paper's convention of excluding the first noisy
        sub-round of morphology. Default = 5.
    Lambda: float, optional
        Subpixel step size in voxel-length units. Used by the noise-
        filter heuristic to translate voxel-scale thresholds (2 voxel
        rounds for the 'risky' region, 1 voxel round for the 'stable
        count = 1' segment) into step counts, and — when
        'rank_peaks_by_smoothed' is True — to size the moving-average
        window. Default = 0.1.
    rank_peaks_by_smoothed: bool, optional
        See the algorithm description above. Default = False.

    Returns
    -------
    int
        Step index of the selected peak (right-most local maximum
        after noise filtering). On any internal error the function
        returns 0.

    Differences vs. 'find_max_location' (v1)
    ---------------------------------------
    * v1 returns the right-most position of the GLOBAL maximum of
      'series[offset:]'. It cannot distinguish a real peak from a
      spurious early peak that happens to be higher; if a small
      segmentation-noise blob briefly creates a component, v1 will
      report the position of that first bump.
    * v2 identifies every LOCAL maximum from the sign of the first
      difference, then runs a first-peak noise filter: if the first
      peak sits within '~ 2 / Lambda' steps (i.e. two full voxel
      layers) AND the count drops back to 1 for at least '~ 1 /
      Lambda' consecutive samples AND another peak still exists
      later in the curve, the first peak is treated as noise and
      excluded. v2 then returns the right-most surviving peak,
      weighted by its count value.
    * v2 handles the degenerate case where the count curve never
      increases (a 'perfect' shape with no noise that collapses
      immediately) as a separate branch.
    """
    try:
        series = series.astype(float)

        # Curve used ONLY for weighting the surviving local maxima
        # when picking the "tallest" peak — the raw series unless
        # the caller asked us to weight by a smoothed version.
        if rank_peaks_by_smoothed:
            weighting_window = max(1, int(np.round(1 / Lambda)))
            weighting_series = moving_average(series, w=weighting_window)
        else:
            weighting_series = series

        # Discrete first difference of the RAW series. Padding with
        # the boundary values keeps the diff array the same length as
        # 'series', so index arithmetic below stays aligned.
        diff_along_series = np.diff(
            series, prepend=series[0], append=series[-1]
        )

        if np.max(diff_along_series) == 0:
            # ── Degenerate branch ────────────────────────────────────
            # The count never rises — most likely a perfect shape
            # without noise that collapses in a single round. In this
            # case treat every position where the diff turns negative
            # (i.e. the collapse points) as a candidate peak. The '[1:]'
            # shifts the index by one to compensate for the padding.
            all_local_maximum = np.signbit(diff_along_series[1:]).astype(int)
            peaks_only_curve_right = all_local_maximum * weighting_series

        else:
            # ── Normal branch ────────────────────────────────────────
            # Reduce the diff to its sign: +1 (rising), -1 (falling),
            # 0 (plateau). '[:-1]' drops the tail padding we no longer
            # need now that a real max exists.
            signed = np.sign(diff_along_series[:-1]).astype(int)

            # Working only with non-zero samples lets us treat brief
            # flat plateaus as if they were a single point when looking
            # for sign changes.
            non_zero_indices = np.where(signed != 0)[0]
            signed_no_zeros = signed[non_zero_indices]

            # A local maximum is a +1 → -1 transition; the diff of
            # signs equals -2 at those crossings. This gives the
            # LEFT-most index of each plateau maximum.
            strict_zeroless_maximum_left = (
                np.diff(np.sign(signed_no_zeros)) == -2
            ).astype(int)
            strict_zeroless_maximum_left_indices = np.where(
                strict_zeroless_maximum_left
            )[0]

            # Mark the left-most maxima back in the original coordinate
            # frame of 'series'.
            all_local_maximums_left = np.zeros(signed.shape, dtype=int)
            all_local_maximums_left[non_zero_indices[:-1]] = (
                strict_zeroless_maximum_left
            )

            # For each left-most maximum, walk forward through the
            # non-zero indices to find the corresponding RIGHT-most
            # plateau edge. Empirically the right edge gives a more
            # accurate peak location.
            all_local_maximums_right = np.zeros(signed.shape, dtype=int)
            all_local_maximums_right[
                non_zero_indices[strict_zeroless_maximum_left_indices + 1] - 1
            ] = 1

            # Weight each peak by its count value so downstream argmax
            # picks the TALLEST surviving peak, not just the last one.
            # 'weighting_series' is the raw curve by default, but the
            # smoothed curve when 'rank_peaks_by_smoothed=True' — see the
            # docstring for when to enable that.
            peaks_only_curve_right = (
                all_local_maximums_right * weighting_series
            )

            # ── First-peak noise filter ──────────────────────────────
            # A very common failure mode: a small segmentation-noise
            # blob creates a briefly-visible extra component early in
            # the morphology, then vanishes. This shows up as a low,
            # narrow first peak, a return to count = 1, and then the
            # real (later) peak. We drop the first peak when all three
            # conditions are met.
            peak_indices = np.where(all_local_maximums_right)[0]
            first_peak_location = peak_indices[0]

            # Condition 1 — first peak lies within ~2 full voxel rounds
            # (the 'noise-risky' region near step 0).
            risking_full_pixel_len_threshold = 2
            risk_threshold = int(
                np.round(risking_full_pixel_len_threshold / Lambda)
            )
            risky_first_peak = first_peak_location <= risk_threshold

            # Condition 2 — after the first peak the count returns to 1
            # and stays there for at least one full voxel round (i.e.
            # '~ 1 / Lambda' consecutive samples).
            find_count_1_after_first_peak = series[peak_indices[0] :] == 1

            if (
                np.sum(find_count_1_after_first_peak)
                >= int(np.round(1 / Lambda))
                and risky_first_peak
            ):
                # First step index (in the original series) at which the
                # count returns to 1 after the first peak.
                index_first_count_of_1 = (
                    np.argmax(series[peak_indices[0] :] == 1) + peak_indices[0]
                )

                # Condition 3 — at least one further peak exists AFTER
                # the return-to-1 point. If not, the first peak might
                # still be the real signal, so keep it.
                peak_check = peak_indices > index_first_count_of_1
                peak_exist_after_count_returned_to_1 = np.sum(peak_check) >= 1

                if peak_exist_after_count_returned_to_1:
                    # All three conditions met → discard every peak before the
                    # return-to-1 point (typically just the noisy first one).
                    peaks_only_curve_right[peak_indices[~peak_check]] = 0

        # Return the right-most non-zero entry of 'peaks_only_curve_right'
        # (which is the count value at each surviving peak position, 0
        # elsewhere). 'np.argmax' on the reversed array picks the tallest,
        # and ties break rightward.
        assert len(series) > offset, (
            'Length of the number array must be larger than offset'
        )
        argmax_output = (
            (
                len(peaks_only_curve_right[offset:][::-1])
                - np.argmax(peaks_only_curve_right[offset:][::-1])
            )
            - 1
            + offset
        )

    except Exception:  # noqa: BLE001 — v2 intentionally returns 0 on any failure
        argmax_output = 0

    return argmax_output


def moving_average(x, w):
    """
    Length-preserving moving-average smoother with window size 'w'.

    Uses 'np.convolve' in 'same' mode so the returned array has the
    same shape as the input — the boundary samples are averaged
    against implicit zeros, which slightly biases the very first and
    very last few samples but keeps indices aligned with the raw
    curve. Used by 'compute_FWHM_v2' to pre-smooth count curves
    before locating half-maximum crossings.
    """
    return np.convolve(x, np.ones(w), 'same') / w


def compute_FWHM(series, offset=5):
    """
    Full-width at half-maximum of a count curve, around its peak.

    .. deprecated::
        Legacy v1 function — no longer used by the napari widget.
        Kept for backwards compatibility with external callers and as
        regression coverage for the noise-tolerant replacement,
        'compute_FWHM_v2'. Prefer v2 for new code.

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


def compute_FWHM_v2(
    series, offset=5, Lambda=0.1, rank_peaks_by_smoothed=False
):
    """
    Full-width at half-maximum of a count curve — v2.

    Deliberately mixes RAW and SMOOTHED views of the curve:

    * Peak *location* comes from 'find_max_location_v2', which runs
      entirely on the raw curve and applies its own noise filter.
    * Peak *height*, from which the half-max threshold is derived,
      is read straight off the raw curve at that location — unless
      'rank_peaks_by_smoothed=True', in which case it is read off the
      moving-average-smoothed curve for consistency with how the
      peak was chosen (see 'find_max_location_v2').
    * The FWHM edges are then found on the MOVING-AVERAGE-SMOOTHED
      version of the curve — window size 'round(1 / Lambda)', i.e.
      one full voxel round — so the width is stable against sample-
      to-sample noise.

    Concretely, with the default 'rank_peaks_by_smoothed=False':
    half-max = raw peak / 2, and the left / right walks step outward
    from the peak until the SMOOTHED curve first drops below that
    raw-derived threshold. With 'rank_peaks_by_smoothed=True': half-max =
    smoothed peak / 2 and the walks use the same threshold.

    Parameters
    ----------
    series: ndarray
        1D count curve — same input as 'find_max_location_v2'.
    offset: int, optional
        Initial samples to skip when locating the peak. Forwarded to
        'find_max_location_v2'. Default = 5.
    Lambda: float, optional
        Subpixel step size in voxel-length units. Sets both the
        moving-average window ('round(1 / Lambda)' samples ≈ one
        voxel round) and the noise-filter thresholds used inside
        'find_max_location_v2'. Default = 0.1.
    rank_peaks_by_smoothed: bool, optional
        Forwarded to 'find_max_location_v2' and used locally to pick
        between the raw and smoothed peak value for the half-max
        threshold. Default = False.

    Returns
    -------
    tuple
        '(FWHM, (left_half, middle, right_half))' where
        'FWHM == left_half + middle + right_half', all in step
        units. The 3-part decomposition exposes the left / right
        asymmetry of the peak — useful downstream for distinguishing
        curves with a long left tail from those with a long right
        tail. On any internal error the tuple is '(0, (0, 0, 0))'.

    Differences vs. 'compute_FWHM' (v1)
    ----------------------------------
    * v1 uses 'find_max_location' (global argmax of 'series[offset:]')
      for the peak and assumes the caller has already Gaussian-
      smoothed the curve inside 'compute_homology_stats'. v2 uses
      'find_max_location_v2' (noise-filtered local maxima) and
      smooths internally with a MOVING AVERAGE whose window is
      derived from 'Lambda', so the caller can pass a raw curve.
    * v1 finds the FWHM edges by collecting every above-half-max
      index in the whole series, splitting them into contiguous
      runs with 'np.split', and taking the run that contains the
      peak. v2 walks outward from the peak until the boolean
      'smoothed >= half_max' mask first turns False, which is
      simpler, requires no contiguity bookkeeping, and never picks
      up an above-half-max region belonging to a different peak.
    * v1 returns a single scalar (the FWHM); v2 returns the FWHM
      plus a '(left, middle, right)' triple that sums to it.
    """
    try:
        # Moving-average window ≈ one full voxel round. Clamp to at
        # least 1 to avoid an empty convolution on extreme Lambdas.
        moving_avg_window_size = int(np.round(1 / Lambda))
        if moving_avg_window_size == 0:
            moving_avg_window_size = 1

        # Peak location comes from v2 (noise-filtered on the raw curve;
        # optionally weighted by the smoothed curve when picking the
        # tallest local max — see 'find_max_location_v2').
        max_location = find_max_location_v2(
            series,
            offset,
            Lambda,
            rank_peaks_by_smoothed=rank_peaks_by_smoothed,
        )

        # Smooth the whole curve once so both edge walks below see
        # the same denoised profile.
        smoothed_series = moving_average(series, w=moving_avg_window_size)

        # Peak height (from which the half-max threshold is derived)
        # matches whichever curve the peak was chosen on: raw by
        # default, smoothed when 'rank_peaks_by_smoothed=True'.
        if rank_peaks_by_smoothed:
            maximum_count = smoothed_series[max_location]
        else:
            maximum_count = series[max_location]
        half_maximum = maximum_count / 2

        larger_HM_series = smoothed_series >= half_maximum

        # Walk left from the peak until the mask first turns False.
        # 'argmin' on the reversed slice returns the index of the
        # first False (False < True), which is the required distance.
        left_half_max = np.argmin(larger_HM_series[:max_location][::-1])
        # The peak sample itself always sits above half-max, so we
        # add 1 for it.
        middle = 1
        # Walk right from (and including) the peak; subtract 1 because
        # 'argmin' returns the position OF the first False whereas we
        # want the count of True samples strictly before it.
        right_half_max = np.argmin(larger_HM_series[max_location:]) - 1

        full_width_half_maximum = left_half_max + middle + right_half_max

    except Exception:  # noqa: BLE001 — v2 intentionally returns 0 on any failure
        full_width_half_maximum = 0
        left_half_max = 0
        middle = 0
        right_half_max = 0

    return full_width_half_maximum, (left_half_max, middle, right_half_max)


def compute_homology_stats(series_of_series, offset=5, SIGMA=3):
    """
    Smooth one or more count curves and extract peak + FWHM statistics.

    .. deprecated::
        Legacy v1 function — no longer used by the napari widget.
        Kept for backwards compatibility with external callers and as
        regression coverage for the noise-tolerant replacement,
        'compute_homology_stats_v2'. Prefer v2 for new code.

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


def compute_homology_stats_v2(
    series, offset=5, Lambda=0.1, rank_peaks_by_smoothed=False
):
    """
    Extract peak + FWHM statistics from a single count curve — v2.

    Wraps 'find_max_location_v2' (noise-tolerant peak detection on
    the RAW curve) and 'compute_FWHM_v2' (raw peak height + FWHM
    edges walked on a moving-average-smoothed curve), then converts
    the step-unit results into voxel units by multiplying by
    'Lambda' — so the caller receives numbers that are directly
    interpretable as voxel-length quantities.

    See each helper's docstring for the exact raw / smoothed split.

    This is the version the napari widget calls; the widget passes
    the raw count curve straight in and displays the returned
    voxel-unit values.

    Parameters
    ----------
    series: ndarray
        A single 1D count curve (object or hole count as a function
        of subpixel-step).
    offset: int, optional
        Initial samples to skip when locating the peak. Forwarded to
        the v2 helpers. Default = 5.
    Lambda: float, optional
        Subpixel step size in voxel-length units. Governs the
        moving-average window ('round(1 / Lambda)') and the noise
        thresholds inside 'find_max_location_v2', and provides the
        step-to-voxel scaling for the returned values. Default = 0.1.
    rank_peaks_by_smoothed: bool, optional
        Forwarded to both v2 helpers. When True, the argmax that
        picks the "tallest" surviving local maximum ranks candidates
        by their moving-average smoothed value instead of their raw
        count (and 'maximum_count' returned below is likewise read
        off the smoothed curve). Candidate identification and the
        noise filter still run on the raw curve — see
        'find_max_location_v2' for the exact split. Default = False.

    Returns
    -------
    tuple
        '(FWHM_vox, max_location_vox, maximum_count)':
        * 'FWHM_vox' — full-width at half-maximum, in voxel units.
        * 'max_location_vox' — peak location, in voxel units.
        * 'maximum_count' — the raw count value at the peak (no
          conversion, since it is already a dimensionless count).

        On an internal failure inside the v2 helpers, the two
        distance values become 0.0 (since v2 falls back to a
        step-index of 0), and the caller should treat this as a
        degenerate curve.

    Differences vs. 'compute_homology_stats' (v1)
    --------------------------------------------
    * v1 accepts a SEQUENCE of curves and returns an 'ndarray(3, N)';
      v2 processes a single curve and returns a plain tuple. Both
      the widget and the paper's own downstream code only ever
      analyse one curve at a time, so v2 avoids the collection
      bookkeeping.
    * v1 requires a 'SIGMA' argument and pre-smooths each curve
      with a Gaussian filter. v2 has no 'SIGMA' — the smoothing is
      a moving average sized by 'Lambda' and lives inside
      'compute_FWHM_v2'.
    * v1 returns FWHM and max-location in RAW STEP UNITS; the
      caller has to divide by 'ceil(1 / Lambda)' to get voxels. v2
      does the conversion internally so the caller works only in
      voxel units.
    * v1 uses the older peak / FWHM helpers ('find_max_location',
      'compute_FWHM') and inherits their noise sensitivity. v2 uses
      the noise-tolerant v2 helpers.
    """
    # 'FWHM_left', 'FWHM_middle', 'FWHM_right' are the three pieces
    # that sum to 'FWHM'. We only need the aggregate here, but the
    # tuple is destructured to make the intent explicit and to leave
    # a hook for callers that want the left / right asymmetry.
    FWHM, (FWHM_left, FWHM_middle, FWHM_right) = compute_FWHM_v2(
        series, offset, Lambda, rank_peaks_by_smoothed=rank_peaks_by_smoothed
    )
    max_location = find_max_location_v2(
        series, offset, Lambda, rank_peaks_by_smoothed=rank_peaks_by_smoothed
    )

    # Peak height reported to the caller: raw by default (matches the
    # original Chenhao v2 return), smoothed when the caller opted in
    # to smoothed-based peak selection.
    if rank_peaks_by_smoothed:
        smooth_window = max(1, int(np.round(1 / Lambda)))
        smoothed_series = moving_average(series, w=smooth_window)
        maximum_count = smoothed_series[max_location]
    else:
        maximum_count = series[max_location]

    # Convert step-unit values back to voxel units
    # ('step_index * Lambda' = voxel-length).
    FWHM_vox = FWHM * Lambda
    max_location_vox = max_location * Lambda

    return FWHM_vox, max_location_vox, maximum_count


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
