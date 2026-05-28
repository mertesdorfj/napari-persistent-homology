"""
Subpixel morphological dilation and erosion in 2D and 3D.

Classical binary morphology operates one voxel at a time, which is too coarse
for accurate shape measurements at the scale of a few voxels. This module
implements subpixel-accurate dilation and erosion by evolving the level-set
PDE  dU/dt = ±|grad U|  with a first-order Osher–Sethian upwind
finite-difference scheme (paper eqs. 3–6). Running the evolution for time 't'
with step 'Lambda < 1' produces an operation equivalent to dilating / eroding
by 't' voxel-lengths but with fractional-voxel accuracy. With 'Lambda = 0.1',
ten steps equal one full round of standard morphology, i.e. one voxel layer
added (dilation) or removed (erosion).

Used by 'ph_functions.persistent_homology_erosion' /
'persistent_homology_dilation' to drive the morphological evolution between
counting steps.

Source: Wang, Østergaard, Hasselholt, Sporring,
"A semi-automatic method for extracting mitochondrial cristae characteristics
from 3D focused ion beam scanning electron microscopy data",
Communications Biology 7:377 (2024),
https://doi.org/10.1038/s42003-024-06045-4
"""

import numpy as np

##############################################################################
# Neighbour-shift helpers
#
# Each function returns a copy of 'U' shifted by one voxel along one axis,
# used inside the finite-difference stencils below to fetch the value of a
# voxel's neighbour. Boundary voxels keep their original value (Neumann /
# zero-flux boundary). Axis naming: i = axis 0 (rows), j = axis 1 (cols),
# k = axis 2 (depth). Suffix p1 = "+1" (forward shift), m1 = "-1" (backward).
##############################################################################


def U_ip1(U):
    """Return 'U' shifted by +1 along axis 0 ('out[i] = U[i+1]')."""
    U_new = np.copy(U)
    U_new[:-1] = U[1:]
    return U_new


def U_im1(U):
    """Return 'U' shifted by -1 along axis 0 ('out[i] = U[i-1]')."""
    U_new = np.copy(U)
    U_new[1:] = U[:-1]
    return U_new


def U_jp1(U):
    """Return 'U' shifted by +1 along axis 1 ('out[:, j] = U[:, j+1]')."""
    U_new = np.copy(U)
    U_new[:, :-1] = U[:, 1:]
    return U_new


def U_jm1(U):
    """Return 'U' shifted by -1 along axis 1 ('out[:, j] = U[:, j-1]')."""
    U_new = np.copy(U)
    U_new[:, 1:] = U[:, :-1]
    return U_new


def U_kp1(U):
    """Return 'U' shifted by +1 along axis 2 ('out[..., k] = U[..., k+1]')."""
    U_new = np.copy(U)
    U_new[:, :, :-1] = U[:, :, 1:]
    return U_new


def U_km1(U):
    """Return 'U' shifted by -1 along axis 2 ('out[..., k] = U[..., k-1]')."""
    U_new = np.copy(U)
    U_new[:, :, 1:] = U[:, :, :-1]
    return U_new


##############################################################################
# Subpixel dilation / erosion (Osher–Sethian upwind scheme)
#
# Each routine integrates the Hamilton–Jacobi equation  dU/dt = ±|grad U|  for
# time 't', using sub-voxel time steps of size 'Lambda'. The gradient
# magnitude is approximated by the upwind formula that uses only neighbour
# differences in the direction the front is moving (positive parts for
# dilation, negative parts for erosion). This is the 3D form of the paper's
# eqs. 3 (dilation) and 5 (erosion). U is clipped to [0, 1] after each step
# (paper eqs. 4 and 6) so it stays a valid soft indicator of the object.
#
# Parameters (common to all four functions):
#   volume: ndarray, shape (H, W) or (H, W, D)
#       Input field, typically the binary segmentation. Cast to float32.
#   t: float
#       Total evolution time in voxel-length units. After 'subpixel_dilation'
#       the object is expanded by 't' voxels; after 'subpixel_erosion' it
#       is shrunk by 't' voxels.
#   Lambda: float in (0, 1]
#       Requested sub-voxel time-step. The actual step used is
#       't / ceil(t / Lambda)' so an integer number of iterations covers
#       exactly 't'. Smaller 'Lambda' → finer accuracy, more iterations.
#
# Returns:
#   ndarray of float32 with the same shape as 'volume', values in [0, 1].
##############################################################################


def subpixel_dilation_2D(volume, t, Lambda):
    """Subpixel dilation of a 2D field by 't' voxels (see module header)."""
    U = volume.astype(np.float32)
    max_iter = int(np.ceil(t / Lambda))
    Lambda = t / max_iter

    for _i in range(max_iter):
        # |grad U| upwind for dilation: sum of squared positive differences
        # toward each in-plane neighbour. Front advances into lower-valued regions.
        U = U + Lambda * np.sqrt(
            np.square(np.clip(U_ip1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_im1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_jp1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_jm1(U) - U, a_min=0, a_max=None))
        )

        # Clip to valid soft-indicator range [0, 1]
        U[U > 1] = 1
        U[U < 0] = 0

        U = U.astype(np.float32)
    return U


def subpixel_erosion_2D(volume, t, Lambda):
    """Subpixel erosion of a 2D field by 't' voxels (see module header)."""
    U = volume.astype(np.float32)
    max_iter = int(np.ceil(t / Lambda))
    Lambda = t / max_iter

    for _i in range(max_iter):
        # |grad U| upwind for erosion: positive differences are taken from the
        # voxel toward its neighbours (front retreats away from lower-valued
        # neighbours).
        U = U - Lambda * np.sqrt(
            np.square(np.clip(U - U_ip1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_im1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_jp1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_jm1(U), a_min=0, a_max=None))
        )

        U[U > 1] = 1
        U[U < 0] = 0

        U = U.astype(np.float32)

    return U


def subpixel_dilation_3D(volume, t, Lambda):
    """Subpixel dilation of a 3D volume by 't' voxels (see module header)."""
    U = volume.astype(np.float32)
    max_iter = int(np.ceil(t / Lambda))
    Lambda = t / max_iter

    for _i in range(max_iter):
        # Same upwind stencil as the 2D version but extended to the third
        # axis (k-neighbours added).
        U = U + Lambda * np.sqrt(
            np.square(np.clip(U_ip1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_im1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_jp1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_jm1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_kp1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_km1(U) - U, a_min=0, a_max=None))
        )

        U[U > 1] = 1
        U[U < 0] = 0

        U = U.astype(np.float32)
    return U


def subpixel_erosion_3D(volume, t, Lambda):
    """Subpixel erosion of a 3D volume by 't' voxels (see module header).

    This is the workhorse used by
    'ph_functions.persistent_homology_erosion' — every persistent-homology
    step shrinks the binary mask by 'Lambda' voxel-lengths using this
    routine, then counts connected components on the binarised result.
    """
    U = volume.astype(np.float32)
    max_iter = int(np.ceil(t / Lambda))
    Lambda = t / max_iter

    for _i in range(max_iter):
        U = U - Lambda * np.sqrt(
            np.square(np.clip(U - U_ip1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_im1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_jp1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_jm1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_kp1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_km1(U), a_min=0, a_max=None))
        )

        U[U > 1] = 1
        U[U < 0] = 0

        U = U.astype(np.float32)

    return U
