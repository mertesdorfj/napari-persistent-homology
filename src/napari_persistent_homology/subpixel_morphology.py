"""
Extracting Mitochondrial Cristae Characteristics from 3D Focused Ion Beam Scanning Electron Microscopy Data

Chenhao Wang, Leif Østergaard, Stine Hasselholt, Jon Sporring

https://doi.org/10.1101/2022.11.08.515664
"""

import numpy as np

##################################################
# basic functions


def U_ip1(U):
    U_new = np.copy(U)
    U_new[:-1] = U[1:]
    return U_new


def U_im1(U):
    U_new = np.copy(U)
    U_new[1:] = U[:-1]
    return U_new


def U_jp1(U):
    U_new = np.copy(U)
    U_new[:, :-1] = U[:, 1:]
    return U_new


def U_jm1(U):
    U_new = np.copy(U)
    U_new[:, 1:] = U[:, :-1]
    return U_new


def U_kp1(U):
    U_new = np.copy(U)
    U_new[:, :, :-1] = U[:, :, 1:]
    return U_new


def U_km1(U):
    U_new = np.copy(U)
    U_new[:, :, 1:] = U[:, :, :-1]
    return U_new


##################################################
# Subpixel Morphology


def subpixel_dilation_2D(volume, t, Lambda):

    U = volume.astype(np.float32)
    max_iter = int(np.ceil(t / Lambda))
    Lambda = t / max_iter

    for _i in range(max_iter):
        U = U + Lambda * np.sqrt(
            np.square(np.clip(U_ip1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_im1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_jp1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_jm1(U) - U, a_min=0, a_max=None))
        )

        # regularization steps
        U[U > 1] = 1
        U[U < 0] = 0

        # enforces float32
        U = U.astype(np.float32)
    return U


def subpixel_erosion_2D(volume, t, Lambda):

    U = volume.astype(np.float32)
    max_iter = int(np.ceil(t / Lambda))
    Lambda = t / max_iter

    for _i in range(max_iter):
        U = U - Lambda * np.sqrt(
            np.square(np.clip(U - U_ip1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_im1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_jp1(U), a_min=0, a_max=None))
            + np.square(np.clip(U - U_jm1(U), a_min=0, a_max=None))
        )

        # regularization steps
        U[U > 1] = 1
        U[U < 0] = 0

        # enforces float32
        U = U.astype(np.float32)

    return U


def subpixel_dilation_3D(volume, t, Lambda):

    U = volume.astype(np.float32)
    max_iter = int(np.ceil(t / Lambda))
    Lambda = t / max_iter

    for _i in range(max_iter):
        U = U + Lambda * np.sqrt(
            np.square(np.clip(U_ip1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_im1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_jp1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_jm1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_kp1(U) - U, a_min=0, a_max=None))
            + np.square(np.clip(U_km1(U) - U, a_min=0, a_max=None))
        )

        # regularization steps
        U[U > 1] = 1
        U[U < 0] = 0

        # enforces float32
        U = U.astype(np.float32)
    return U


def subpixel_erosion_3D(volume, t, Lambda):

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

        # regularization steps
        U[U > 1] = 1
        U[U < 0] = 0

        # enforces float32
        U = U.astype(np.float32)

    return U
