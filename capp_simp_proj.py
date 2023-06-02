#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 07:13:13 2023

@author: forootani
"""

from itertools import repeat
from typing import Sequence

import numpy as np
from scipy.optimize import bisect
from sklearn.base import MultiOutputMixin
from sklearn.utils.validation import check_array

# Define a special object for the default value of t in
# validate_input. Normally we would set the default
# value of t to be None, but it is possible for the user
# to pass in None, in which case validate_input performs
# no checks on t.
T_DEFAULT = object()


def capped_simplex_projection(trimming_array, trimming_fraction):
    """Projection of trimming_array onto the capped simplex"""
    a = np.min(trimming_array) - 1.0
    b = np.max(trimming_array) - 0.0

    def f(x):
        return (
            np.sum(np.maximum(np.minimum(trimming_array - x, 1.0), 0.0))
            - (1.0 - trimming_fraction) * trimming_array.size
        )

    x = bisect(f, a, b)

    return np.maximum(np.minimum(trimming_array - x, 1.0), 0.0)


def prox_l0(x, threshold):
    """Proximal operator for L0 regularization."""
    return x * (np.abs(x) > threshold)

def prox_l1(x, threshold):
    """Proximal operator for L1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def prox_l2(x, threshold):
    """Proximal operator for ridge regularization."""
    return 2 * threshold * x

def prox_weighted_l0(x, thresholds):
    """Proximal operator for weighted l0 regularization."""
    y = np.zeros(np.shape(x))
    transp_thresholds = thresholds.T
    for i in range(transp_thresholds.shape[0]):
        for j in range(transp_thresholds.shape[1]):
            y[i, j] = x[i, j] * (np.abs(x[i, j]) > transp_thresholds[i, j])
    return y

def prox_weighted_l1(x, thresholds):
    """Proximal operator for weighted l1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - thresholds, np.zeros(x.shape))

def prox_weighted_l2(x, thresholds):
    """Proximal operator for ridge regularization."""
    return 2 * thresholds * x

# TODO: replace code block with proper math block
def prox_cad(x, lower_threshold):
    """
    Proximal operator for CAD regularization

    .. code ::

        prox_cad(z, a, b) =
            0                    if |z| < a
            sign(z)(|z| - a)   if a < |z| <= b
            z                    if |z| > b

    Entries of :math:`x` smaller than a in magnitude are set to 0,
    entries with magnitudes larger than b are untouched,
    and entries in between have soft-thresholding applied.

    For simplicity we set :math:`b = 5*a` in this implementation.
    """
    upper_threshold = 5 * lower_threshold
    return prox_l0(x, upper_threshold) + prox_l1(x, lower_threshold) * (
        np.abs(x) < upper_threshold
    )


def get_prox(regularization):
    prox = {
        "l0": prox_l0,
        "weighted_l0": prox_weighted_l0,
        "l1": prox_l1,
        "weighted_l1": prox_weighted_l1,
        "l2": prox_l2,
        "weighted_l2": prox_weighted_l2,
        "cad": prox_cad,
    }
    if regularization.lower() in prox.keys():
        return prox[regularization.lower()]
    else:
        raise NotImplementedError("{} has not been implemented".format(regularization))
        
        
def get_regularization(regularization):
    if regularization.lower() == "l0":
        return lambda x, lam: lam * np.count_nonzero(x)
    elif regularization.lower() == "weighted_l0":
        return lambda x, lam: np.sum(lam[np.nonzero(x)])
    elif regularization.lower() == "l1":
        return lambda x, lam: lam * np.sum(np.abs(x))
    elif regularization.lower() == "weighted_l1":
        return lambda x, lam: np.sum(np.abs(lam @ x))
    elif regularization.lower() == "l2":
        return lambda x, lam: lam * np.sum(x**2)
    elif regularization.lower() == "weighted_l2":
        return lambda x, lam: np.sum(lam @ x**2)
    elif regularization.lower() == "cad":  # dummy function
        return lambda x, lam: 0
    else:
        raise NotImplementedError("{} has not been implemented".format(regularization))