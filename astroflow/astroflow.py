# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "kepler",
    "searchsorted", "interp",
]

import os
import sysconfig
import tensorflow as tf


# Load the ops library
suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
libfile = os.path.join(dirname, "astroflow_ops")
if suffix is not None:
    libfile += suffix
else:
    libfile += ".so"
astroflow_ops = tf.load_op_library(libfile)


# ------
# Kepler
# ------

def kepler(M, e, **kwargs):
    """Solve Kepler's equation

    Args:
        M: A Tensor of mean anomaly values.
        e: The eccentricity (this must be a scalar).
        maxiter (Optional): The maximum number of iterations to run.
        tol (Optional): The convergence tolerance.

    Returns:
        A Tensor with the eccentric anomaly evaluated for each entry in ``M``.

    """
    return astroflow_ops.kepler(M, e, **kwargs)


@tf.RegisterGradient("Kepler")
def _kepler_grad(op, *grads):
    M, e = op.inputs
    E = op.outputs[0]
    bE = grads[0]
    bM = bE / (1.0 - e * tf.cos(E))
    be = tf.reduce_sum(tf.sin(E) * bM)
    return [bM, be]


# -------------
# Interpolation
# -------------

def searchsorted(a, v, **kwargs):
    """Find indices where elements should be inserted to maintain order

    Based loosely on the Numpy ``searchsorted`` function.

    Find the indices into a sorted array a such that, if the corresponding
    elements in ``v`` were inserted before the indices, the order of ``a``
    would be preserved.

    Args:
        a: The input array. This must be sorted in ascending order.
        v: Values to insert into ``a``. This must also be sorted in ascending
            order.

    """
    return astroflow_ops.searchsorted(a, v, **kwargs)


def interp(t, x, y):
    """One-dimensional linear interpolation

    Returns the one-dimensional piecewise linear interpolant to a function with
    given values at discrete data-points.

    Args:
        t: The x-coordinates of the interpolated values. This must be sorted
            in ascending order.
        x: The x-coordinates of the data points, must be increasing order.
        y: The y-coordinates of the data points, same length as ``x``.

    """
    inds = searchsorted(x, t)
    x_ext = tf.concat((x[:1], x, x[-1:]), axis=0)
    y_ext = tf.concat((y[:1], y, y[-1:]), axis=0)
    dx = x_ext[1:] - x_ext[:-1]
    dy = y_ext[1:] - y_ext[:-1]
    dx = tf.where(tf.greater(tf.abs(dx), tf.zeros_like(dx)),
                  dx, tf.ones_like(dx))

    x0 = tf.gather(x_ext, inds)
    y0 = tf.gather(y_ext, inds)
    slope = tf.gather(dy / dx, inds)

    return slope * (t - x0) + y0
