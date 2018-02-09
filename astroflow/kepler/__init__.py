# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["kepler"]

import tensorflow as tf
from ..tf_utils import load_op_library

mod = load_op_library(__file__, "kepler_op")
kepler = mod.kepler


@tf.RegisterGradient("Kepler")
def _kepler_grad(op, *grads):
    M, e = op.inputs
    E = op.outputs[0]
    bE = grads[0]
    bM = bE / (1.0 - e * tf.cos(E))
    be = tf.reduce_sum(tf.sin(E) * bM)
    return [bM, be]
