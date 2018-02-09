#!/usr/bin/env python

import os
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):

    def build_extension(self, ext):
        include_dirs = ["astroflow", os.path.join("astroflow", "ops")]
        library_dirs = []
        libraries = []
        compile_flags = ["-O2", "-std=c++11", "-stdlib=libc++"]
        link_flags = []

        if sys.platform == "darwin":
            compile_flags += ["-march=native", "-mmacosx-version-min=10.9"]
            link_flags += ["-march=native", "-mmacosx-version-min=10.9"]
        else:
            libraries += ["m", "c++", "stdc++"]

        # Link to TensorFlow
        import tensorflow as tf
        try:
            # This is new in v1.5
            compile_flags += tf.sysconfig.get_compile_flags()
            link_flags += tf.sysconfig.get_link_flags()

        except AttributeError:
            include_dirs += [tf.sysconfig.get_include()]
            include_dirs.append(os.path.join(
                include_dirs[-1], "external", "nsync", "public"))
            library_dirs += [tf.sysconfig.get_lib()]

        # Update the extension
        ext.include_dirs += include_dirs
        ext.library_dirs += library_dirs
        ext.library_dirs += libraries
        ext.extra_compile_args += compile_flags
        ext.extra_link_args += link_flags

        _build_ext.build_extension(self, ext)


extensions = [
    Extension(
        "astroflow.astroflow_ops",
        sources=[
            os.path.join("astroflow", "ops", "kepler_op.cc"),
            os.path.join("astroflow", "ops", "searchsorted_op.cc"),
        ],
        language="c++",
    ),
]

setup(
    name="astroflow",
    license="MIT",
    packages=["astroflow"],
    ext_modules=extensions,
    cmdclass=dict(build_ext=build_ext),
)
