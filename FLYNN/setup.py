import sys
from cx_Freeze import setup, Executable

import os

additional_mods = ['numpy.core._methods', 'numpy.lib.format', 'scipy.sparse.csgraph._validation', \
                    'scipy.spatial.ckdtree', 'scipy.stats', 'scipy.signal', \
                    'scipy.optimize', 'scipy.ndimage._ni_support']
build_exe_options = {"packages": ["os"], "excludes": ["tkinter"], "includes": additional_mods, "include_msvcr": []}

# GUI applications require a different base on Windows (the default is for a
# console application).

base = "Win32GUI"

setup(  name = "FLYNN",
        version = "0.1",
        description = "NMR control for the FLYNN filling station",
        options = {"build_exe": build_exe_options},
        executables = [Executable("nmr_control.py", base=base)])