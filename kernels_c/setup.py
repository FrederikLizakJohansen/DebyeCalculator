"""
Builds the C kernels as a plain shared library inside the debyecalculator_ckernels package (loaded with ctypes).
Install with: pip install ./kernels_c
"""
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

if sys.platform == 'win32':
    compile_args = ['/O2', '/fp:precise']
else:
    # No floating-point contraction (FMA), so the per-pair arithmetic rounds as in the PyTorch code
    compile_args = ['-O3', '-ffp-contract=off', '-fno-math-errno', '-std=c99']


class BuildPlainLibrary(build_ext):
    """build_ext for a library without a Python module init function."""

    def get_export_symbols(self, ext):
        return [f'dc_{name}_{suffix}' for name in ('spread', 'backward') for suffix in ('f32', 'f64')] + [
            'dc_interface_version']


setup(
    ext_modules=[Extension(
        'debyecalculator_ckernels._ckernels',
        sources=['src/kernels.c'],
        extra_compile_args=compile_args,
    )],
    cmdclass={'build_ext': BuildPlainLibrary},
)
