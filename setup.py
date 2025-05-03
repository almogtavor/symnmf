from setuptools import Extension, setup

module = Extension(
    "symnmf",
    sources=[
        'symnmfmodule.c',
        'symnmf.c',
        'utils.c'
    ],
    extra_compile_args=['-Wall', '-Wextra', '-Werror', '-pedantic-errors']
)

setup(
    name='symnmf',
    version='1.0',
    description='Python wrapper for Symmetric NMF algorithm',
    ext_modules=[module]
)
