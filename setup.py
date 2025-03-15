from setuptools import Extension, setup

module = Extension(
    "mykmeanssp",
    sources=[
        'kmeansmodule.c',
        'kmeans.c'
    ]
)

setup(
    name='mykmeanssp',
    version='1.0',
    description='Python wrapper for K-means algorithm',
    ext_modules=[module]
)
