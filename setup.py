"""A setuptools based setup module.
"""

from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here.joinpath('README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Add configuration files
extra_files = [str(here / 'climada_petals/conf/climada.conf')]

setup(
    name='climada_petals',

    version='3.2.0-dev',

    description='CLIMADA in Python',

    long_description=long_description,

    url='https://github.com/davidnbresch/climada_python',

    author='ETH',

    license='OSI Approved :: GNU General Public License v3 (GPLv3)',

    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Topic :: Climate Adaptation',
        'Programming Language :: Python :: 3.8',
    ],

    keywords='climate adaptation',

    packages=find_packages(where='.'),

    install_requires=[
        'bottleneck',
        'climada',
        'cartopy',
        'cdsapi',
        'cfgrib',
        'contextily',
        'dantro',
        'dask',
        'deprecation',
        'geopandas',
        'h5py',
        'haversine',
        'matplotlib',
        'netcdf4',
        'numba',
        'overpy',
        'pandas',
        'pandas-datareader',
        'pathos',
        'peewee',
        'pillow',
        'pint',
        'pybufrkit',
        'pycountry',
        'rasterio',
        'ruamel.yaml',
        'scikit-image',
        'scikit-learn',
        'statsmodels',
        'tables',
        'tabulate',
        'tqdm',
        'xarray',
        'xlrd',
        'xlsxwriter',
        'xmlrunner'
    ],

    package_data={'': extra_files},

    include_package_data=True
)
