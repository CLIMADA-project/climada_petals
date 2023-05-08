"""A setuptools based setup module.
"""

from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here / 'doc/misc/README.md', encoding='utf-8') as f:
    long_description = f.read()

# Add configuration files
extra_files = [str(here / 'climada_petals/conf/climada.conf'), str(here / 'doc/misc/README.md')]

setup(
    name='climada_petals',

    version='3.3.0',

    description='CLIMADA in Python',

    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/davidnbresch/climada_python',

    author='ETH',

    license='OSI Approved :: GNU General Public License v3 (GPLv3)',

    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    keywords='climate adaptation',

    packages=find_packages(where='.'),

    install_requires=[
        'climada',
        'scikit-image',
    ],

    package_data={'': extra_files},

    include_package_data=True
)
