"""A setuptools based setup module.
"""

from pathlib import Path
from setuptools import setup, find_namespace_packages

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here / 'doc/misc/README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='climada_petals',

    version='5.0.0',

    description='CLIMADA Extensions in Python',

    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/CLIMADA-project/climada_python',

    author='ETH',

    license='OSI Approved :: GNU General Public License v3 (GPLv3)',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    keywords='climate adaptation',

    install_requires=[
        'climada>=5.0',
        'cdsapi',
        'osm-flex',
        "pymrio",
        'rioxarray',
        'ruamel.yaml',
        'scikit-image',
        'xesmf',
    ],

    packages=find_namespace_packages(include=['climada_petals*']),

    setup_requires=['setuptools_scm'],
    include_package_data=True,
)
