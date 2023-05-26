#!/bin/bash -e

source activate
mamba env remove -n petals_env
mamba env create -n petals_env -f ~/jobs/climada_install_env/workspace/requirements/env_climada.yml
mamba env update -n petals_env -f requirements/env_climada.yml

conda activate petals_env
pip install -e ~/jobs/climada_install_env/workspace/[dev]

make install_test
conda deactivate
