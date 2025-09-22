#!/bin/bash -e

mamba env remove -n petals_env -y
mamba create -n petals_env python=3.11 -y
mamba env update -n petals_env -f ~/jobs/climada_install_env/workspace/requirements/env_climada.yml -y
mamba env update -n petals_env -f requirements/env_climada.yml -y

source activate petals_env
python -m pip install -e ~/jobs/climada_install_env/workspace/[dev]
python -m pip install -e "."

make install_test

conda deactivate
