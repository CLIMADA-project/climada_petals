#!/bin/bash -e

mamba env remove -n petals_env
mamba create -n petals_env python=3.9
mamba env update -n petals_env -f ~/jobs/climada_install_env/workspace/requirements/env_climada.yml
mamba env update -n petals_env -f requirements/env_climada.yml

source activate petals_env
python -m pip install -e ~/jobs/climada_install_env/workspace/[test]
python -m pip install -e "."

make install_test

conda deactivate
