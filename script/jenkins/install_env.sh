#!/bin/bash -e

mamba env remove -n petals_env
mamba create -n petals_env python=3.9
mamba env update -n petals_env -f ~/jobs/climada_install_env/workspace/requirements/env_climada.yml
mamba env update -n petals_env -f requirements/env_climada.yml

source activate petals_env
#python -m pip install -e ~/jobs/climada_install_env/workspace/[test]
#TODO: after mergin PR-122 remove line below and uncomment line above
python -m pip install -e ~/jobs/petals_branches/core_env/develop-freeze/[test]
python -m pip install -e "."

make install_test

conda deactivate
