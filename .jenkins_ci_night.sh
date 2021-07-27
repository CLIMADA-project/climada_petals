#!/bin/bash -e

source activate
mamba env update -n petals_env -f requirements/env_developer.yml

conda activate petals_env
make lint
make test
conda deactivate
