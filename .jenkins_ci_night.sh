#!/bin/bash -e

source activate petals_env
mamba env update --file requirements/env_developer.yml

make lint
make test
mamba deactivate
