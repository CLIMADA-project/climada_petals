#!/bin/bash -e

source activate petals_env
make data_test
conda deactivate
