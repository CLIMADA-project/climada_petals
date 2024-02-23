#!/bin/bash
export PATH=$PATH:$CONDAPATH
source activate petals_env

rm -f pylint.log

pylint -ry climada_petals | tee pylint.log
