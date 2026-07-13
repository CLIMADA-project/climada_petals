#!/bin/bash
set -e

export PATH=$PATH:$CONDAPATH

BRANCH=$1
CORENV=~/jobs/petals_branches/core_env

echo $CORENV/$BRANCH

if [ -f $CORENV/$BRANCH ]; then
    echo file exists
    cat $CORENV/$BRANCH

    source activate petals_env

    python -m venv --system-site-packages tvenv
    source tvenv/bin/activate

    pip install -e `cat $CORENV/$BRANCH`

    deactivate
    conda deactivate
fi
