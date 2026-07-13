#!/bin/bash
set -e

export PATH=$PATH:$CONDAPATH

source activate petals_env
rm -rf tests_xml/
rm -rf coverage/

BRANCH=$1
shift
CORENV=~/jobs/petals_branches/core_env
if [ -f $CORENV/$BRANCH ]; then
    source tvenv/bin/activate
fi

make $@

if [ -f $CORENV/$BRANCH ]; then
    deactivate
fi

conda deactivate
