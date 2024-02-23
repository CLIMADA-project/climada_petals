#!/bin/bash
export PATH=$PATH:$CONDAPATH

source activate petals_env
rm -rf tests_xml/
rm -rf coverage/

CORENV=~/jobs/petals_branches/core_env
BRANCH=`git name-rev --name-only HEAD | cut -f 3- -d /`
if [ -f $CORENV/$BRANCH ]; then
    python -m venv --system-site-packages tvenv
    source tvenv/bin/activate

    pip install -e `cat $CORENV/$BRANCH`
fi

make unit_test

if [ -f $CORENV/$BRANCH ]; then
    deactivate
    #rm -r tvenv
fi
