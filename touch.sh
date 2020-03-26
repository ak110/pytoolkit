#!/bin/bash
set -eu
#
# *.py に対応する *_test.py を作成するスクリプト
#

for f in $(find pytoolkit -name '*.py') ; do
    if [[ "$f" == *"_test.py"* || "$f" == *"conftest.py"* || "$f" == *"/bin/"* || "$f" == *"__init__.py"* ]]; then
        :
    else
        t="$(dirname $f)/$(basename $f .py)_test.py"
        if [ ! -e "$t" ] ; then
            echo $t
            touch $t
        fi
    fi
done
