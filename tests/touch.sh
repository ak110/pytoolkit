#!/bin/bash
set -eu
#
# テストコードを書くためのファイルを作るスクリプト。
#

for n in $(cd ../pytoolkit ; \ls -F | grep -v /) ; do
    if [ ! -f test_$n ] ; then
        echo test_$n
        touch test_$n
    fi
done
