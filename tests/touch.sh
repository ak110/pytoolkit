#!/bin/bash
set -eu
#
# テストコードを書くためのファイルを作るスクリプト。
#

for n in $(cd ../pytoolkit ; find . -maxdepth 1 -mindepth 1 -type f | sed 's@^./@@') ; do
    if [ ! -f test_$n ] ; then
        echo test_$n
        touch test_$n
    fi
done
for n in $(cd ../pytoolkit ; find . -maxdepth 1 -mindepth 1 -type d | sed 's@^./@@') ; do
    if [ $n == .ipynb_checkpoints ] ; then
        sleep 0  # skip
    elif [ ! -f test_$n.py ] ; then
        echo test_$n.py
        touch test_$n.py
    fi
done
