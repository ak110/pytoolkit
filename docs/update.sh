#!/bin/bash -eux
rm modules.rst pytoolkit.rst pytoolkit.*.rst || true
sphinx-apidoc --no-toc --separate --force -o . ../pytoolkit ../*/*_test.py ../*/*/*_test.py ../pytoolkit/conftest.py ../pytoolkit/bin/*.py
