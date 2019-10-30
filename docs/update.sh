#!/bin/bash -eux
rm modules.rst pytoolkit.rst pytoolkit.*.rst || true
sphinx-apidoc --no-toc --force -o . ../pytoolkit ../**/*_test.py
