#!/bin/bash -eux
rm modules.rst pytoolkit.rst pytoolkit.*.rst
sphinx-apidoc --force -o . ../pytoolkit
