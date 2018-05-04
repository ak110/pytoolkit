@echo off
title %~nx0
pushd "%~dp0.."
:Retry

python exp\bench_generator.py

pause
goto :Retry
popd
