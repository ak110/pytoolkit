@echo off
title %~nx0
:Retry

python bench_generator.py

pause
goto :Retry

