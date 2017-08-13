@echo off
title %~nx0
:Retry

cls
pytest tests && flake8 && prospector

pause
goto :Retry

