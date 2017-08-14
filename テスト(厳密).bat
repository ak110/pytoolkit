@echo off
rem
rem -n: pytest-xdist‚É‚æ‚é•À—ñÀsBpip install pytest-xdist
rem
title %~nx0
:Retry

cls
pytest tests_strict tests -n 4

pause
goto :Retry

