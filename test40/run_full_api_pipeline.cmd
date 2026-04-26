@echo off
setlocal

set PYTHON_EXE=C:\Users\vanya\AppData\Local\Programs\Python\Python312\python.exe
set SCRIPT_DIR=%~dp0
set PYTHONUNBUFFERED=1

"%PYTHON_EXE%" -u "%SCRIPT_DIR%run_full_api_pipeline.py" %*
exit /b %ERRORLEVEL%
