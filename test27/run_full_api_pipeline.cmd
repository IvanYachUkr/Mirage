@echo off
setlocal

set SCRIPT_DIR=%~dp0
set PYTHONUNBUFFERED=1

set PYTHON_EXE=
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 -u "%SCRIPT_DIR%run_full_api_pipeline.py" %*
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python -u "%SCRIPT_DIR%run_full_api_pipeline.py" %*
  exit /b %ERRORLEVEL%
)

echo Could not find a Python interpreter. Install Python 3 and make sure ^`py^` or ^`python^` is on PATH.
exit /b 1
