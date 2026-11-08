@echo off
setlocal
title Trial screening workbench
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
echo First run: creating a local Python environment (needs internet, several minutes)...
set "PYCMD="
for %%V in (3.12 3.11 3.13) do (
  if not defined PYCMD ( py -%%V -c "import sys" >nul 2>&1 && set "PYCMD=py -%%V" )
)
if not defined PYCMD ( python -c "import sys; assert (3,11)<=sys.version_info[:2]<=(3,13)" >nul 2>&1 && set "PYCMD=python" )
if not defined PYCMD (
  echo Python 3.11, 3.12 or 3.13 ^(64-bit^) was not found. Install Python 3.12 from https://www.python.org and run this file again.
  goto fail
)
%PYCMD% -m venv .venv || goto fail
:install
if not exist ".venv\screening-ready.txt" (
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto fail
  echo Ready>".venv\screening-ready.txt"
)
".venv\Scripts\python.exe" setup_model.py || goto fail
echo Starting the local server. The browser opens at http://127.0.0.1:8764 when the models are loaded.
".venv\Scripts\python.exe" app.py || goto fail
exit /b
:fail
echo.
echo Startup failed. See the message above. Common fixes: install 64-bit Python 3.12, allow internet access on first run, close any program using port 8764.
pause
