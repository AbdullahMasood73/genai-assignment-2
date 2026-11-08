@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv || goto fail
)
if not exist ".venv\screening-ready.txt" (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
  echo Ready>".venv\screening-ready.txt"
)
".venv\Scripts\python.exe" setup_model.py || goto fail
".venv\Scripts\python.exe" app.py || goto fail
exit /b
:fail
echo Startup failed. Install Python 3.12 and allow first-run downloads, then retry.
pause
