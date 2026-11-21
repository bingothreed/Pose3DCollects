@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

if exist ".venv\Scripts\python.exe" goto deps_check

echo [Pose3D] Creating the hub environment in .venv ...
where py >nul 2>nul
if errorlevel 1 goto try_python
py -3.12 -m venv .venv
if not errorlevel 1 goto install_deps
py -3.11 -m venv .venv
if not errorlevel 1 goto install_deps
py -3.10 -m venv .venv
if not errorlevel 1 goto install_deps

:try_python
where python >nul 2>nul
if errorlevel 1 goto no_python
python -m venv .venv
if errorlevel 1 goto setup_failed

:install_deps
echo [Pose3D] Installing hub dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed
goto run

:deps_check
".venv\Scripts\python.exe" -c "import flask" >nul 2>nul
if not errorlevel 1 goto run
echo [Pose3D] Hub dependencies are missing, installing ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed

:run
if /I "%POSE3D_DESKTOP%"=="1" goto run_desktop
if /I "%POSE3D_NO_OPEN%"=="1" goto run_headless
".venv\Scripts\python.exe" app.py --open %*
exit /b 0

:run_desktop
".venv\Scripts\python.exe" -c "import webview" >nul 2>nul
if not errorlevel 1 goto launch_desktop
echo [Pose3D] Installing optional desktop window support ...
".venv\Scripts\python.exe" -m pip install -r requirements-desktop.txt
if errorlevel 1 echo [Pose3D] pywebview unavailable; using the Edge/Chrome app window fallback.

:launch_desktop
".venv\Scripts\python.exe" desktop.py %*
exit /b 0

:run_headless
".venv\Scripts\python.exe" app.py %*
exit /b 0

:no_python
echo.
echo [Pose3D] Python 3.10+ was not found on PATH.
echo          Install Python from https://www.python.org/downloads/ and tick
echo          "Add python.exe to PATH", then run start.bat again.
pause
exit /b 1

:setup_failed
echo.
echo [Pose3D] Environment setup failed. Check the messages above.
pause
exit /b 1
