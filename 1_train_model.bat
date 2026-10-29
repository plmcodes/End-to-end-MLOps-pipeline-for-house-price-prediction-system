@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo Run setup_windows.bat first. & pause & exit /b 1)
set PATH=%~dp0venv\Scripts;%PATH%
set PYTHONIOENCODING=utf-8
set ZENML_ANALYTICS_OPT_IN=false
echo === Training the house price model (about 1-2 minutes) ===
python run_pipeline.py
pause
