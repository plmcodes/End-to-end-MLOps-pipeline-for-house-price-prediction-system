@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (echo Run setup_windows.bat first. & pause & exit /b 1)
set PATH=%~dp0venv\Scripts;%PATH%
set PYTHONIOENCODING=utf-8
set ZENML_ANALYTICS_OPT_IN=false
echo === Asking the deployed model to price a sample house ===
python sample_predict.py
pause
