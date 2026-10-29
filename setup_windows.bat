@echo off
cd /d "%~dp0"
echo === One-time setup (needs internet, takes 5-10 minutes) ===
set ZENML_ANALYTICS_OPT_IN=false
set PYTHONIOENCODING=utf-8

if not exist venv\Scripts\python.exe (
    py -3.11 -m venv venv
    if errorlevel 1 (
        echo.
        echo Python 3.11 is required but was not found. Install it with:
        echo     winget install -e --id Python.Python.3.11
        echo or from https://www.python.org/downloads/release/python-3119/
        echo then run this file again.
        pause
        exit /b 1
    )
)
set PATH=%~dp0venv\Scripts;%PATH%

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (echo Installing packages failed. & pause & exit /b 1)

if not exist .zen zenml init
zenml experiment-tracker describe mlflow_tracker >nul 2>&1 || zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer describe mlflow >nul 2>&1 || zenml model-deployer register mlflow --flavor=mlflow
zenml stack describe local-mlflow-stack >nul 2>&1 || zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker
zenml stack set local-mlflow-stack

echo.
echo === Setup complete. Next: double-click 2_deploy_model.bat ===
pause
