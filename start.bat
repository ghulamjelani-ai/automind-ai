@echo off
rem Starts the backend (port 8000) and the frontend (port 5173) in two windows,
rem then opens the app. The AI models run separately on Colab (see README).
cd /d "%~dp0"

if not exist "backend\venv\Scripts\python.exe" (
    echo Creating the backend virtual environment...
    python -m venv backend\venv || goto :error
    backend\venv\Scripts\python.exe -m pip install -r backend\requirements.txt || goto :error
)

if not exist "frontend\node_modules" (
    echo Installing frontend packages...
    pushd frontend
    call npm install || (popd & goto :error)
    popd
)

start "Marketing AI - Backend" /D "%~dp0backend" cmd /k venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
start "Marketing AI - Frontend" /D "%~dp0frontend" cmd /k npm run dev

timeout /t 6 /nobreak >nul
start "" http://localhost:5173
exit /b 0

:error
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
