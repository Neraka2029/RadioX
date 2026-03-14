@echo off
title RadioX.AI
color 0B
cls

echo.
echo  ============================================
echo       RadioX.AI - Analyse de Radiographies
echo  ============================================
echo.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo  [1/5] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERREUR: Python non installe.
    echo  Allez sur https://www.python.org/downloads
    echo  Cochez "Add Python to PATH"
    pause
    exit /b 1
)
echo  [OK] Python detecte

echo  [2/5] Verification de Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo  ERREUR: Node.js non installe.
    echo  Allez sur https://nodejs.org
    pause
    exit /b 1
)
echo  [OK] Node.js detecte

echo  [3/5] Installation des dependances si necessaire...

if not exist "%ROOT%\backend\venv\" (
    echo  Installation backend...
    python -m venv "%ROOT%\backend\venv"
    "%ROOT%\backend\venv\Scripts\pip.exe" install bcrypt==4.0.1 --quiet
    "%ROOT%\backend\venv\Scripts\pip.exe" install -r "%ROOT%\backend\requirements.txt" --quiet
    echo  [OK] Backend installe
) else (
    echo  [OK] Backend OK
)

if not exist "%ROOT%\ml_service\venv\" (
    echo  Installation service IA...
    python -m venv "%ROOT%\ml_service\venv"
    "%ROOT%\ml_service\venv\Scripts\pip.exe" install -r "%ROOT%\ml_service\requirements.txt" --quiet
    echo  [OK] Service IA installe
) else (
    echo  [OK] Service IA OK
)

if not exist "%ROOT%\frontend\node_modules\" (
    echo  Installation frontend...
    cd /d "%ROOT%\frontend"
    npm install --silent
    cd /d "%ROOT%"
    echo  [OK] Frontend installe
) else (
    echo  [OK] Frontend OK
)

echo  [4/5] Demarrage des services...
start "RadioX-Backend" cmd /k "cd /d "%ROOT%\backend" && "%ROOT%\backend\venv\Scripts\uvicorn.exe" main:app --port 8000 --log-level warning"
timeout /t 4 /nobreak >nul

start "RadioX-MLService" cmd /k "cd /d "%ROOT%\ml_service" && "%ROOT%\ml_service\venv\Scripts\uvicorn.exe" main:app --port 8001 --log-level warning"
timeout /t 8 /nobreak >nul

start "RadioX-Frontend" cmd /k "cd /d "%ROOT%\frontend" && npm run dev"
timeout /t 5 /nobreak >nul

echo  [5/5] Ouverture du navigateur...
start http://localhost:3000

cls
echo.
echo  ============================================
echo   RadioX.AI est demarre !
echo.
echo   Navigateur: http://localhost:3000
echo.
echo   Email   : demo@radiox.ai
echo   Pass    : demo123
echo  ============================================
echo.
echo  Appuyez sur une touche pour ARRETER RadioX.AI
echo.
pause >nul

echo  Arret en cours...
taskkill /FI "WINDOWTITLE eq RadioX-Backend" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq RadioX-MLService" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq RadioX-Frontend" /F >nul 2>&1
echo  Arrete. Au revoir !
timeout /t 2 /nobreak >nul
