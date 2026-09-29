@echo off
TITLE OLYMPIA - Olympic Performance Analytics & Intelligence System
echo ============================================================
echo   OLYMPIA: Olympic Performance Analytics & Intelligence System
echo ============================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your PATH.
    echo Please install Python 3.10+ from python.org and add it to PATH.
    pause
    exit /b 1
)

echo [OK] Python detected:
python --version

:: 2. Check if Database exists, if not run ETL
if not exist "database\olympics.db" (
    echo.
    echo [INFO] Database not found. Initializing ETL Pipeline...
    python -m etl.load
    if %errorlevel% neq 0 (
        echo [ERROR] ETL Pipeline failed. Please verify dependencies in requirements.txt.
        pause
        exit /b 1
    )
) else (
    echo [OK] Olympic Star Schema Database found: database\olympics.db
)

:: 3. Launch Application
echo.
echo ============================================================
echo Starting Streamlit Dashboard...
echo ============================================================
streamlit run dashboard\app.py

pause
