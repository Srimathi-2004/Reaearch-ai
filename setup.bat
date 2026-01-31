@echo off
echo ========================================
echo  Intelligent Document Search Setup
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/5] Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo [3/5] Creating data directories...
if not exist "data" mkdir data
if not exist "chroma_db" mkdir chroma_db

echo [4/5] Copying environment file...
if not exist ".env" (
    copy .env.example .env
    echo.
    echo [IMPORTANT] Please edit .env file with your API keys:
    echo   - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys
    echo   - KAGGLE_USERNAME and KAGGLE_KEY: Get from https://kaggle.com/settings
    echo.
)

echo [5/5] Setup complete!
echo.
echo ========================================
echo  Next Steps:
echo ========================================
echo 1. Edit .env file with your API keys
echo 2. Download dataset: python backend/data_processor.py
echo 3. Start backend: python backend/main.py
echo 4. In new terminal, setup frontend:
echo    cd frontend
echo    npm install
echo    npm start
echo.
pause
