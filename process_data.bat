@echo off
echo ========================================
echo  Processing ArXiv Dataset
echo ========================================
echo.
call venv\Scripts\activate
python backend/data_processor.py
echo.
echo Done! You can now start the backend server.
pause
