@echo off
echo Starting Backend Server...
call venv\Scripts\activate
cd backend
python main.py
pause
