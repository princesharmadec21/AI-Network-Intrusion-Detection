@echo off
call .venv\Scripts\activate
python -m src.train
if errorlevel 1 pause & exit /b 1
python -m src.evaluate
pause
