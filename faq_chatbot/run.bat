@echo off
cd /d "%~dp0"
where py >nul 2>nul || (echo Python not found. Install Python 3.12 from python.org and tick "Add to PATH". & pause & exit /b 1)
if not exist venv (echo Creating virtual environment... & py -3 -m venv venv)
call venv\Scripts\activate.bat
echo Installing requirements...
python -m pip install -q -r requirements.txt || (echo Install failed. Check your internet connection. & pause & exit /b 1)
echo Preparing NLTK resources...
python -c "import preprocess" || (echo NLTK setup failed. Check your internet connection. & pause & exit /b 1)
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:5000"
echo Starting server at http://127.0.0.1:5000  (Ctrl+C to stop)
python app.py
pause
