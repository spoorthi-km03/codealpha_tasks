@echo off
cd /d "%~dp0"
where py >nul 2>nul || (echo Python not found. Install Python 3.12 from python.org & pause & exit /b 1)
if not exist .venv ( py -3.12 -m venv .venv || (echo Could not create venv & pause & exit /b 1) )
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt || (echo Dependency install failed. See README troubleshooting. & pause & exit /b 1)
if not exist models\model.pt (
  echo.
  echo No trained model found. First-time setup: downloading MIDI data + training ^(takes a while on CPU^).
  python -m app.prepare_data || (echo Dataset step failed. See README. & pause & exit /b 1)
  python -m app.train || (echo Training failed. & pause & exit /b 1)
)
echo Starting AI Music Studio at http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python -m app.server
pause
