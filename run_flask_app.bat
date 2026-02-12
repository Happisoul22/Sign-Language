@echo off
title Sign Language Translator - Flask Server

echo ============================================
echo  Sign Language Translator
echo  Starting Flask Server...
echo ============================================
echo.

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask is not installed!
    echo Installing Flask now...
    pip install flask
    echo.
)

echo Starting server at http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

python app.py

pause
