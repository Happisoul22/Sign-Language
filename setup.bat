@echo off
echo ============================================
echo Sign Language Translator - Setup
echo ============================================
echo.

echo Installing required packages...
echo.

pip install streamlit==1.31.0
pip install opencv-python==4.9.0.80
pip install mediapipe==0.10.9
pip install tensorflow==2.15.0
pip install numpy==1.26.3
pip install pyttsx3==2.90
pip install pillow==10.2.0
pip install scikit-learn==1.4.0

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo To run the app, use:
echo     streamlit run streamlit_app.py
echo.
pause
