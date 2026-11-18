@echo off
rem Double-click to open the Stochastic Music app in your browser.
cd /d "%~dp0"
py -m streamlit run src\stochastic_music\app.py
pause
