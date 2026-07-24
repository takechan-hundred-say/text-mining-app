@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHON=python
if exist "python\python.exe" set PYTHON=python\python.exe
echo アプリを起動しています...
%PYTHON% -m streamlit run main.py
pause