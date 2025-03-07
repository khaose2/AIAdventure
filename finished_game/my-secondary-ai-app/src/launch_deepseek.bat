@echo off
:: Check for administrative privileges.
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~f0' -Verb runAs"
    exit /b
)

echo Running with administrative privileges.

:: Change directory to your project folder.
cd /d "C:\Users\Khaose\Documents\GitHub\AIAdventure\finished_game\my-secondary-ai-app\src"

:: Launch the simple_text_ai.py application.
python simple_text_ai.py

pause
