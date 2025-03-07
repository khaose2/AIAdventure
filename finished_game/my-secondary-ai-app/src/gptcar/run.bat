@echo off
:: Check for administrative privileges.
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~f0' -Verb runAs"
    exit /b
)

echo Running with administrative privileges.

:: Set your OpenAI API key here.
set OPENAI_API_KEY=sk-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

:: Change directory to your project folder.
cd /d "C:\Users\Khaose\Documents\GitHub\AIAdventure\finished_game\my-secondary-ai-app\src\gptcar"

:: Launch the gptcar.py application.
python gptcar.py

pause
