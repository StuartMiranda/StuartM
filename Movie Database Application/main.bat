@echo off

set "install=n"

set /p install= Install necessary modules (required for program to run)? (y/[n])
if "%install%"=="y" (
	call "%~dp0\scripts\install.bat"
)

echo 1: CLI
echo 2: GUI

set /p ui= which UI would you like to use? ([1]/2)
if "%ui%"=="2" (
	streamlit run "%~dp0\GUI\final_version_GUI.py"
) else (
	py "%~dp0\main.py"
)

