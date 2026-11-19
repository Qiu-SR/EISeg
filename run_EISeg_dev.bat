@echo off
rem Run EISeg directly from this source tree (not the installed eiseg package).
rem Keep this file ASCII-only: cmd.exe reads .bat in the system ANSI codepage,
rem so UTF-8 Chinese comments here would be garbled and executed as commands.
setlocal
set "PY=D:\miniconda3\envs\eiseg\python.exe"
set "SRC=%~dp0"

if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    pause
    exit /b 1
)

pushd "%SRC%"
"%PY%" -c "import eiseg; from eiseg.run import main; main()"
set "RC=%errorlevel%"
popd

if not "%RC%"=="0" (
    echo.
    echo [ERROR] EISeg exited with code %RC%
    pause
)
endlocal
