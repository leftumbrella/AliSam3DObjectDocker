@echo off
setlocal

set "QT_DIR=D:\Qt\5.15.2\msvc2019_x64"
set "APP=%~dp0build\release\SAM3DQtClient.exe"

call "%~dp0build-release.bat"
if errorlevel 1 exit /b %errorlevel%

set "PATH=%QT_DIR%\bin;%PATH%"
"%APP%" %*
