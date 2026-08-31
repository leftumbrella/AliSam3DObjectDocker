@echo off
setlocal

set "QT_DIR=D:\Qt\5.15.2\msvc2019_x64"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%QT_DIR%\bin\qmake.exe" (
    echo Qt was not found at %QT_DIR%
    exit /b 1
)

if not exist "%VCVARS%" (
    echo Visual Studio x64 build environment was not found.
    exit /b 1
)

if not exist build mkdir build
call "%VCVARS%" >nul
if errorlevel 1 exit /b %errorlevel%

pushd build
"%QT_DIR%\bin\qmake.exe" ..\SAM3DQtClient.pro -spec win32-msvc CONFIG+=release
if errorlevel 1 goto :failed
nmake /NOLOGO
if errorlevel 1 goto :failed
popd

echo Built build\release\SAM3DQtClient.exe
exit /b 0

:failed
set "BUILD_ERROR=%errorlevel%"
popd
exit /b %BUILD_ERROR%
