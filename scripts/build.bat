@echo off
REM ForgeRT Build Script
REM Configures MSVC 14.39 environment and builds with Ninja

echo ========================================
echo ForgeRT Build Script
echo ========================================
echo.

REM Set MSVC 14.39 environment
call "D:\Development\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39 >nul

if "%1"=="clean" (
    echo Cleaning build directory...
    if exist build rmdir /s /q build
    mkdir build
)

if "%1"=="config" (
    echo Configuring with CMake...
    cmake -S . -B build -G Ninja ^
        -DCMAKE_CXX_COMPILER="D:/Development/VSBuildTools/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64/cl.exe" ^
        -DCMAKE_CUDA_COMPILER="D:/Development/CUDA/v12.6/bin/nvcc.exe" ^
        -DCMAKE_CUDA_HOST_COMPILER="D:/Development/VSBuildTools/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64/cl.exe" ^
        -DCMAKE_CUDA_ARCHITECTURES=61
    goto :end
)

if "%1"=="test" (
    echo Running tests...
    cmake --build build --parallel
    ctest --test-dir build --output-on-failure
    goto :end
)

echo Building ForgeRT...
cmake --build build --parallel

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Build succeeded!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    exit /b 1
)

:end
