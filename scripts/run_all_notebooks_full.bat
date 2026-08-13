@echo off
setlocal EnableExtensions

cd /d "%~dp0.."

if defined TOXICITY_PYTHON (
    set "PYTHON_EXE=%TOXICITY_PYTHON%"
) else (
    set "PYTHON_EXE=D:\Users\anaconda3\envs\toxicity-screening\python.exe"
)

if defined TOXICITY_PLOT_PYTHON (
    set "PLOT_PYTHON=%TOXICITY_PLOT_PYTHON%"
) else (
    set "PLOT_PYTHON=D:\Users\anaconda3\python.exe"
)

if not exist "%PYTHON_EXE%" (
    echo ERROR: toxicity-screening Python was not found:
    echo   %PYTHON_EXE%
    echo Set TOXICITY_PYTHON to the correct environment python.exe.
    exit /b 2
)

if not exist "%PLOT_PYTHON%" (
    echo ERROR: stable base-Anaconda plotting Python was not found:
    echo   %PLOT_PYTHON%
    echo Set TOXICITY_PLOT_PYTHON to the correct base python.exe.
    exit /b 3
)

set "TOX_SCREEN_PROFILE=full"
set "MPLBACKEND=Agg"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "VECLIB_MAXIMUM_THREADS=1"
set "BLIS_NUM_THREADS=1"
set "TOKENIZERS_PARALLELISM=false"
set "PYTHONNOUSERSITE=1"
set "PYTHONFAULTHANDLER=1"
set "PYTHONUNBUFFERED=1"
set "TOXICITY_PLOT_PYTHON=%PLOT_PYTHON%"

if not defined PYTHONPATH (
    set "PYTHONPATH=%CD%\src"
) else (
    set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
)

echo ===============================================================
echo Toxicity Screening - FULL profile notebook execution
echo ===============================================================
echo Project root:     %CD%
echo Training Python:  %PYTHON_EXE%
echo Plotting Python:  %PLOT_PYTHON%
echo Profile:          %TOX_SCREEN_PROFILE%
echo Notebooks:        00 through 25
echo Cell timeout:     43200 seconds ^(12 hours^)
echo.
echo The runner stops at the first failed notebook.
echo Restart with --resume to skip notebooks that previously passed
echo with the same notebook SHA256.
echo.

"%PYTHON_EXE%" -X faulthandler scripts\run_all_notebooks_full.py ^
    --root "%CD%" ^
    --start 0 ^
    --end 25 ^
    --profile full ^
    --expected-python "%PYTHON_EXE%" ^
    --cell-timeout 43200 ^
    %*

set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo All full-profile notebooks completed successfully.
    echo Review reports\remaining_notebooks_execution_summary.csv
) else (
    echo Full-profile execution stopped or completed with errors.
    echo Review reports\remaining_notebooks_execution_summary.csv
    echo and logs\notebook_runs\ for the exact failure.
    echo After correcting the issue, restart with:
    echo   scripts\run_all_notebooks_full.bat --resume
)

echo.
exit /b %EXIT_CODE%
