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
    exit /b 2
)

if not exist "%PLOT_PYTHON%" (
    echo ERROR: stable base-Anaconda plotting Python was not found:
    echo   %PLOT_PYTHON%
    echo Set TOXICITY_PLOT_PYTHON to the correct base python.exe.
    exit /b 3
)

set "TOX_SCREEN_PROFILE=smoke"
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

echo Project root: %CD%
echo Training Python: %PYTHON_EXE%
echo Plotting Python: %PLOT_PYTHON%
echo Profile: %TOX_SCREEN_PROFILE%
echo Notebooks: 12 through 25
echo.
echo Add --resume to skip notebooks that previously passed with an identical SHA256.
echo.

echo Applying validated Notebook 12-25 hardening payload...
"%PYTHON_EXE%" -X faulthandler scripts\apply_full_run_hardening.py --root "%CD%"
if not "%ERRORLEVEL%"=="0" (
    echo ERROR: hardening payload installation failed.
    exit /b 4
)
echo.

"%PYTHON_EXE%" -X faulthandler scripts\run_remaining_notebooks.py ^
    --root "%CD%" ^
    --start 12 ^
    --end 25 ^
    --profile "%TOX_SCREEN_PROFILE%" ^
    --expected-python "%PYTHON_EXE%" ^
    --plot-python "%PLOT_PYTHON%" ^
    --cell-timeout 14400 ^
    %*

set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo All selected notebooks completed successfully.
) else (
    echo Notebook execution stopped or completed with errors.
    echo Review reports\remaining_notebooks_execution_summary.csv
    echo and logs\notebook_runs\ for the exact failure.
)

exit /b %EXIT_CODE%
