@echo off
setlocal enabledelayedexpansion

REM Initialize variables
set "limit=5"
set "episode=10000"
set /a "n=0"

REM Choose the parameter to test
echo Please choose a parameter to test:
echo 1. gamma
echo 2. epsilon
echo 3. epsilon_discount
echo 4. discover_percent
echo 5. discover_epsilon
echo 6. Run all parameters
set /p "choice=Enter your choice: "

REM Loop over the chosen parameter or all parameters
if "!choice!"=="6" (
    for /L %%i in (2,1,5) do (
        set "choice=%%i"
        call :run_parameter
    )
) else (
    call :run_parameter
)

goto :eof

:run_parameter
if "!choice!"=="1" (
    REM Clear the folder
    del /Q MC_data\gamma\*
    for /L %%j in (0,1,10) do (
        set /a "n+=1" 
        echo Running with gamma: %%j
        python monted_carlo_method.py --n !n! --limit !limit! --gamma %%j --episode !episode! --v "gamma"
    )
)
if "!choice!"=="2" (
    REM Clear the folder
    del /Q MC_data\epsilon\*
    for /L %%j in (6,1,10) do (
        set /a "n+=1"
        echo Running with epsilon: %%j
        python monted_carlo_method.py --n !n! --limit !limit! --epsilon %%j --episode !episode! --v "epsilon"
    )
)
if "!choice!"=="3" (
    REM Clear the folder
    del /Q MC_data\epsilon_discount\*
    for /L %%k in (980,1,1000) do (
        set /a "n+=1"    
        echo Running with epsilon_discount: %%k
        python monted_carlo_method.py --n !n! --limit !limit! --epsilon_discount %%k --episode !episode! --v "epsilon_discount"
    )
)
if "!choice!"=="4" (
    REM Clear the folder
    del /Q MC_data\discover_percent\*
    for /L %%j in (0,1,8) do (
        set /a "n+=1"
        echo Running with discover_percent: %%j
        python monted_carlo_method.py --n !n! --limit !limit! --discover_percent %%j --episode !episode! --v "discover_percent"
    )
)
if "!choice!"=="5" (
    REM Clear the folder
    del /Q MC_data\discover_epsilon\*
    for /L %%j in (1,1,10) do (
        set /a "n+=1"
        echo Running with discover_epsilon: %%j
        python monted_carlo_method.py --n !n! --limit !limit! --discover_epsilon %%j --episode !episode! --v "discover_epsilon"
    )
)
goto :eof

REM Shumonted_carlo_methodown command
REM shumonted_carlo_methodown /s /t 60
