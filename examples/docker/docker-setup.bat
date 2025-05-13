@echo off
echo ===================================
echo finetrainers Docker Setup
echo ===================================
echo.
 
REM Check if Docker is installed
docker -v > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or Docker Desktop is not running.
    echo 1. Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    echo 2. If Docker Desktop is already installed, please start it.
    echo   (Search for "Docker Desktop" in the Start menu and launch it)
    exit /b 1
)
 
REM Check if Docker Desktop is actually running (additional check)
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Cannot connect to Docker Desktop engine.
    echo Please make sure Docker Desktop is running.
    echo If the Docker icon is not shown in the task tray, start it from the Start menu.
    exit /b 1
)
 
REM Check if Docker Compose is installed
docker-compose -v > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not installed.
    echo Docker Desktop usually includes Docker Compose.
    exit /b 1
)
 
echo [INFO] Docker is installed correctly.
 
REM Create folders
if not exist "datasets" (
    echo [INFO] Creating datasets folder...
    mkdir datasets
)
 
if not exist "outputs" (
    echo [INFO] Creating outputs folder...
    mkdir outputs
)
 
echo.
echo [INFO] Building Docker image...
docker-compose build
 
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build Docker image.
    echo Possible causes:
    echo - Docker Desktop is not running properly
    echo - WSL2 configuration issues
    echo - Insufficient disk space
    echo Please check Docker Desktop settings and restart Windows if necessary.
    exit /b 1
)
 
echo.
echo [INFO] Setup completed!
echo.
echo Usage:
echo  1. Make sure Docker Desktop is running
echo  2. docker-compose up -d        (Start the container)
echo  3. docker exec -it finetrainers bash  (Connect to the container)
echo  4. docker-compose down         (Stop the container)
echo.
echo For details, see README-docker.md.
echo.