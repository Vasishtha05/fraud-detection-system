
# ==========================================
# FRAUD DETECTION SYSTEM - ONE COMMAND START
# ==========================================

$ErrorActionPreference = "Continue"

$ProjectPath = "C:\Users\Victus\PycharmProjects\fraud-detection"
$DockerDesktop = "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe"
$Docker = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe"
$ContainerName = "fraud-api"
$ImageName = "fraud-detection-api"

# Project URLs
$HealthUrl = "http://localhost:8000/health"
$SwaggerUrl = "http://localhost:8000/docs"
$DashboardUrl = "file:///C:/Users/Victus/PycharmProjects/fraud-detection/frontend/index.html"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "       FRAUD DETECTION SYSTEM" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check project folder
if (-not (Test-Path $ProjectPath)) {
    Write-Host "Project folder not found: $ProjectPath" -ForegroundColor Red
    exit 1
}
Set-Location $ProjectPath

# 2. Open Docker Desktop
Write-Host "`n[1/5] Starting Docker Desktop..." -ForegroundColor Yellow

if (-not (Test-Path $Docker)) {
    Write-Host "Docker CLI not found: $Docker" -ForegroundColor Red
    exit 1
}

if (-not (Get-Process "Docker Desktop" -ErrorAction SilentlyContinue)) {
    if (Test-Path $DockerDesktop) {
        Start-Process $DockerDesktop
    } else {
        Write-Host "Docker Desktop executable not found." -ForegroundColor Red
        exit 1
    }
}

# 3. Wait for Docker Engine
Write-Host "[2/5] Waiting for Docker Engine..." -ForegroundColor Yellow

$DockerReady = $false

for ($i = 0; $i -lt 60; $i++) {
    & $Docker info 2>$null | Out-Null

    if ($LASTEXITCODE -eq 0) {
        $DockerReady = $true
        break
    }

    Start-Sleep -Seconds 2
}

if (-not $DockerReady) {
    Write-Host "Docker Engine did not become ready." -ForegroundColor Red
    Write-Host "Open Docker Desktop, wait until Engine running, then type fraud again." -ForegroundColor Yellow
    exit 1
}

Write-Host "Docker Engine is ready." -ForegroundColor Green

# 4. Start or create API container
Write-Host "[3/5] Starting Fraud Detection API..." -ForegroundColor Yellow

$ExistingContainer = & $Docker ps -a --filter "name=^/$ContainerName$" --format "{{.Names}}"

if ($ExistingContainer -eq $ContainerName) {
    $RunningContainer = & $Docker ps --filter "name=^/$ContainerName$" --format "{{.Names}}"

    if ($RunningContainer -ne $ContainerName) {
        & $Docker start $ContainerName
    }
} else {
    if (-not (& $Docker image inspect $ImageName 2>$null)) {
        Write-Host "Building the Docker image..." -ForegroundColor Yellow
        & $Docker build -t $ImageName .

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Docker image build failed." -ForegroundColor Red
            exit 1
        }
    }

    # Create a persistent SQLite database mount
    if (-not (Test-Path "$ProjectPath\fraud_detection.db")) {
        New-Item -Path "$ProjectPath\fraud_detection.db" -ItemType File | Out-Null
    }

    & $Docker run -d `
        --name $ContainerName `
        -p 8000:8000 `
        -v "${ProjectPath}\fraud_detection.db:/app/fraud_detection.db" `
        $ImageName

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Container failed to start. Check Docker Desktop." -ForegroundColor Red
        exit 1
    }
}

# 5. Wait for API health check
Write-Host "[4/5] Checking API health..." -ForegroundColor Yellow

$ApiReady = $false

for ($i = 0; $i -lt 30; $i++) {
    try {
        $Health = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
        if ($Health.status -eq "healthy") {
            $ApiReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $ApiReady) {
    Write-Host "API did not become healthy. Recent container logs:" -ForegroundColor Red
    & $Docker logs --tail 40 $ContainerName
    exit 1
}

Write-Host "Fraud Detection API is healthy." -ForegroundColor Green

# 6. Open required pages
Write-Host "[5/5] Opening project pages..." -ForegroundColor Yellow

Start-Process $HealthUrl
Start-Sleep -Milliseconds 500
Start-Process $SwaggerUrl
Start-Sleep -Milliseconds 500
Start-Process $DashboardUrl

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "   FRAUD DETECTION SYSTEM IS READY!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Dashboard: $DashboardUrl"
Write-Host "Swagger:   $SwaggerUrl"
Write-Host "Health:    $HealthUrl"
Write-Host ""
