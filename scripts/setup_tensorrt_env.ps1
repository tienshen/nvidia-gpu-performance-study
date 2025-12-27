# TensorRT Environment Setup Script
# This script sets up environment variables needed for TensorRT

Write-Host "Setting up TensorRT environment..." -ForegroundColor Cyan

# Common TensorRT installation paths (including versioned directories)
$tensorrtPaths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT",
    "C:\TensorRT",
    "$env:USERPROFILE\TensorRT",
    "C:\Program Files\NVIDIA\TensorRT"
)

# Find TensorRT installation
$tensorrtRoot = $null

# First check exact paths
foreach ($path in $tensorrtPaths) {
    if (Test-Path $path) {
        Write-Host "Found TensorRT at: $path" -ForegroundColor Green
        $tensorrtRoot = $path
        break
    }
}

# If not found, search for versioned TensorRT directories
if (-not $tensorrtRoot) {
    Write-Host "Searching for versioned TensorRT installations..." -ForegroundColor Yellow
    $versionedDirs = Get-ChildItem -Path "C:\Program Files" -Filter "TensorRT-*" -Directory -ErrorAction SilentlyContinue
    if ($versionedDirs) {
        $tensorrtRoot = $versionedDirs[0].FullName
        Write-Host "Found TensorRT at: $tensorrtRoot" -ForegroundColor Green
    }
}

# If not found, search for versioned TensorRT directories
if (-not $tensorrtRoot) {
    Write-Host "Searching for versioned TensorRT installations..." -ForegroundColor Yellow
    $versionedDirs = Get-ChildItem -Path "C:\Program Files" -Filter "TensorRT-*" -Directory -ErrorAction SilentlyContinue
    if ($versionedDirs) {
        $tensorrtRoot = $versionedDirs[0].FullName
        Write-Host "Found TensorRT at: $tensorrtRoot" -ForegroundColor Green
    }
}

# If not found in common locations, search for nvinfer_10.dll
if (-not $tensorrtRoot) {
    Write-Host "Searching for TensorRT DLLs..." -ForegroundColor Yellow
    $foundDll = Get-ChildItem -Path "C:\Program Files" -Filter "nvinfer*.dll" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($foundDll) {
        $tensorrtRoot = $foundDll.Directory.Parent.FullName
        Write-Host "Found TensorRT DLLs at: $tensorrtRoot" -ForegroundColor Green
    }
}

if (-not $tensorrtRoot) {
    Write-Host "ERROR: Could not find TensorRT installation." -ForegroundColor Red
    Write-Host "Please specify the TensorRT installation path manually:" -ForegroundColor Yellow
    Write-Host "  `$env:TENSORRT_PATH = 'C:\path\to\tensorrt'" -ForegroundColor Yellow
    Write-Host "  `$env:PATH = `"`$env:TENSORRT_PATH\lib;`$env:PATH`"" -ForegroundColor Yellow
    exit 1
}

# Set TensorRT environment variables
$env:TENSORRT_PATH = $tensorrtRoot

# Add TensorRT lib directories to PATH
$libPaths = @()

# Check for lib directory
$libDir = Join-Path $tensorrtRoot "lib"
if (Test-Path $libDir) {
    $libPaths += $libDir
}

# Check for bin directory
$binDir = Join-Path $tensorrtRoot "bin"
if (Test-Path $binDir) {
    $libPaths += $binDir
}

# Check for versioned lib directories (e.g., lib/x64)
$libX64 = Join-Path $tensorrtRoot "lib\x64"
if (Test-Path $libX64) {
    $libPaths += $libX64
}

# Add all found paths to PATH
foreach ($libPath in $libPaths) {
    if ($env:PATH -notlike "*$libPath*") {
        $env:PATH = "$libPath;$env:PATH"
        Write-Host "Added to PATH: $libPath" -ForegroundColor Green
    } else {
        Write-Host "Already in PATH: $libPath" -ForegroundColor Gray
    }
}

# Verify TensorRT DLLs are accessible
Write-Host "`nVerifying TensorRT DLLs..." -ForegroundColor Cyan
$requiredDlls = @("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll")
$allFound = $true

foreach ($dll in $requiredDlls) {
    $found = $false
    foreach ($libPath in $libPaths) {
        $dllPath = Join-Path $libPath $dll
        if (Test-Path $dllPath) {
            Write-Host "  [OK] $dll found at $dllPath" -ForegroundColor Green
            $found = $true
            break
        }
    }
    if (-not $found) {
        Write-Host "  [MISSING] $dll not found" -ForegroundColor Red
        $allFound = $false
    }
}

if ($allFound) {
    Write-Host "`nTensorRT environment configured successfully!" -ForegroundColor Green
    Write-Host "TENSORRT_PATH = $env:TENSORRT_PATH" -ForegroundColor Cyan
} else {
    Write-Host "`nWARNING: Some TensorRT DLLs were not found." -ForegroundColor Yellow
    Write-Host "You may need to install TensorRT or check your installation." -ForegroundColor Yellow
}

# Setup Nsight Systems
Write-Host "`nSearching for Nsight Systems..." -ForegroundColor Cyan

# First try to find standalone Nsight Systems installations (prefer newer versions)
$nsysInstallations = Get-ChildItem -Path "C:\Program Files\NVIDIA Corporation" -Directory -ErrorAction SilentlyContinue | 
    Where-Object {$_.Name -like "Nsight Systems*"} | 
    Sort-Object Name -Descending

$nsysPath = $null
$nsysUiPath = $null
if ($nsysInstallations) {
    # Use the newest version (sorted descending)
    $newestInstall = $nsysInstallations[0]
    $nsysPath = Get-ChildItem -Path $newestInstall.FullName -Filter "nsys.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    $nsysUiPath = Get-ChildItem -Path $newestInstall.FullName -Filter "nsys-ui.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
}

# Fallback to searching anywhere if standalone not found
if (-not $nsysPath) {
    $nsysPath = Get-ChildItem -Path "C:\Program Files" -Filter "nsys.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $nsysUiPath) {
    $nsysUiPath = Get-ChildItem -Path "C:\Program Files" -Filter "nsys-ui.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
}

if ($nsysPath) {
    $nsysDir = $nsysPath.DirectoryName
    if ($env:PATH -notlike "*$nsysDir*") {
        $env:PATH = "$nsysDir;$env:PATH"
        Write-Host "Added Nsight Systems to PATH: $nsysDir" -ForegroundColor Green
    } else {
        Write-Host "Nsight Systems already in PATH" -ForegroundColor Gray
    }
    Write-Host "Nsight Systems: $(& $nsysPath.FullName --version 2>&1 | Select-Object -First 1)" -ForegroundColor Cyan
} else {
    Write-Host "Nsight Systems (nsys) not found. GPU profiling will not be available." -ForegroundColor Yellow
}

if ($nsysUiPath) {
    $nsysUiDir = $nsysUiPath.DirectoryName
    if ($env:PATH -notlike "*$nsysUiDir*") {
        $env:PATH = "$nsysUiDir;$env:PATH"
        Write-Host "Added Nsight Systems UI to PATH: $nsysUiDir" -ForegroundColor Green
    } else {
        Write-Host "Nsight Systems UI already in PATH" -ForegroundColor Gray
    }
} else {
    Write-Host "Nsight Systems UI (nsys-ui) not found." -ForegroundColor Yellow
}

Write-Host "`nTo make these changes permanent for this session, source this script:" -ForegroundColor Yellow
Write-Host "  . .\scripts\setup_tensorrt_env.ps1" -ForegroundColor Cyan
Write-Host "`nTo make changes permanent across sessions, add to your PowerShell profile:" -ForegroundColor Yellow
Write-Host "  notepad `$PROFILE" -ForegroundColor Cyan
