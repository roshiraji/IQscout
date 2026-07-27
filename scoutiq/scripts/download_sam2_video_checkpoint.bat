@echo off
REM Download the SAM 2.1 video-predictor checkpoint into scoutiq\models\sam2\
REM Confirm uses this to propagate the clicked player masks through contact.
setlocal
cd /d "%~dp0.."

set MODEL=%~1
if "%MODEL%"=="" set MODEL=base_plus

if /I "%MODEL%"=="tiny" (
  set FILE=sam2.1_hiera_tiny.pt
) else if /I "%MODEL%"=="small" (
  set FILE=sam2.1_hiera_small.pt
) else if /I "%MODEL%"=="base_plus" (
  set FILE=sam2.1_hiera_base_plus.pt
) else if /I "%MODEL%"=="large" (
  set FILE=sam2.1_hiera_large.pt
) else (
  echo Unknown model size: %MODEL%
  echo Usage: download_sam2_video_checkpoint.bat [tiny^|small^|base_plus^|large]
  exit /b 1
)

set URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/%FILE%

if not exist "models\sam2" mkdir "models\sam2"

if exist "models\sam2\%FILE%" (
  echo Checkpoint already present: models\sam2\%FILE%
  exit /b 0
)

echo Downloading SAM 2.1 %MODEL% video checkpoint...
curl -L --fail -o "models\sam2\%FILE%" "%URL%"
if errorlevel 1 (
  echo Download failed: %URL%
  exit /b 1
)
echo Saved to models\sam2\%FILE%
