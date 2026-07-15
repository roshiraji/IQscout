@echo off
REM Download SAM checkpoints into scoutiq\models\sam\
setlocal
cd /d "%~dp0.."

set MODEL=%~1
if "%MODEL%"=="" set MODEL=vit_b

if /I "%MODEL%"=="vit_b" (
  set URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  set FILE=sam_vit_b_01ec64.pth
) else if /I "%MODEL%"=="vit_l" (
  set URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  set FILE=sam_vit_l_0b3195.pth
) else if /I "%MODEL%"=="vit_h" (
  set URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  set FILE=sam_vit_h_4b8939.pth
) else (
  echo Unknown model type: %MODEL%
  echo Usage: download_sam_checkpoint.bat [vit_b^|vit_l^|vit_h]
  exit /b 1
)

if not exist "models\sam" mkdir "models\sam"

if exist "models\sam\%FILE%" (
  echo Checkpoint already present: models\sam\%FILE%
  exit /b 0
)

echo Downloading %MODEL% checkpoint...
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile 'models\sam\%FILE%' -UseBasicParsing"
echo Saved to models\sam\%FILE%
