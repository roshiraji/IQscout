@echo off
REM One-click local smoke test (no cloud, no Colab).
REM Runs: validate -> mock pipeline -> single real image with previews.
cd /d "%~dp0"
set PY=%~dp0..\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo ERROR: Virtual env not found at ..\.venv
    echo Run scripts\setup.bat from repo root first.
    exit /b 1
)

echo === Step 1/3: Validate environment ===
"%PY%" -m auto_annotation.run --validate-only
if errorlevel 1 exit /b 1

echo.
echo === Step 2/3: Mock pipeline (no GPU models) ===
"%PY%" -m auto_annotation.run --mock --limit 2 --config config\auto_annotation_dog_demo.yaml --overwrite
if errorlevel 1 exit /b 1

echo.
echo === Step 3/3: Real inference on dog-3.jpeg ===
"%PY%" -m auto_annotation.run --config config\auto_annotation_dog_demo.yaml --single-image data\input_images\dog-3.jpeg --overwrite --save-previews
if errorlevel 1 exit /b 1

echo.
echo === Local test complete ===
echo Previews:  data\auto_annotated_dog_demo\previews\
echo Reports:   data\auto_annotated_dog_demo\reports\
echo Annotations: data\auto_annotated_dog_demo\annotations\
echo.
echo Open a preview JPG in data\auto_annotated_dog_demo\previews\ to verify detections.
