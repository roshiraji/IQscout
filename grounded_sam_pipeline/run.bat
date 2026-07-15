@echo off
REM Run the auto-annotation CLI from grounded_sam_pipeline/
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" -m auto_annotation.run %*
