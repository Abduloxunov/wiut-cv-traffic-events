@echo off
REM One-time setup on a Windows machine with an NVIDIA GPU. Run from the project folder: setup_lab.bat
REM Needs Python 3.10+ on PATH and internet for the package downloads.
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
REM CUDA build of PyTorch (plain "pip install torch" on Windows is CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || goto :error
pip install -r requirements.txt -r train\requirements-train.txt || goto :error
python train\check_gpu.py || goto :error
echo.
echo Setup done. Next time just run: .venv\Scripts\activate.bat
goto :eof
:error
echo Setup failed at the step above.
exit /b 1
