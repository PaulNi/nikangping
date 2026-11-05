@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
  echo [ERR] .venv not found. Create it first: python -m venv .venv
  exit /b 1
)

echo [1/3] Uninstall CPU torch if present...
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

echo [2/3] Install CUDA 12.4 PyTorch wheels...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
  echo [ERR] CUDA torch install failed
  exit /b 1
)

echo [3/3] Install project deps...
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERR] requirements install failed
  exit /b 1
)

echo.
.\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.
echo Done. Start the bot; look for: [LieAlert] YOLO device=cuda:0
endlocal
