# Object Detection for Blind Assistance (Educational Prototype)

A Python + Flask + OpenCV + YOLO object-detection prototype. It uses a webcam, detects common objects, draws boxes around them, and reads detected labels aloud using the browser's built-in speech synthesis.

## Requirements
- Windows 10/11
- Python 3.10–3.12 recommended for smoother ML package compatibility
- Webcam
- Internet connection on first run so Ultralytics can download `yolov8n.pt`

## Run on Windows PowerShell

1. Extract the ZIP.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python app.py
   ```

   If Python 3.11 is not installed, use `py -m venv .venv` and ensure your installed Python version is supported by PyTorch/Ultralytics.

5. Open `http://127.0.0.1:5000` in your browser.
6. Keep the terminal open while using the application. Allow camera permissions if asked.

## How it works
- `app.py`: Flask website, webcam stream, JSON detections.
- `detector.py`: loads YOLO and identifies objects in camera frames.
- `templates/` is not required; the simple HTML interface is embedded in `app.py`.
- Browser speech synthesis speaks the object labels. Audio output depends on browser/device settings.

## Troubleshooting
- **Camera unavailable:** close Windows Camera/Zoom/Teams, check Windows camera privacy settings, or change `CAMERA_INDEX` environment variable to `1`.
- **First run is slow:** the model weights download and the model initializes.
- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate the venv again.
- **Install errors:** use a supported Python version (Python 3.11 is a good default) and update pip.

## Important safety limitation
This project is a classroom demonstration, not a certified assistive device. It does not reliably detect every obstacle, estimate safe distances, identify drop-offs/stairs, or provide safe navigation instructions. Do not use it as the only aid for walking or crossing roads. Test only in a controlled environment with a sighted helper.
