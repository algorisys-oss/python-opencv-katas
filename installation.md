# Installation Guide

Complete setup instructions for the OpenCV Interactive Playground on Windows, Linux, and macOS.

---

## Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.10+ | Backend, kata execution |
| **Node.js** | 18+ | Frontend build tooling |
| **npm** | 9+ | Package management (comes with Node.js) |
| **Git** | 2.30+ | Version control |
| **Webcam** (optional) | Any USB/built-in | Live camera katas (100+) |

---

## 1. Python Installation

### Windows

1. Download Python from [python.org](https://www.python.org/downloads/).
2. **Important:** Check "Add Python to PATH" during installation.
3. Verify:
   ```powershell
   python --version
   pip --version
   ```

Alternatively, use the Microsoft Store:
```powershell
winget install Python.Python.3.12
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

### macOS

Using Homebrew (recommended):
```bash
brew install python@3.12
python3 --version
```

Or download from [python.org](https://www.python.org/downloads/macos/).

---

## 2. Node.js Installation

### Windows

Download from [nodejs.org](https://nodejs.org/) (LTS version) or use:
```powershell
winget install OpenJS.NodeJS.LTS
```

### Linux (Ubuntu/Debian)

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs
node --version
npm --version
```

### macOS

```bash
brew install node@20
node --version
npm --version
```

---

## 3. Clone the Repository

```bash
git clone https://github.com/rajeshpillai/opencv-python-katas.git
cd opencv-python-katas
```

---

## 4. Backend Setup

### Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for the API |
| `uvicorn` | ASGI server to run FastAPI |
| `opencv-python-headless` | OpenCV without GUI (for sandbox execution) |
| `numpy` | Array operations, image processing |
| `python-frontmatter` | Parsing kata Markdown + YAML frontmatter |
| `python-multipart` | Form data handling |
| `bcrypt` | Password hashing for auth |
| `pyjwt` | JWT token generation |

### Install OpenCV with GUI Support (for Live Camera Katas)

The backend uses `opencv-python-headless` for sandboxed execution. To run **live camera katas locally** (100+), you need the full OpenCV with GUI support:

```bash
# Install alongside headless (for local testing)
pip install opencv-python
```

> **Note:** `opencv-python` and `opencv-python-headless` can conflict. If you see import errors, uninstall both and reinstall just `opencv-python`:
> ```bash
> pip uninstall opencv-python opencv-python-headless
> pip install opencv-python
> ```

### Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`. The database and kata seeds are created automatically on first startup.

---

## 5. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server starts at `http://localhost:5173` (default Vite port).

### Frontend Dependencies

Installed automatically via `npm install`:

| Package | Purpose |
|---------|---------|
| `solid-js` | UI framework |
| `@solidjs/router` | Client-side routing |
| `vite` | Build tool and dev server |
| `vite-plugin-solid` | SolidJS support for Vite |
| `tailwindcss` | Utility-first CSS (v4) |
| `monaco-editor` | Code editor component |

---

## 6. Camera Setup (for Live Camera Katas 100+)

### Verify Camera Access

Run this quick test to confirm your camera works with OpenCV:

```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    print(f'Camera OK: {frame.shape}' if ret else 'Camera opened but no frame')
    cap.release()
else:
    print('No camera found at index 0')
"
```

### Windows

- Built-in webcams work out of the box.
- USB cameras are detected automatically.
- If the camera isn't detected, check **Device Manager > Cameras** and ensure drivers are installed.
- Some antivirus software blocks camera access — add Python to the allow list.

### Linux

Camera access requires the `video` group:

```bash
# Check if your user is in the video group
groups $USER

# Add yourself to the video group (if not already)
sudo usermod -aG video $USER
# Log out and back in for the change to take effect
```

Install Video4Linux utilities for debugging:
```bash
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

If you're running in a **virtual machine**, camera passthrough must be enabled in the VM settings.

### macOS

- macOS may prompt for **camera permission** the first time Python accesses the camera. Click "Allow".
- If permission was denied, re-enable it in **System Preferences > Security & Privacy > Camera**.
- The first frame capture can take 2-3 seconds while the driver initializes — this is normal.

### Multiple Cameras

If you have multiple cameras (built-in + USB), try different indices:

```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'Camera {i}: {frame.shape[1]}x{frame.shape[0]}')
        cap.release()
"
```

### Running Live Camera Katas Locally

Live camera katas (100+) are designed to run **outside the sandbox** — directly on your machine with a real camera:

```bash
# From the project root, with the virtual environment activated
python backend/data/katas/scripts/run-kata.py 100
# Or run the starter code directly:
python -c "$(python -c "
import frontmatter, re
post = frontmatter.load('backend/data/katas/100-live-camera-fps.md')
match = re.search(r'\`\`\`python\n(.*?)\`\`\`', post.content, re.DOTALL)
print(match.group(1) if match else '')
")"
```

Or simply copy the starter code from the web UI and run it in your terminal.

---

## 7. Platform-Specific Notes

### Windows

| Issue | Solution |
|-------|----------|
| `cv2.imshow` window doesn't appear | Ensure you're not running in WSL (use native Windows Python) |
| `ImportError: DLL load failed` | Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Camera permission denied | Run terminal as Administrator, or check Windows privacy settings |
| `pip` not found | Reinstall Python with "Add to PATH" checked |

### Linux

| Issue | Solution |
|-------|----------|
| `cv2.imshow` crashes or no window | Install GUI backend: `sudo apt install libgtk-3-dev` then reinstall OpenCV |
| `libGL.so.1 not found` | `sudo apt install libgl1-mesa-glx` |
| No camera device | Check `ls /dev/video*` and ensure `v4l2` module is loaded |
| Display errors over SSH | Use `ssh -X` for X11 forwarding, or use VNC |

### macOS

| Issue | Solution |
|-------|----------|
| `cv2.imshow` crashes on M1/M2 | Install OpenCV via pip (not conda); ensure arm64 Python |
| Camera permission denied | System Preferences > Security & Privacy > Camera > Allow Terminal/Python |
| Slow first frame | Normal — macOS camera driver init takes 2-3 seconds |

---

## 8. Verifying the Full Setup

Run these checks to confirm everything works:

```bash
# 1. Python + OpenCV
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# 2. NumPy
python -c "import numpy as np; print(f'NumPy {np.__version__}')"

# 3. FastAPI backend starts
uvicorn backend.main:app --port 8000 &
curl http://localhost:8000/api/katas | python -m json.tool | head -5
kill %1

# 4. Frontend builds
cd frontend && npm run build && cd ..

# 5. Camera (optional)
python -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera:', 'OK' if cap.isOpened() else 'Not found')
cap.release()
"
```

---

## 9. Updating

To get the latest katas and platform updates:

```bash
git pull origin main
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
```

If you encounter database issues after an update, delete the SQLite database to reseed:

```bash
rm backend/data/opencv-katas.db
# Restart the backend — it will recreate and reseed automatically
```
