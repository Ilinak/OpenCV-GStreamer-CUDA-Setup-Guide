# 🚀 OpenCV with GStreamer and CUDA — Setup Guide (Ubuntu)

A comprehensive, battle-tested guide to building **OpenCV** from source with **GStreamer**, **CUDA**, and **cuDNN** support on Ubuntu. Use this to replicate a high-performance environment for video processing and deep learning on NVIDIA GPUs.

> **Tested On:** Ubuntu 22.04 · CUDA 12.8 · OpenCV 4.13.0-dev  
> **Last Updated:** August 2025

---

## 📝 Table of Contents

- [Prerequisites](#-prerequisites)
  - [System Requirements](#-system-requirements)
  - [Environment Setup](#-environment-setup)
- [Installation Steps](#-installation-steps)
  - [Step 1: Update System and Install Build Dependencies](#step-1-update-system-and-install-build-dependencies)
  - [Step 2: Install OpenCV Dependencies](#step-2-install-opencv-dependencies)
  - [Step 3: Download OpenCV Source Code](#step-3-download-opencv-source-code)
  - [Step 4: Extract Source Code](#step-4-extract-source-code)
  - [Step 5: Create Build Directory](#step-5-create-build-directory)
  - [Step 6: Configure Build with CMake](#step-6-configure-build-with-cmake)
  - [Step 7: Compile OpenCV](#step-7-compile-opencv)
  - [Step 8: Install OpenCV](#step-8-install-opencv)
  - [Step 9: Update Library Cache](#step-9-update-library-cache)
- [Configuration Options](#-configuration-options)
- [Verification](#-verification)
  - [Method 1: Detailed Build Information](#method-1-detailed-build-information)
  - [Method 2: Quick Functional Test](#method-2-quick-functional-test)
  - [Method 3: Test GStreamer Pipeline](#method-3-test-gstreamer-pipeline)
- [GPU Architecture Reference](#-gpu-architecture-reference)
- [Troubleshooting](#-troubleshooting)
  - [Common Issues and Solutions](#common-issues-and-solutions)
  - [Clean Build (If Needed)](#clean-build-if-needed)
- [Additional Resources](#-additional-resources)
- [Support](#-support)
- [How to Use This Repository](#-how-to-use-this-repository)

---

## ✅ Prerequisites

Ensure your system meets the following requirements before starting.

### Required Software

- **Ubuntu:** 18.04 or newer (tested on **Ubuntu 22.04**)
- **NVIDIA GPU:** CUDA compute capability **3.0+**
- **CUDA Toolkit:** Installed (>= 10.0; **12.x recommended**)
- **cuDNN:** Installed and compatible with your CUDA Toolkit
- **Internet connection** for downloading sources and dependencies

### Hardware Requirements

- **CPU:** Multi-core (4+ cores recommended)
- **RAM:** Minimum 4GB (8GB+ recommended)
- **Storage:** At least 10GB free (20GB+ recommended)
- **GPU:** NVIDIA GPU with compute capability matching your `CUDA_ARCH_BIN`

### Environment Setup

- **Conda Environment:** Assumes a conda env named `sentinel` with **Python 3.10**.  
  Adjust paths if using a different environment.
- **Root Access:** Required for installing system dependencies.

---

## 💻 System Requirements

| Component       | Minimum | Recommended |
|----------------|---------:|------------:|
| Ubuntu Version | 18.04    | 22.04+      |
| RAM            | 4GB      | 8GB+        |
| Storage        | 10GB     | 20GB+       |
| CPU Cores      | 2        | 4+          |
| CUDA Version   | 10.0     | 12.x        |
| Python Version | 3.7      | 3.10+       |

---

## 🛠️ Installation Steps

Follow these steps to build and install OpenCV from source with GStreamer, CUDA, and cuDNN support.

### Step 1: Update System and Install Build Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y cmake g++ wget unzip
```

**What this does**

- `sudo apt update && sudo apt upgrade -y`: Updates package lists and upgrades installed packages.
- `sudo apt install -y cmake g++ wget unzip`: Installs **CMake**, **GNU C++**, **wget**, and **unzip**.

---

### Step 2: Install OpenCV Dependencies

```bash
sudo apt install -y     build-essential     libgtk2.0-dev     pkg-config     libavcodec-dev     libavformat-dev     libswscale-dev     python3-dev     python3-numpy     libtbbmalloc2     libtbb-dev     libdc1394-dev     libv4l-dev     libgstreamer1.0-dev     libgstreamer-plugins-base1.0-dev
```

**Package explanations**

- `build-essential`: gcc, g++, make
- `libgtk2.0-dev`: GUI support (GTK+ 2.0 development files)
- `pkg-config`: Manages compile/link flags for libraries
- `libavcodec-dev`, `libavformat-dev`, `libswscale-dev`: FFmpeg core libs
- `python3-dev`: Python 3 headers/static libs
- `python3-numpy`: NumPy (required for Python bindings)
- `libtbbmalloc2`, `libtbb-dev`: Intel TBB for parallelism
- `libdc1394-dev`: IEEE 1394 (FireWire) camera support
- `libv4l-dev`: Video4Linux camera support
- `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev`: GStreamer dev files

> **Tip:** If using Ubuntu 22.04+, some users prefer `libgtk-3-dev`. Keep as-is unless you need GTK3 specifically.

---

### Step 3: Download OpenCV Source Code

Create a working directory, then download both the **opencv** and **opencv_contrib** sources.

```bash
# Create a working directory
mkdir -p ~/opencv_build && cd ~/opencv_build

# Download OpenCV main repository
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip

# Download OpenCV contrib modules
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
```

**Why contrib modules?** They include extra algorithms (including **non-free** ones like **SIFT**/**SURF**) and cutting-edge features.

---

### Step 4: Extract Source Code

```bash
unzip opencv.zip
unzip opencv_contrib.zip

# Verify extraction
ls -la
```

Expected directories: `opencv-4.x/` and `opencv_contrib-4.x/`.

---

### Step 5: Create Build Directory

```bash
cd opencv-4.x
mkdir -p build && cd build
```

**Why a separate build dir?** Keeps source clean, enables multiple configs, and makes cleanup easy.

---

### Step 6: Configure Build with CMake

> ⚠️ **Important:** Activate your conda environment first:
>
> ```bash
> conda activate sentinel
> ```

Now configure with CMake (adjust paths to your environment):

```bash
cmake     -D CMAKE_BUILD_TYPE=RELEASE     -D CMAKE_INSTALL_PREFIX=/home/sentinel/miniconda3/envs/sentinel     -D INSTALL_PYTHON_EXAMPLES=ON     -D INSTALL_C_EXAMPLES=ON     -D WITH_TBB=ON     -D ENABLE_FAST_MATH=1     -D CUDA_FAST_MATH=1     -D WITH_CUBLAS=1     -D WITH_CUDA=ON     -D BUILD_opencv_cudacodec=OFF     -D WITH_CUDNN=ON     -D OPENCV_DNN_CUDA=ON     -D WITH_V4L=ON     -D WITH_QT=OFF     -D BUILD_opencv_apps=OFF     -D BUILD_opencv_python2=OFF     -D BUILD_opencv_python3=ON     -D OPENCV_GENERATE_PKGCONFIG=ON     -D OPENCV_PC_FILE_NAME=opencv.pc     -D OPENCV_ENABLE_NONFREE=ON     -D WITH_OPENGL=OFF     -D WITH_GSTREAMER=ON     -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.x/modules     -D PYTHON3_EXECUTABLE=/home/sentinel/miniconda3/envs/sentinel/bin/python3     -D PYTHON3_INCLUDE_DIR=/home/sentinel/miniconda3/envs/sentinel/include/python3.10     -D PYTHON3_LIBRARY=/home/sentinel/miniconda3/envs/sentinel/lib/libpython3.10.so     -D PYTHON3_NUMPY_INCLUDE_DIRS=/home/sentinel/miniconda3/envs/sentinel/lib/python3.10/site-packages/numpy/core/include     -D PYTHON3_PACKAGES_PATH=/home/sentinel/miniconda3/envs/sentinel/lib/python3.10/site-packages     -D BUILD_EXAMPLES=ON     -D CUDA_ARCH_BIN="8.6"     -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda     -D WITH_NVCUVID=OFF     -D WITH_NVCUVENC=OFF     ..
```

**Notes**

- **Adjust Paths:** Replace `/home/sentinel/miniconda3/envs/sentinel` with your actual conda env path for `CMAKE_INSTALL_PREFIX` and all `PYTHON3_*` entries.
- **`CUDA_ARCH_BIN`:** Set to your GPU’s compute capability (see [GPU Architecture Reference](#-gpu-architecture-reference)).
- The trailing `..` points to the parent dir (`opencv-4.x/`) containing `CMakeLists.txt`.

You should see a configuration summary in the console—review for warnings/errors.

---

### Step 7: Compile OpenCV

```bash
make -j$(nproc)
```

- `make`: Builds the project generated by CMake.  
- `-j$(nproc)`: Uses all CPU cores for faster compilation.

> ⏱️ **Compilation time:** ~30–90+ minutes depending on hardware.

---

### Step 8: Install OpenCV

```bash
sudo make install
```

Installs libraries, headers, and Python bindings into `CMAKE_INSTALL_PREFIX`.

---

### Step 9: Update Library Cache

```bash
sudo ldconfig
```

Rebuilds `/etc/ld.so.cache` so the system can find newly installed shared libraries.

---

## ⚙️ Configuration Options

### Core Build Options

| Option                | Value    | Description                                   |
|----------------------|----------|-----------------------------------------------|
| `CMAKE_BUILD_TYPE`   | RELEASE  | Optimized build (use `DEBUG` for debugging)   |
| `CMAKE_INSTALL_PREFIX` | /path/... | Install destination for OpenCV               |

### Python Integration

| Option                    | Value     | Description                                         |
|--------------------------|-----------|-----------------------------------------------------|
| `INSTALL_PYTHON_EXAMPLES`| ON        | Installs Python example scripts                     |
| `BUILD_opencv_python3`   | ON        | Builds Python 3 bindings                            |
| `PYTHON3_EXECUTABLE`     | /path/... | Path to Python 3 interpreter                        |
| `PYTHON3_PACKAGES_PATH`  | /path/... | Target `site-packages` for Python bindings          |

### Performance Options

| Option              | Value | Description                                 |
|--------------------|------:|---------------------------------------------|
| `WITH_TBB`         | ON    | Enable Intel TBB                            |
| `ENABLE_FAST_MATH` | 1     | Fast math (slight precision trade-offs)     |
| `CUDA_FAST_MATH`   | 1     | Fast math optimizations for CUDA            |

### CUDA/GPU Options

| Option             | Value   | Description                                              |
|-------------------|---------|----------------------------------------------------------|
| `WITH_CUDA`       | ON      | Enable CUDA support                                      |
| `WITH_CUBLAS`     | 1       | Enable cuBLAS                                           |
| `WITH_CUDNN`      | ON      | Enable cuDNN                                            |
| `OPENCV_DNN_CUDA` | ON      | Use CUDA backend for DNN module                          |
| `CUDA_ARCH_BIN`   | "8.6"   | Target GPU arch (adjust to your GPU)                     |
| `CUDA_TOOLKIT_ROOT_DIR` | /path/... | CUDA Toolkit root                               |

### Media/Video Options

| Option                 | Value | Description                                      |
|-----------------------|------:|--------------------------------------------------|
| `WITH_GSTREAMER`      | ON    | Enable GStreamer framework                       |
| `WITH_V4L`            | ON    | Enable Video4Linux camera support                |
| `BUILD_opencv_cudacodec` | OFF | Disable CUDA video codec module (optional)      |

### Optional Features

| Option                      | Value | Description                                          |
|----------------------------|------:|------------------------------------------------------|
| `OPENCV_ENABLE_NONFREE`    | ON    | Enables patented algorithms (e.g., SIFT, SURF)       |
| `OPENCV_GENERATE_PKGCONFIG`| ON    | Generates `pkg-config` file                          |
| `OPENCV_PC_FILE_NAME`      | opencv.pc | Sets pkg-config filename                          |
| `OPENCV_EXTRA_MODULES_PATH`| ../../... | Path to `opencv_contrib` modules                  |
| `BUILD_EXAMPLES`           | ON    | Build example applications                           |

---

## ✅ Verification

### Method 1: Detailed Build Information

Open a Python interpreter (`conda activate sentinel` first) and run:

```python
import cv2
print(cv2.getBuildInformation())
```

**Look for lines like:**

- ✅ `GStreamer: YES (1.20.3)`  
- ✅ `NVIDIA CUDA: YES (ver 12.8, CUFFT CUBLAS FAST_MATH)`  
- ✅ `cuDNN: YES (ver 9.12.0)`  
- ✅ `Python 3: ...` (correct interpreter & path)  
- ✅ CUDA modules (e.g., `cudaimgproc`, `cudafeatures2d`)

---

### Method 2: Quick Functional Test

```python
import cv2

# OpenCV version
print("OpenCV version:", cv2.__version__)

# CUDA device count (>0 if GPU is available)
cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
print("CUDA-enabled devices:", cuda_devices)

# GStreamer presence (indirect check)
print("GStreamer support available:", cv2.getBuildInformation().find('GStreamer') != -1)
```

**Expected output (example):**

```
OpenCV version: 4.13.0-dev
CUDA-enabled devices: 1
GStreamer support available: True
```

---

### Method 3: Test GStreamer Pipeline

```python
import cv2

# Simple GStreamer test source → OpenCV
pipeline = "videotestsrc ! video/x-raw,format=BGR ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if cap.isOpened():
    print("✅ GStreamer pipeline opened successfully")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured: {frame.shape}")
    else:
        print("❌ Failed to capture frame from GStreamer pipeline.")
    cap.release()
else:
    print("❌ Failed to open GStreamer pipeline. Ensure GStreamer is correctly installed and configured.")
```

---

## 📟 GPU Architecture Reference

Set `CUDA_ARCH_BIN` based on your GPU’s compute capability.

| GPU Series                         | Compute Capability | `CUDA_ARCH_BIN` |
|-----------------------------------|--------------------|-----------------|
| RTX 40 Series (4090/4080/4070)    | 8.9                | `"8.9"`         |
| RTX 30 Series (3090/3080/3070)    | 8.6                | `"8.6"`         |
| RTX 20 Series (2080/2070/2060)    | 7.5                | `"7.5"`         |
| GTX 16 Series (1660/1650)         | 7.5                | `"7.5"`         |
| GTX 10 Series (1080/1070/1060)    | 6.1                | `"6.1"`         |

To identify your GPU:

```bash
nvidia-smi
# or
lspci | grep -i nvidia
```

Then look up the compute capability on NVIDIA’s official tables.

---

## ⚠️ Troubleshooting

### Common Issues and Solutions

**CMake: Could not find CUDA**

```bash
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev
```
Ensure CUDA is installed and paths are set properly.

**CMake: Could not find cuDNN**

```bash
sudo apt install libcudnn8-dev
```
Verify cuDNN install and headers/libs are discoverable. You may specify `CUDNN_INCLUDE_DIR` and `CUDNN_LIBRARY` manually in CMake if needed.

**Python bindings not found (`ModuleNotFoundError: No module named 'cv2'`)**

- Ensure `conda activate sentinel` before running **Step 6** (CMake).  
- Verify all `PYTHON3_*` paths and `CMAKE_INSTALL_PREFIX` point to your conda env.

**CUDA devices not detected (`cv2.cuda.getCudaEnabledDeviceCount() == 0`)**

- Check NVIDIA driver with `nvidia-smi`  
- Confirm CUDA with `nvcc --version`  
- Verify `CUDA_ARCH_BIN` matches your GPU (e.g., `nvidia-smi --query-gpu=compute_cap --format=csv`)

**GStreamer not working (pipelines fail to open)**

Install additional plugins:

```bash
sudo apt install -y     gstreamer1.0-plugins-base     gstreamer1.0-plugins-good     gstreamer1.0-plugins-bad     gstreamer1.0-plugins-ugly     gstreamer1.0-libav
```

**Compilation runs out of memory**

- Use fewer parallel jobs:

```bash
make -j2
```

- Increase swap temporarily:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# To make permanent, add the following line to /etc/fstab:
# /swapfile swap swap defaults 0 0
```

---

### Clean Build (If Needed)

```bash
# Go to your build directory
cd ~/opencv_build/opencv-4.x/build

# Remove previous build artifacts
rm -rf *

# Reconfigure and rebuild
cmake [your_options] ..
make -j$(nproc)
sudo make install
```

---

## 📚 Additional Resources

### Documentation
- [OpenCV Official Documentation](https://docs.opencv.org/4.x/)
- [OpenCV CUDA Documentation](https://docs.opencv.org/4.x/d6/d15/tutorial_cuda_intro.html)
- [GStreamer Documentation](https://gstreamer.freedesktop.org/documentation/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)

### Official Repositories
- [OpenCV (GitHub)](https://github.com/opencv/opencv)
- [opencv_contrib (GitHub)](https://github.com/opencv/opencv_contrib)
- [GStreamer (GitLab)](https://gitlab.freedesktop.org/gstreamer/gstreamer)

---

## ❓ Support

- Check OpenCV GitHub Issues
- Verify all prerequisites and dependencies
- Try a clean build and/or enable debug info for more detailed logs

---

## How to Use This Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

Then follow the installation steps in this `README.md`. Adjust paths and `CUDA_ARCH_BIN` to match your hardware. Contributions welcome—feel free to open Pull Requests or Issues!

---
