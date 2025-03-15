# InTheBlinkOfAnEye - Automatic Selfie Taker

A Flask web application that uses OpenCV, dlib, and facial landmark detection to capture a photo when a blink is detected. The app displays a live video feed alongside the captured selfie, all within a modern, professional UI.

## Features

- **Live Video Feed:** View your webcam feed in real time.
- **Blink Detection:** Uses dlib’s 68-point facial landmark predictor to detect blinks.
- **Automatic Capture:** Captures a selfie at the moment of a blink.

## Prerequisites

- **Git:** Make sure Git is installed. [Download Git](https://git-scm.com/downloads)
- **Python 3.12:4** Download and install from [python.org](https://www.python.org/downloads/)
- **CMake:** Required for building `dlib`
  - **Windows:** [Download CMake](https://cmake.org/download/) and add it to your PATH.
  - **macOS:** Install via Homebrew:
    ```bash
    brew install cmake
    ```
  - **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update && sudo apt install cmake
    ```

## Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/marriojeyakumar/InTheBlinkOfAnEye
cd InTheBlinkOfAnEye
```

To install dependencies:

```bash
pip install -r requirements.txt
```

To run the server:

```bash
python app.py
```
