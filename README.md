
# 6D Pose and Landmark Detection

A robust computer vision pipeline that integrates RealSense camera data, YOLO-based object detection, square detection, and 6D pose estimation to detect objects, estimate their orientation, and compute landmarks.

## Overview

This project demonstrates a complete pipeline for detecting objects with YOLO, extracting square regions from the detections, estimating their 6D pose (position and orientation), and overlaying the pose and other landmarks on the image. The implementation uses Intel RealSense for capturing depth and color frames, and leverages multiple modules to ensure a clean, modular design.

## Features

- **RealSense Integration:** Captures color and depth frames from an Intel RealSense camera.
- **Object Detection:** Uses a YOLO detector to identify regions of interest.
- **Square Detection:** Extracts square-like contours from the detected regions.
- **6D Pose Estimation:** Estimates the object’s pose using camera intrinsics and depth information.
- **Rotation Marker Estimation:** Adjusts the ordering of square corners based on a rotation marker.
- **Modular Design:** Separate modules for detection, pose estimation, rotation, and square-related computations.

## Requirements

- Python 3.7+
- [pyrealsense2](https://pypi.org/project/pyrealsense2/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- Additional dependencies as required by your YOLO detector and custom modules

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/6D-pose-landmark.git
   cd 6D-pose-landmark
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install the Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** If you don't have a `requirements.txt`, create one with the needed packages.

## Project Structure

```
.
├── /path/to/reference_image.jpg  # Reference image for rotation marker estimation
├── detector.py                          # YOLO detector implementation
├── main.py                              # Main script for pipeline setup and processing
├── models
│   └── alpha-1.pt                       # YOLO model file
├── pose.py                              # Pose estimation module
├── rotation.py                          # Rotation marker estimation module
├── square.py                            # Square detection and helper functions module
└── __pycache__                          # Cached compiled Python files
```

- **main.py:**  
  Sets up the RealSense pipeline, handles frame processing, invokes YOLO detection, and displays results.

- **detector.py:**  
  Contains the implementation for the YOLO-based object detector.

- **pose.py:**  
  Implements the 6D pose estimation using camera intrinsics and depth data.

- **rotation.py:**  
  Handles the rotation marker estimation to adjust object orientation.

- **square.py:**  
  Includes functions to detect squares, order corners, and clamp point coordinates.

## Usage

1. **Connect your Intel RealSense Camera** and ensure it is properly installed.

2. **Run the main script:**

   ```bash
   python main.py
   ```

   The script will open a window displaying the processed video feed. Press `q` to exit.

## Contributing

Contributions are welcome! If you find a bug or have an enhancement in mind, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please reach out at [alanantony6174@gmail.com](mailto:alanantony6174@gmail.com).
EOF
```
