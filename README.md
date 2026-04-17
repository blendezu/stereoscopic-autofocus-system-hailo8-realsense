# Stereoscopic Autofocus System for Cinema lenses (Intel RealSense D455 + Hailo-8 AI accelerator + Raspberry Pi 5)

## About the project
This project was developed as part of the Camera Technology module in the Media Technology program at the University of Applied Sciences Cologne during the Summer Semester 2025.

**Supervision:**
- Prof. Gregor Fischer - IMP F07
- Prof. Dirk Poggemann - IMP F07
- Christian Loebich - IMP F07

**Project Manager:** Anh Duong Tran

**Contributions:**
- Project Idea, Concept & Equipment Selection: Anh Duong Tran
- Control Hardware Setup: Angelika Allgäuer, Mark Ulanowski
- 3D Print: Timo Schulz
- Motor Control System: Angelika Allgäuer, Doron Kohler, Mark Ulanowski
- Stereo Camera System & GUI Development: Hannah Strippel, Anh Duong Tran
- Object Detection & Tracking: Anh Duong Tran

## Achievements

<table align="center" width="100%">
  <tr>
    <td align="center" width="45%">
      <img src="images/postersession.png" alt="Poster Session" width="100%">
    </td>
    <td align="left" width="55%">
      We won second place at the Poster Session of the Media Technology program and received a grade of 1.0 (German) for the Camera Technology module.
    </td>
  </tr>
</table>

## Overview
<div align="center" style="margin-bottom:20px;">
  <img src="images/finalResult.png" alt="Finalset" style="max-width:80%; height:auto; display:block; margin:0 auto;">
</div>

This project implements a real-time stereoscopic autofocus system, which provides an interactive GUI to perform person detection, tracking, depth-based distance estimation, and stepper-motor driven lens focusing in real time. It uses:
- Intel RealSense D455 for RGB + depth frames
- YOLO models for detection person, face, and segmentation
- Hailo-8 AI accelerator for object detection
- SORT for multi-object tracking
- An Adafruit Motor HAT to drive a stepper on the lens focus ring
- Raspberry Pi 5
- The whole system powered by a Vmount baterry

<!-- replace the previous flex container: show images first and captions centered below each image -->
<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="images/controlHardware.png" alt="Hardware" width="100%">
      <br><sub><b>Control Hardware</b></sub>
    </td>
    <td align="center" width="67%">
      <img src="images/pipeline.png" alt="Pipeline" width="100%">
      <br><sub><b>System pipeline</b></sub>
    </td>
  </tr>
</table>

## Key Features

- Launch flow: Loading screen → Calibration screen → Main screen
- Calibration checklist with lighting condition selection
- Live video view with adjustable ROI (drag corners)
- Person detection + SORT tracking; tap a tracked person to focus on them
- Face detection for precise distance; person mask sampling fallback if no face is visible
- Optical-flow point in ROI for tap-to-focus when no tracked person is selected
- Depth correction via lighting-dependent LUTs (inside/outside, good/bad light)
- Real-time focus distance computation and conversion to stepper steps
- Focus timing slider (smooth vs. fast moves) and hysteresis to reduce jitter
- Depth profile panel with:
  - Valid focus range guidance
  - Current target focus distance (white bar)
  - Motor focus plane indicator
  - Colored sample points (green = selected target, gray = others)
- FPS display, reset tracking button, and safe cleanup on exit

## Demo Video

Tracking-by-detection with YOLO und SORT:

<div align="center">
  <a href="https://www.youtube.com/shorts/-5ozBTq2l24">
    <img src="images/trackingSortYT.png" alt="Object Tracking" width="30%">
  </a>
  <a href="https://www.youtube.com/shorts/xMcakpQYOgM">
    <img src="images/oFYT.png" alt="Optical Flow" width="30%">
  </a>
  <a href="https://www.youtube.com/shorts/RXhNjNWLM1I?feature=share">
    <img src="images/GUIShortYT.png" alt="GUI" width="30%">
  </a>
</div>

## Distance Correction for Stereo Camera

We tested the Intel RealSense D455 stereo camera and discovered that the measured distances were consistently greater than the actual distances. To address this, we developed a correction function and lookup tables (LUTs) to calibrate the measured distances to true distances across different lighting conditions. We messured again an had the results on the right. 

You can find the measurements in this  [google sheet](https://docs.google.com/spreadsheets/d/1Mt2iLZNg792dCo4I_J54miuZmEHCaRKvOUn7tRtqKD0/edit?usp=sharing).

This is a quick comparasion between before and with our correction:

<table align="center" width="100%">
  <tr>
    <td align="center" width="50%">
      <img src="images/InsideGoodLighting.png" alt="Inside good lighting (raw)" width="85%">
    </td>
    <td align="center" width="50%">
      <img src="images/InsideGoodLightingCorrected.png" alt="Inside good lighting (corrected)" width="85%">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center"><sub>Inside in good lighting condition</sub></td>
  </tr>
</table>

<br>

<table align="center" width="100%">
  <tr>
    <td align="center" width="50%">
      <img src="images/InsideBadLighting.png" alt="Inside bad lighting (raw)" width="85%">
    </td>
    <td align="center" width="50%">
      <img src="images/InsideBadLightingCorrected.png" alt="Inside bad lighting (corrected)" width="85%">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center"><sub>Inside in bad lighting condition</sub></td>
  </tr>
</table>

<br>

<table align="center" width="100%">
  <tr>
    <td align="center" width="50%">
      <img src="images/OutsideGoodLighting.png" alt="Outside good lighting (raw)" width="85%">
    </td>
    <td align="center" width="50%">
      <img src="images/OutsideGoodLightingCorrected.png" alt="Outside good lighting (corrected)" width="85%">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center"><sub>Outside in good lighting condition</sub></td>
  </tr>
</table>



## Computer Vision Pipeline
- Person detection (YOLO model via DeGirum/Hailo)
- SORT tracker for consistent IDs and track selection
- Face detection within the selected person’s crop
- Person segmentation mask for robust depth sampling (selected vs. non-selected)
- Optical flow for non-person objects

## Focusing Logic
- Depth readouts in meters from aligned depth frame
- Lighting-dependent correction:
  - Linear interpolation across a measured→true distance LUT
- Focus offset (camera-to-lens baseline) added to corrected distance
- Conversion to motor steps via a calibrated steps↔distance LUT
- Hysteresis threshold to trigger motor moves only on meaningful distance changes

## Hardware Control
- RealSense D455 streams:
  - Color: 1280×720 @ 30 FPS
  - Depth: 848×480 @ 30 FPS
  - Depth laser power set to 360 mW
- Motor HAT:
  - Background process consumes move commands (steps + focus_time)
  - Interleaved stepping; homing back to zero on shutdown

## Requirements
- Python 3.10+
- Kivy, NumPy, OpenCV
- degirum PySDK for Inference on Hailo device (or HailoRT runtime as required)
- Adafruit MotorKit and adafruit_motor
- [pyrealsense2](https://github.com/realsenseai/librealsense)
- [SORT](https://github.com/abewley/sort)
