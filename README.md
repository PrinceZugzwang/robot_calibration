# Robot Calibration (MuJoCo)

This repo contains scripts for performing **eye-to-hand calibration** and **visualizing predicted vs. ground truth marker positions** using MuJoCo.

## Structure

- `eye_to_hand_calibration.py`  
  Captures images, detects checkerboard poses, and computes `T_cam_to_base` using AX=XB calibration.

- `live_viewer.py`  
  Visualizes the predicted vs. ground truth marker positions inside MuJoCo.

- `debug_view_fixed/`  
  Stores captured calibration images and pose `.npy` files.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
