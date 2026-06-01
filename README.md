# Parkinson's Tremor Detector

A wearable biomedical signal-processing prototype using an ESP32 and MPU6050 accelerometer/gyroscope to detect tremor-like wrist motion in real time.

The system collects wrist motion data, analyzes acceleration and gyroscope signals, and classifies movement as:

* Resting / no major movement
* Tremor-like rhythmic movement
* Normal large movement

## Project Overview

This project was built as a biomedical engineering prototype for tremor monitoring. The ESP32 reads motion data from the MPU6050 sensor and sends the data to a Python program for real-time signal processing and classification.

A later version also includes an ESP32 Wi-Fi dashboard so the results can be viewed from a phone browser.

## Hardware Used

* ESP32 development board
* MPU6050 accelerometer/gyroscope
* Female-to-female jumper wires
* Wrist strap or Velcro mount
* USB power or 3.7V LiPo battery support

## Wiring

| MPU6050 Pin | ESP32 Pin |
| ----------- | --------- |
| VCC         | 3.3V      |
| GND         | GND       |
| SDA         | GPIO 21   |
| SCL         | GPIO 22   |

## How It Works

The MPU6050 measures acceleration and angular velocity across the x, y, and z axes.

The Python program calculates total acceleration magnitude using the 3D vector magnitude formula:

```python
motion = sqrt(ax**2 + ay**2 + az**2)
```

This formula comes from the 3D Pythagorean theorem because the sensor measures motion across three perpendicular axes.

The detector then analyzes:

* RMS motion intensity
* Gyroscope RMS intensity
* Peak motion frequency
* Tremor-frequency bands
* Harmonic frequency bands

Based on these features, the program classifies the motion as resting, tremor-like, or normal movement.

## Dataset Results

The prototype was tested using three recorded motion datasets:

| Dataset                | Result                        |
| ---------------------- | ----------------------------- |
| `rest_normal.csv`      | Resting / no major movement   |
| `simulated_tremor.csv` | Tremor-like rhythmic movement |
| `normal_movement.csv`  | Normal large movement         |

## Main Files

| File                  | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `tremor_graph.py`     | Live graphing and CSV data recording                   |
| `compare_datasets.py` | Compares motion features across datasets               |
| `better_detector.py`  | Improved detector using RMS and frequency features     |
| `live_detector.py`    | Real-time tremor classification from ESP32 serial data |
| `serial_check.py`     | Checks raw ESP32 serial output                         |

## Current Status

Working prototype completed.

The system can classify:

* A still hand as resting
* A simulated tremor as tremor-like rhythmic movement
* Normal wrist motion as large movement

## Future Improvements

* Improve physical wearable housing
* Add safer battery power management
* Add Bluetooth or Wi-Fi data transmission
* Improve frequency analysis on the ESP32
* Collect more labeled datasets
* Train a machine learning model for tremor classification

## Disclaimer

This project is an educational biomedical engineering prototype. It is not a medical diagnostic device.
