# Smart Traffic ML System

## Overview

This project is an end-to-end machine learning + IoT system that predicts traffic congestion in real time and integrates with an Arduino-based setup for intelligent traffic control.

It combines data processing, model training, and live prediction with hardware interaction to simulate a smart traffic management system.

---

## Features

* Traffic congestion prediction using ML
* Real-time data processing via serial communication
* Arduino integration for live control
* Modular ML pipeline (training, evaluation, inference)
* Visualization and analysis tools

---

## Project Structure

```
smart-traffic-ml/
│
├── traffic_congestion/
│   ├── config.py
│   ├── data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── realtime_window.py
│   └── export_arduino.py
│
├── arduino/
│   └── sketch_feb23a/
│
├── train.py
├── realtime_serial_predict.py
├── requirements.txt
└── README.md
```

---

## How It Works

```
Sensor Data → Arduino → Serial Communication → Python
→ Preprocessing → ML Model → Prediction → Arduino Action
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/riyajenson/smart-traffic-ml.git
cd smart-traffic-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python train.py
```

### Run real-time prediction

```bash
python realtime_serial_predict.py
```

Make sure your Arduino is connected and the correct serial port is configured.

---

## Arduino Integration

* Arduino collects traffic-related inputs (e.g., sensors/timing)
* Sends data via serial communication to Python
* Python predicts congestion level
* Output can be used to control signals dynamically

---

## Outputs

* Predicted congestion levels
* Visualization graphs
* Real-time decision signals

---

## Tech Stack

* Python (NumPy, Pandas, scikit-learn)
* Arduino (C/C++)
* Serial Communication
* Data Visualization

---

## Future Improvements

* Add live dashboard (Streamlit / web app)
* Integrate computer vision (vehicle detection)
* Deploy model for edge inference
* Improve dataset and model accuracy

---

## Acknowledgment

Built as a practical implementation of combining Machine Learning + Embedded Systems for real-world problem solving.
