# Smart Traffic Congestion Detection (Arduino-friendly ML)

Lightweight Decision Tree classifier for traffic congestion (**LOW / MEDIUM / HIGH**) using windowed features derived from:

- per-vehicle speed estimates (2× HC-SR04 time-of-flight, fixed sensor distance)
- vehicle detection timestamps
- vehicle counts in a fixed window (e.g., 10–15 seconds)

## Project layout

- `train.py`: end-to-end pipeline (generate/load data → train → evaluate → export rules + Arduino header)
- `traffic_congestion/`: modular library code
- Outputs (by default): `outputs/`
  - `evaluation_plots.png`
  - `decision_tree.png`
  - `tree_rules.txt`
  - `congestion_classifier.h`

## Quickstart

Create a venv, then:

```bash
pip install -r requirements.txt
python train.py --use-synthetic --feature-set arduino
```

To train on your real labeled CSV:

```bash
python train.py --csv traffic_log.csv --feature-set arduino
```

Expected CSV columns:

- `avg_speed` (cm/s)
- `vehicle_count` (int, per window)
- `speed_variance`
- `inter_arrival_avg` (seconds)
- `flow_rate` (vehicle_count / window_seconds)
- `label` in `{LOW, MEDIUM, HIGH}`

## Real-time testing with Arduino Serial (VERY IMPORTANT)

1. Train once to create `outputs/model.joblib` and `outputs/model_metadata.json`:

```bash
python train.py --use-synthetic --feature-set arduino
```

2. Upload an Arduino sketch that prints lines like:

- **Recommended**: `speed` (one vehicle per line)
  - Example: `123.4`
  - Python will compute `vehicle_count` + `flow_rate` using a 15s window
- Also supported (older format): `speed,vehicle_count`
  - Example: `123.4,6`

3. Run the realtime predictor (Windows example):

```bash
python realtime_serial_predict.py --port COM3 --baud 9600 --mode speed_only
```

It will print predicted congestion live for each incoming line.

## Arduino integration

After training, include the generated header:

```cpp
#include "congestion_classifier.h"
// ...
// If you trained with --feature-set arduino:
String level = classifyCongestion(avg_speed, vehicle_count, flow_rate);
```

Regenerate the header any time you retrain:

```bash
python train.py --use-synthetic --feature-set arduino
```

