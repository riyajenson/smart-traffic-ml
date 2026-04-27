from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import serial

from traffic_congestion.realtime_window import WindowState


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read Arduino Serial data and predict traffic congestion in real-time.")
    p.add_argument("--port", required=True, help='Serial port, e.g. "COM3"')
    p.add_argument("--baud", type=int, default=9600, help="Baud rate (match Arduino Serial.begin())")
    p.add_argument("--model", type=str, default="outputs/model.joblib", help="Path to trained model.joblib")
    p.add_argument("--meta", type=str, default="outputs/model_metadata.json", help="Path to model_metadata.json")
    p.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout seconds")
    p.add_argument(
        "--mode",
        choices=["speed_only", "speed_and_count"],
        default="speed_only",
        help='Serial line format: "speed" OR "speed,vehicle_count". Your Arduino code uses speed_only.',
    )
    p.add_argument(
        "--print-every",
        type=str,
        default="event",
        choices=["event", "window"],
        help='Print prediction on every new vehicle ("event") or once per full window ("window").',
    )
    p.add_argument(
        "--write-back",
        action="store_true",
        help='Write the prediction back to Arduino via Serial as a single line: "LOW" | "MEDIUM" | "HIGH".',
    )
    p.add_argument(
        "--write-prefix",
        type=str,
        default="",
        help='Optional prefix for write-back lines (e.g. "PRED:"). Sent as "{prefix}{label}\\n".',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    if not model_path.is_absolute():
        model_path = project_dir / model_path
    if not meta_path.is_absolute():
        meta_path = project_dir / meta_path

    clf = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names: list[str] = meta["feature_names"]
    window_seconds: float = float(meta.get("window_seconds", 15.0))

    print(f"Loaded model: {model_path}")
    print(f"Features   : {feature_names}")
    print(f"Listening  : {args.port} @ {args.baud}")
    if args.mode == "speed_only":
        print('Expected   : "speed" (one vehicle speed per line, cm/s)')
    else:
        print('Expected   : "speed,vehicle_count" (e.g., "123.4,6")')
    print("Press Ctrl+C to stop.\n")

    win = WindowState(window_seconds=window_seconds)
    last_window_print_t = 0.0

    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        # Give Arduino time to reset on Serial connect
        time.sleep(2.0)

        while True:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue

            # ---- Parse Arduino line
            speed = None
            vehicle_count = None

            if args.mode == "speed_only":
                try:
                    speed = float(raw)
                except ValueError:
                    print(f"Skipping non-numeric line: {raw!r}")
                    continue
                win.add_speed(speed_cm_s=speed)
                feat_map = win.features()
            else:
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) < 2:
                    print(f"Skipping malformed line: {raw!r}")
                    continue
                try:
                    speed = float(parts[0])  # cm/s
                    vehicle_count = int(float(parts[1]))
                except ValueError:
                    print(f"Skipping non-numeric line: {raw!r}")
                    continue
                feat_map = {"avg_speed": speed, "vehicle_count": float(vehicle_count), "flow_rate": vehicle_count / window_seconds}

            try:
                x = np.array([feat_map[name] for name in feature_names], dtype=float).reshape(1, -1)
            except KeyError as e:
                raise RuntimeError(
                    f"Model expects feature {e!s} but realtime script doesn't know how to compute it. "
                    f"Retrain with --feature-set arduino or extend realtime feature computation."
                )

            pred = clf.predict(x)[0]
            now = time.time()
            should_emit = False
            if args.print_every == "event":
                should_emit = True
                vc = int(feat_map.get("vehicle_count", 0.0))
                print(f"speed={speed:7.2f} cm/s  window_count={vc:3d}  avg={feat_map['avg_speed']:7.2f}  =>  congestion={pred}")
            else:
                if (now - last_window_print_t) >= window_seconds:
                    should_emit = True
                    last_window_print_t = now
                    vc = int(feat_map.get("vehicle_count", 0.0))
                    print(f"[window] count={vc:3d}  avg={feat_map['avg_speed']:7.2f}  flow={feat_map['flow_rate']:.3f}  =>  congestion={pred}")

            if args.write_back and should_emit:
                out_line = f"{args.write_prefix}{pred}\n"
                ser.write(out_line.encode("utf-8", errors="ignore"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

