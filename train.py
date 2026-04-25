from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from traffic_congestion import (
    CLASS_ORDER,
    DEFAULT_FEATURE_SET,
    FEATURE_SETS,
    build_model,
    build_xy,
    evaluate_model,
    export_tree_rules_text,
    export_tree_to_arduino_cpp,
    generate_synthetic_data,
    load_labeled_csv,
    preprocess_dataframe,
    save_evaluation_plots,
    save_tree_plot,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/export Arduino-friendly traffic congestion decision tree.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--use-synthetic", action="store_true", help="Train using generated synthetic dataset.")
    src.add_argument("--csv", type=str, help="Path to labeled CSV (avg_speed, vehicle_count, speed_variance, inter_arrival_avg, flow_rate, label).")

    p.add_argument("--window-seconds", type=float, default=15.0, help="Window duration used to compute flow_rate for synthetic data.")
    p.add_argument("--test-size", type=float, default=0.20, help="Test split fraction.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--feature-set",
        type=str,
        default=DEFAULT_FEATURE_SET,
        choices=sorted(FEATURE_SETS.keys()),
        help="Which input features to train/export (match what Arduino can send).",
    )

    p.add_argument("--outputs", type=str, default="outputs", help="Output directory for plots/rules/header.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    out_dir = Path(args.outputs)
    if not out_dir.is_absolute():
        out_dir = project_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data
    if args.use_synthetic:
        df = generate_synthetic_data(n_samples=900, seed=args.seed, window_seconds=args.window_seconds)
        print("Using SYNTHETIC data.")
    else:
        res = load_labeled_csv(args.csv)
        df = res.df
        print(f"Loaded {len(df)} labeled samples from {args.csv} (dropped {res.dropped_rows}).")

    # ---- Preprocess / build X,y
    feature_names = FEATURE_SETS[args.feature_set]
    df = preprocess_dataframe(df, feature_set=args.feature_set)

    # If we’re in the Arduino feature set, ensure flow_rate exists even for real CSVs.
    if "flow_rate" in feature_names and "flow_rate" not in df.columns:
        df["flow_rate"] = df["vehicle_count"] / float(args.window_seconds)

    X, y = build_xy(df, feature_set=args.feature_set)

    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    for cls in CLASS_ORDER:
        print(f"  {cls:<6} {int((y == cls).sum())}")

    # ---- Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # ---- Train
    clf = build_model(random_state=args.seed)
    clf.fit(X_train, y_train)

    print("\nDecision Tree trained.")
    print(f"  Total nodes  : {clf.tree_.node_count}")
    print(f"  Actual depth : {clf.get_depth()}")
    print(f"  Leaf nodes   : {clf.get_n_leaves()}")

    # ---- Evaluate
    eval_res = evaluate_model(clf, X_train, y_train, X_test, y_test, X_all=X, y_all=y, cv_folds=5)
    y_pred = clf.predict(X_test)
    importances = clf.feature_importances_

    print("\n" + "=" * 60)
    print(f"TEST ACCURACY: {eval_res.accuracy * 100:.2f}%")
    print("=" * 60)
    print("\nCLASSIFICATION REPORT")
    print(eval_res.classification_report)
    print(f"5-Fold CV Accuracy: {eval_res.cv_scores.mean() * 100:.2f}% ± {eval_res.cv_scores.std() * 100:.2f}%")
    print(f"CV Scores: {[round(float(s) * 100, 1) for s in eval_res.cv_scores]}")

    print("\nFEATURE IMPORTANCES (ranked)")
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx):
        print(f"  {rank+1}. {feature_names[idx]:<22} {importances[idx]:.4f}")

    # ---- Visualize
    save_evaluation_plots(df, y_test, y_pred, feature_names, importances, out_dir / "evaluation_plots.png")
    save_tree_plot(clf, feature_names, out_dir / "decision_tree.png")

    # ---- Export rules + Arduino header
    export_tree_rules_text(clf, feature_names, out_dir / "tree_rules.txt")
    export_tree_to_arduino_cpp(clf, feature_names, out_dir / "congestion_classifier.h")

    # ---- Save model + metadata (used by realtime_serial_predict.py)
    joblib.dump(clf, out_dir / "model.joblib")
    (out_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "feature_set": args.feature_set,
                "feature_names": feature_names,
                "class_order": CLASS_ORDER,
                "window_seconds": float(args.window_seconds),
                "serial_formats_supported": ["speed", "speed,vehicle_count"],
                "recommended_serial_format": "speed",
                "notes": "Recommended: stream one vehicle speed per line from Arduino; Python windows it to compute avg_speed, vehicle_count, flow_rate.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved outputs:")
    print(f"  {out_dir / 'congestion_classifier.h'}")
    print(f"  {out_dir / 'tree_rules.txt'}")
    print(f"  {out_dir / 'evaluation_plots.png'}")
    print(f"  {out_dir / 'decision_tree.png'}")
    print(f"  {out_dir / 'model.joblib'}")
    print(f"  {out_dir / 'model_metadata.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

