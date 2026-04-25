FEATURE_SETS: dict[str, list[str]] = {
    # Arduino/Serial-friendly: can be produced from "speed,vehicle_count"
    # where speed is treated as the window's avg_speed.
    "arduino": [
        "avg_speed",       # from Serial speed (cm/s)
        "vehicle_count",   # from Serial count per window
        "flow_rate",       # derived: vehicle_count / window_seconds
    ],
    # Full set (requires you to compute variance/inter-arrival on-device or in Python from timestamps)
    "full": [
        "avg_speed",
        "vehicle_count",
        "speed_variance",
        "inter_arrival_avg",
        "flow_rate",
    ],
}

DEFAULT_FEATURE_SET = "arduino"

CLASS_ORDER = ["LOW", "MEDIUM", "HIGH"]

