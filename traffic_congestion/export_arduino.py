from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.tree import _tree, export_text


def export_tree_rules_text(clf, feature_names: list[str], output_path: str | Path) -> str:
    """
    Export sklearn's human-readable rule text and save it.
    """
    rules_text = export_text(clf, feature_names=feature_names)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rules_text, encoding="utf-8")
    return rules_text


def export_tree_to_arduino_cpp(
    clf,
    feature_names: list[str],
    output_path: str | Path,
    int_features: set[str] | None = None,
) -> str:
    """
    Walk the fitted sklearn DecisionTree and emit an Arduino-friendly C++ header.
    The generated function mirrors the tree exactly with nested if/else.
    """
    tree_ = clf.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    classes = clf.classes_

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    if int_features is None:
        int_features = {"vehicle_count"}

    lines += [
        "// ============================================================",
        "// congestion_classifier.h",
        "// AUTO-GENERATED — do not edit manually.",
        "// Regenerate by rerunning train.py",
        "//",
        "// Model    : DecisionTreeClassifier(max_depth=4, gini)",
        "// Classes  : LOW | MEDIUM | HIGH",
        "//",
        "// Feature order passed to classifyCongestion():",
    ]
    for i, n in enumerate(feature_names):
        lines.append(f"//   [{i}] {n}")
    lines += [
        "// ============================================================",
        "",
        "#ifndef CONGESTION_CLASSIFIER_H",
        "#define CONGESTION_CLASSIFIER_H",
        "",
        "#include <Arduino.h>",
        "",
        "// Returns: \"LOW\", \"MEDIUM\", or \"HIGH\"",
        "String classifyCongestion(",
    ]
    for idx, fname in enumerate(feature_names):
        ctype = "int" if fname in int_features else "float"
        comma = "," if idx < len(feature_names) - 1 else ""
        lines.append(f"    {ctype} {fname}{comma}")
    lines += [
        ") {",
    ]

    def recurse(node: int, depth: int) -> None:
        indent = "  " * (depth + 1)
        if feature[node] != _tree.TREE_UNDEFINED:
            fname = feature_names[feature[node]]
            tval = float(threshold[node])
            if fname in int_features:
                # sklearn commonly uses "k + 0.5" thresholds for integer features.
                # For exact equivalence, compare to floor(threshold), e.g. <= 3.5  ==>  <= 3
                t_int = int(np.floor(tval))
                lines.append(f"{indent}if ({fname} <= {t_int}) {{")
            else:
                # Keep threshold readable while avoiding over-rounding
                tval_s = f"{tval:.4f}".rstrip("0").rstrip(".")
                lines.append(f"{indent}if ({fname} <= {tval_s}f) {{")
            recurse(int(tree_.children_left[node]), depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(int(tree_.children_right[node]), depth + 1)
            lines.append(f"{indent}}}")
        else:
            cls = classes[int(np.argmax(tree_.value[node]))]
            counts = tree_.value[node][0].astype(int)
            count_str = ", ".join(f"{classes[i]}:{counts[i]}" for i in range(len(classes)))
            lines.append(f'{indent}return "{cls}";  // samples [{count_str}]')

    recurse(0, 0)

    lines += [
        "  return \"LOW\";  // fallback — should never be reached",
        "}",
        "",
        "#endif  // CONGESTION_CLASSIFIER_H",
    ]

    cpp_code = "\n".join(lines)
    output_path.write_text(cpp_code, encoding="utf-8")
    return cpp_code

