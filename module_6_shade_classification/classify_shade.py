import os
import glob
import json
import math

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
# NCT and SmileImages have no VITA shade ground truth - only tooth masks, used
# to train segmentation (modules 2/3). The VITA guide itself is the only
# thing with both known colour AND a known label. So the shade decision here
# is nearest-colour matching against the guide's reference Lab values, not a
# trained classifier - no model, no GPU, no training data requirement at all.
# See module_5_integration/notes.md for where the input (tooth L/a/b) comes from.
#
# Reference values: mean CIELAB L*/a*/b* measured across 100 tabs per shade
# with a spectrophotometer (Consistency in color parameters of a commonly
# used shade guide, PMC3792636) - one internally-consistent source, rather
# than mixing numbers from studies that used different instruments/lighting
# (published VITA Lab values vary by a few ΔE across papers; this matters
# for borderline cases, see the confidence tiers below).
VITA_SHADE_LAB = {
    "A1":  (61.18411, -0.25212, 6.5407),
    "A2":  (61.28707,  0.92826, 9.27927),
    "A3":  (57.97017,  1.05499, 10.03341),
    "A3.5": (56.00544, 1.97744, 12.79715),
    "A4":  (54.33394,  2.05272, 11.90831),
    "B1":  (59.70447, -0.63835, 5.82622),
    "B2":  (60.74294, -0.14412, 9.17686),
    "B3":  (56.1455,   1.26299, 12.44569),
    "B4":  (56.08678,  1.1876,  12.78952),
    "C1":  (57.97205,  0.2474,  6.82626),
    "C2":  (55.65995,  0.50887, 9.16877),
    "C3":  (54.26528,  0.96021, 9.91913),
    "C4":  (50.21337,  1.93824, 11.25651),
    "D2":  (56.5636,   0.20202, 5.38453),
    "D3":  (55.18432,  1.12593, 8.02208),
    "D4":  (55.58672,  0.26679, 10.66036),
}

# Thresholds from the project's own base paper's literature review (Table 4,
# docs/"Full Length Paper.pdf"): "ΔE < 2.69 is acceptable, ΔE < 1.6 perfect
# color match" (citing Cielab/Vita 3D Master digital shade-matching studies).
# Reused here directly rather than inventing new thresholds.
DELTA_E_PERFECT = 1.6
DELTA_E_ACCEPTABLE = 2.69

# module_5's per-patient output directory (module_5_integration/pipeline.py's
# own OUTPUT_DIR) - this script reads its meta.json files, nothing from
# modules 2/3/4 directly.
PATIENT_OUTPUT_DIR = "/kaggle/working/pipeline_outputs"
RESULTS_PATH = "/kaggle/working/shade_predictions.json"


# --------------------------------------------------------------------------
# NEAREST-SHADE MATCHING
# --------------------------------------------------------------------------
def delta_e_cie76(lab1, lab2):
    """Euclidean distance in Lab space - the simplest standard ΔE formula,
    matches what the cited literature's thresholds above were computed with."""
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(lab1, lab2)))


def classify_shade(tooth_lab):
    distances = {
        shade: delta_e_cie76(tooth_lab, ref_lab)
        for shade, ref_lab in VITA_SHADE_LAB.items()
    }
    best_shade = min(distances, key=distances.get)
    best_delta_e = distances[best_shade]

    if best_delta_e < DELTA_E_PERFECT:
        confidence = "perfect"
    elif best_delta_e < DELTA_E_ACCEPTABLE:
        confidence = "acceptable"
    else:
        confidence = "low"  # tooth colour is far from every reference tab - likely a bad
                             # segmentation/colour-extraction upstream, not a real shade

    return {
        "predicted_shade": best_shade,
        "delta_e": round(best_delta_e, 3),
        "confidence": confidence,
        "all_distances": {k: round(v, 3) for k, v in distances.items()},
    }


# --------------------------------------------------------------------------
# MAIN - runs classification over every patient module_5 already processed
# --------------------------------------------------------------------------
def main():
    meta_paths = sorted(glob.glob(os.path.join(PATIENT_OUTPUT_DIR, "*", "meta.json")))
    if not meta_paths:
        print(f"No meta.json files found under {PATIENT_OUTPUT_DIR} - "
              f"run module_5_integration/pipeline.py first.")
        return

    all_results = {}
    for meta_path in meta_paths:
        patient_id = os.path.basename(os.path.dirname(meta_path))
        with open(meta_path) as f:
            meta = json.load(f)

        tooth_lab = meta["color_vector"][:3]  # tooth L, a, b - first 3 of the 9-number vector
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in tooth_lab):
            print(f"{patient_id}: skipped - no tooth colour extracted (check module_5's flags)")
            continue

        result = classify_shade(tuple(tooth_lab))
        all_results[patient_id] = result
        print(f"{patient_id}: {result['predicted_shade']}  "
              f"(ΔE={result['delta_e']}, {result['confidence']})")

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} predictions to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
