"""
Glucose error metrics, including descriptive Clarke zones.

Identical metric definitions alone do not make different cohorts or target
construction protocols directly comparable. Revision-v2 comparisons use
matched evaluation manifests and report all five Clarke zones.
"""

import numpy as np


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mard(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / y_true) * 100.0)


def clarke_error_grid_zones(y_true, y_pred):
    """
    Classifies each (reference, prediction) pair into Clarke Error Grid zones A-E.
    Reference: Clarke et al., Diabetes Care 1987;10(5):622-628.
    Implementation follows the standard piecewise boundary definitions used for
    self-monitoring blood glucose accuracy evaluation (mg/dL units).
    Returns a dict of zone -> count and zone -> percentage.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    zones = np.full(n, "", dtype=object)

    for i in range(n):
        ref, pred = y_true[i], y_pred[i]

        if (ref <= 70 and pred <= 70) or (pred <= 1.2 * ref and pred >= 0.8 * ref):
            zone = "A"
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zone = "E"
        elif ((ref >= 70 and ref <= 290) and pred >= ref + 110) or \
             (130 <= ref <= 180 and pred <= (7.0 / 5.0) * ref - 182):
            zone = "C"
        elif (ref >= 240 and (pred >= 70 and pred <= 180)) or \
             (ref <= 175 / 3 and pred <= 180 and pred >= 70) or \
             ((ref >= 175 / 3 and ref <= 70) and pred >= 6.0 / 5.0 * ref):
            zone = "D"
        else:
            zone = "B"
        zones[i] = zone

    counts = {z: int(np.sum(zones == z)) for z in ["A", "B", "C", "D", "E"]}
    pct = {z: (counts[z] / n * 100.0 if n else 0.0) for z in counts}
    return {"counts": counts, "percent": pct, "n": n}


def summarize(y_true, y_pred):
    ceg = clarke_error_grid_zones(y_true, y_pred)
    return {
        "n": len(y_true),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MARD_pct": mard(y_true, y_pred),
        "Clarke_A_pct": ceg["percent"]["A"],
        "Clarke_B_pct": ceg["percent"]["B"],
        "Clarke_C_pct": ceg["percent"]["C"],
        "Clarke_D_pct": ceg["percent"]["D"],
        "Clarke_E_pct": ceg["percent"]["E"],
    }
