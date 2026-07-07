import logging
import numpy as np
import pandas as pd
import requests
from io import StringIO
from typing import Optional, Callable

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

from src import config

logger = logging.getLogger(__name__)




def fetch_known_params(planet_name: str) -> dict:

    formatted_name = planet_name
    if planet_name[-1].isalpha() and planet_name[-2] != ' ':
        formatted_name = planet_name[:-1] + ' ' + planet_name[-1]

    query = f"""
    SELECT pl_name, pl_orbper, pl_trandep, pl_trandur, pl_tranmid, pl_rade
    FROM ps
    WHERE pl_name = '{formatted_name}' AND default_flag = 1
    """

    try:
        logger.info(f"Querying NASA Exoplanet Archive for '{formatted_name}'...")
        response = requests.get(
            config.NASA_TAP_URL,
            params={"query": query, "format": "csv"},
            timeout=30
        )
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            logger.warning(f"No data found for '{formatted_name}' in NASA Archive. "
                          f"Falling back to local config.")
            return _fallback_to_config(planet_name)

        row = df.iloc[0]
        result = {
            "planet_name": row.get("pl_name", formatted_name),
            "period_days": float(row["pl_orbper"]) if pd.notna(row.get("pl_orbper")) else None,
            "depth_ppm": float(row["pl_trandep"]) if pd.notna(row.get("pl_trandep")) else None,
            "duration_hours": float(row["pl_trandur"]) if pd.notna(row.get("pl_trandur")) else None,
            "transit_midpoint": float(row["pl_tranmid"]) if pd.notna(row.get("pl_tranmid")) else None,
            "planet_radius_earth": float(row["pl_rade"]) if pd.notna(row.get("pl_rade")) else None,
        }

        logger.info(f"  → Period: {result['period_days']} days, "
                    f"Depth: {result['depth_ppm']} ppm, "
                    f"Duration: {result['duration_hours']} hours")
        return result

    except requests.RequestException as e:
        logger.error(f"NASA API request failed: {e}. Falling back to config.")
        return _fallback_to_config(planet_name)
    except Exception as e:
        logger.error(f"Error parsing NASA response: {e}. Falling back to config.")
        return _fallback_to_config(planet_name)


def _fallback_to_config(planet_name: str) -> dict:
    """Fall back to locally stored parameters from config.TARGETS."""
    for name, params in config.TARGETS.items():
        if name.lower() == planet_name.lower() or planet_name.lower() in name.lower():
            return {
                "planet_name": name,
                "period_days": params["known_period_days"],
                "depth_ppm": params["known_depth_ppm"],
                "duration_hours": params["known_duration_hours"],
                "transit_midpoint": None,
                "planet_radius_earth": None,
                "source": "local_config"
            }
    logger.warning(f"Target '{planet_name}' not found in config either.")
    return {"planet_name": planet_name, "period_days": None, "depth_ppm": None,
            "duration_hours": None, "transit_midpoint": None, "planet_radius_earth": None}


def validate_detection(detected: dict, known: dict,
                       tolerances: Optional[dict] = None) -> dict:
    
    if tolerances is None:
        tolerances = {
            "period": config.PERIOD_TOLERANCE,
            "depth": config.DEPTH_TOLERANCE,
            "duration": config.DURATION_TOLERANCE,
        }

    results = {"target": known.get("planet_name", "Unknown")}

    if detected.get("period") and known.get("period_days"):
        period_error = abs(detected["period"] - known["period_days"]) / known["period_days"]
        results["period"] = {
            "detected": detected["period"],
            "known": known["period_days"],
            "relative_error": period_error,
            "tolerance": tolerances["period"],
            "pass": period_error <= tolerances["period"]
        }
    else:
        results["period"] = {"detected": None, "known": None, "pass": False,
                             "relative_error": None, "tolerance": tolerances["period"]}

    detected_depth_ppm = detected.get("depth", 0)
    if detected_depth_ppm and detected_depth_ppm < 1:
        detected_depth_ppm *= 1e6

    if detected_depth_ppm and known.get("depth_ppm"):
        depth_error = abs(detected_depth_ppm - known["depth_ppm"]) / known["depth_ppm"]
        results["depth"] = {
            "detected_ppm": detected_depth_ppm,
            "known_ppm": known["depth_ppm"],
            "relative_error": depth_error,
            "tolerance": tolerances["depth"],
            "pass": depth_error <= tolerances["depth"]
        }
    else:
        results["depth"] = {"detected_ppm": None, "known_ppm": None, "pass": False,
                            "relative_error": None, "tolerance": tolerances["depth"]}
    detected_duration = detected.get("duration_hours", detected.get("duration", 0))
    if detected_duration and known.get("duration_hours"):
        dur_error = abs(detected_duration - known["duration_hours"]) / known["duration_hours"]
        results["duration"] = {
            "detected_hours": detected_duration,
            "known_hours": known["duration_hours"],
            "relative_error": dur_error,
            "tolerance": tolerances["duration"],
            "pass": dur_error <= tolerances["duration"]
        }
    else:
        results["duration"] = {"detected_hours": None, "known_hours": None, "pass": False,
                               "relative_error": None, "tolerance": tolerances["duration"]}
    results["overall_pass"] = results["period"].get("pass", False)

    return results

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)


    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc_roc"] = None
        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["average_precision"] = None
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=["No Transit", "Transit"], zero_division=0
    )

    return metrics




def run_benchmark(pipeline_func: Callable,
                  targets: Optional[dict] = None) -> pd.DataFrame:

    if targets is None:
        targets = config.TARGETS

    rows = []

    for target_name, target_info in targets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {target_name}")
        logger.info(f"{'='*60}")

        try:
            from src.data_pipeline import load_lightcurve
            time, flux = load_lightcurve(target_name)

            detected = pipeline_func(time, flux)
        
            known = fetch_known_params(target_name)

            validation = validate_detection(detected, known)

            row = {
                "target": target_name,
                "detected_period": detected.get("period"),
                "known_period": known.get("period_days"),
                "period_error": validation["period"].get("relative_error"),
                "period_pass": validation["period"].get("pass"),
                "detected_depth_ppm": validation["depth"].get("detected_ppm"),
                "known_depth_ppm": known.get("depth_ppm"),
                "depth_error": validation["depth"].get("relative_error"),
                "depth_pass": validation["depth"].get("pass"),
                "detected_duration": validation["duration"].get("detected_hours"),
                "known_duration": known.get("duration_hours"),
                "duration_error": validation["duration"].get("relative_error"),
                "duration_pass": validation["duration"].get("pass"),
                "confidence": detected.get("confidence"),
                "overall_pass": validation["overall_pass"],
            }
            rows.append(row)

            status = "✅ PASS" if validation["overall_pass"] else "❌ FAIL"
            logger.info(f"  Result: {status}")

        except FileNotFoundError:
            logger.warning(f"  Data file not found for {target_name}. "
                          f"Run data pipeline first.")
            rows.append({"target": target_name, "overall_pass": False,
                         "error": "Data not found"})
        except Exception as e:
            logger.error(f"  Benchmark failed for {target_name}: {e}")
            rows.append({"target": target_name, "overall_pass": False,
                         "error": str(e)})

    return pd.DataFrame(rows)


def generate_report(results: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  COSMIC ORBIT — Exoplanet Transit Detection Benchmark Report")
    lines.append("  BAH 2026 — ISRO x Hack2skill")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    total = len(results)
    passed = results["overall_pass"].sum() if "overall_pass" in results.columns else 0
    lines.append(f"  Targets tested: {total}")
    lines.append(f"  Passed:         {passed}/{total} ({100*passed/total:.0f}%)")
    lines.append("")

    # Per-target details
    for _, row in results.iterrows():
        target = row.get("target", "Unknown")
        status = "✅ PASS" if row.get("overall_pass") else "❌ FAIL"
        lines.append(f"  {target}: {status}")

        if row.get("error"):
            lines.append(f"    Error: {row['error']}")
            continue

        if pd.notna(row.get("detected_period")) and pd.notna(row.get("known_period")):
            err = row.get("period_error", 0) * 100
            p_status = "✓" if row.get("period_pass") else "✗"
            lines.append(f"    Period:   {row['detected_period']:.4f} days "
                        f"(known: {row['known_period']:.4f}) "
                        f"[error: {err:.2f}%] {p_status}")

        if pd.notna(row.get("detected_depth_ppm")) and pd.notna(row.get("known_depth_ppm")):
            err = row.get("depth_error", 0) * 100
            d_status = "✓" if row.get("depth_pass") else "✗"
            lines.append(f"    Depth:    {row['detected_depth_ppm']:.1f} ppm "
                        f"(known: {row['known_depth_ppm']:.1f}) "
                        f"[error: {err:.2f}%] {d_status}")

        if pd.notna(row.get("detected_duration")) and pd.notna(row.get("known_duration")):
            err = row.get("duration_error", 0) * 100
            dur_status = "✓" if row.get("duration_pass") else "✗"
            lines.append(f"    Duration: {row['detected_duration']:.2f} hrs "
                        f"(known: {row['known_duration']:.2f}) "
                        f"[error: {err:.2f}%] {dur_status}")

        if pd.notna(row.get("confidence")):
            lines.append(f"    Confidence: {row['confidence']:.2%}")

        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)
    logger.info(report)
    return report
