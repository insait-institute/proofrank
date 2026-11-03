import math
from itertools import combinations
from .pairs_analysis import compute_bt_ratings
import pandas as pd
import numpy as np
from collections import defaultdict

NEGATIVE_METRICS = {
    "solution_spacy_word_count",
    "spacy_word_count_ratio",
    "pairwise_computation_tie",
    "pairwise_complexity_tie",
}

TIE_METRICS = {"spacy_word_count_ratio"}
TIE_THRESHOLD = 0.1

def _model_accuracy_from_correctness_map(correctness_dict: dict, target_models=None) -> pd.DataFrame:
    counts = defaultdict(int)
    correct = defaultdict(int)

    for (solver, _pid), is_ok in correctness_dict.items():
        if target_models and solver not in target_models:
            continue
        counts[solver] += 1
        correct[solver] += int(bool(is_ok))

    rows = []
    for m in sorted(counts.keys()):
        n = counts[m]
        acc = (100.0 * correct[m] / n) if n > 0 else np.nan
        rows.append({"model": m, "accuracy_pct": acc, "n_items": n})

    return pd.DataFrame(rows)


def _prepare_pairwise_from_scalar_scores(solver_data, target_models=None, tie_threshold=None):
    records = []
    solvers = list(solver_data.keys())

    for solver_a, solver_b in combinations(solvers, 2):
        if target_models and (solver_a not in target_models or solver_b not in target_models):
            continue

        pa = set(solver_data.get(solver_a, {}).keys())
        pb = set(solver_data.get(solver_b, {}).keys())
        common = pa.intersection(pb)

        for pid in common:
            sa = solver_data[solver_a][pid]
            sb = solver_data[solver_b][pid]

            if tie_threshold is not None and math.isclose(sa, sb, rel_tol=tie_threshold):
                winner = "tie"
            elif sa > sb:
                winner = "model_a"
            elif sb > sa:
                winner = "model_b"
            else:
                winner = "tie"

            records.append({"model_a": solver_a, "model_b": solver_b, "winner": winner})

    return pd.DataFrame(records)


def _scale_bt_to_elo(metric: str, rating: float) -> float:
    is_negative = any(neg in metric.lower() for neg in NEGATIVE_METRICS)
    return (rating * (-400 if is_negative else 400)) + 1200


def _compute_elo_for_scalar_metric(
    setting_scores,
    metric,
    target_models=None,
):
    solver_data = setting_scores.get(metric, {})
    if not solver_data:
        return {}

    tie_threshold = TIE_THRESHOLD if metric in TIE_METRICS else None
    df_pairs = _prepare_pairwise_from_scalar_scores(
        solver_data,
        target_models=target_models,
        tie_threshold=tie_threshold,
    )
    if df_pairs.empty:
        return {}

    ratings, _ = compute_bt_ratings(df_pairs, C=1.0)
    return {m: _scale_bt_to_elo(metric, r) for m, r in ratings.items()}


def replacement_elo_change_for_model_swap(
    base_scores,
    anti_scores,
    metric = "spacy_word_count_ratio",
    setting = "correct",
    target_models=None,
    baseline_acc_map: dict | None = None,
    anti_acc_map: dict | None = None,
):
    base_setting = base_scores.get(setting, {})
    anti_setting = anti_scores.get(setting, {})

    base_metric_data = base_setting.get(metric, {})
    anti_metric_data = anti_setting.get(metric, {})

    if not base_metric_data or not anti_metric_data:
        return pd.DataFrame()

    baseline_elo = _compute_elo_for_scalar_metric(base_setting, metric, target_models=target_models)
    if not baseline_elo:
        return pd.DataFrame()

    candidate_models = sorted(set(base_metric_data.keys()).intersection(set(anti_metric_data.keys())))
    if target_models:
        candidate_models = [m for m in candidate_models if m in target_models]

    rows = []
    for model in candidate_models:
        cf_metric_data = {s: dict(prob_scores) for s, prob_scores in base_metric_data.items()}
        cf_metric_data[model] = dict(anti_metric_data[model])

        cf_setting = dict(base_setting)
        cf_setting[metric] = cf_metric_data

        cf_elo = _compute_elo_for_scalar_metric(cf_setting, metric, target_models=target_models)

        base_val = baseline_elo.get(model, np.nan)
        cf_val = cf_elo.get(model, np.nan)
        delta = cf_val - base_val if np.isfinite(base_val) and np.isfinite(cf_val) else np.nan

        base_acc = baseline_acc_map.get(model, np.nan) if baseline_acc_map else np.nan
        anti_acc = anti_acc_map.get(model, np.nan) if anti_acc_map else np.nan
        delta_acc = anti_acc - base_acc if np.isfinite(base_acc) and np.isfinite(anti_acc) else np.nan

        rows.append(
            {
                "model": model,
                "baseline_acc_pct": base_acc,      # NEW
                "anti_acc_pct": anti_acc,          # NEW
                "delta_acc_pct": delta_acc,        # NEW
                "baseline_elo": base_val,
                "counterfactual_elo": cf_val,
                "delta_elo": delta,
                "baseline_n": len(base_metric_data.get(model, {})),
                "anti_n": len(anti_metric_data.get(model, {})),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("delta_elo", ascending=False)