import math
import pandas as pd
import numpy as np
from loguru import logger
from collections import defaultdict
from itertools import combinations
from .pairs_analysis import compute_bt_ratings



def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 2:
        return np.nan, n
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, n

    r = float(np.corrcoef(x, y)[0, 1])
    return r, n

def _pearson_raw_vs_elo(df_raw_means, df_elo):
    shared_metrics = [m for m in df_raw_means.columns if m in df_elo.columns]
    rows = []

    for metric in shared_metrics:
        r, n = _pearson(df_raw_means[metric].values, df_elo[metric].values)
        rows.append(
            {
                "metric": metric,
                "pearson_r_raw_vs_elo": r,
                "n_models": n,
                "abs_r": abs(r) if np.isfinite(r) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("abs_r", ascending=False)
        .drop(columns=["abs_r"])
    )

def _scale_bt(rating, is_negative):
    return (rating * (-400 if is_negative else 400)) + 1200


def _bootstrap_bt_elo_ci(df_pairs, models, is_negative,n_bootstrap = 1000,alpha = 0.05,C = 1.0, random_state = 42):
    if df_pairs.empty or len(models) == 0:
        return {m: (np.nan, np.nan) for m in models}

    rng = np.random.default_rng(random_state)
    n = len(df_pairs)
    draws = {m: [] for m in models}

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        df_boot = df_pairs.iloc[sample_idx]

        try:
            boot_ratings, _ = compute_bt_ratings(df_boot, C=C)
        except Exception:
            continue

        for m in models:
            if m in boot_ratings and np.isfinite(boot_ratings[m]):
                draws[m].append(_scale_bt(boot_ratings[m], is_negative))

    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)

    ci = {}
    for m in models:
        vals = np.asarray(draws[m], dtype=float)
        if vals.size == 0:
            ci[m] = (np.nan, np.nan)
        else:
            ci[m] = (float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q)))

    return ci

def _prepare_bt_dataframe(solver_data: dict, target_models, tie_threshold=None) -> pd.DataFrame:
    """
    Converts nested solver data {solver_id -> problem_id -> score}
    into a pairwise comparison dataframe.
    """
    records = []
    solvers = list(solver_data.keys())

    solver_pairs = list(combinations(solvers, 2))

    for solver_a, solver_b in solver_pairs:
        if target_models and (solver_a not in target_models or solver_b not in target_models):
            continue
        problems_a = set(solver_data[solver_a].keys())
        problems_b = set(solver_data[solver_b].keys())
        common_problems = problems_a.intersection(problems_b)

        for pid in common_problems:
            score_a = solver_data[solver_a][pid]
            score_b = solver_data[solver_b][pid]

            if tie_threshold is not None and math.isclose(score_a, score_b, rel_tol=tie_threshold):
                winner = "tie"
            elif score_a > score_b:
                winner = "model_a"
            elif score_b > score_a:
                winner = "model_b"
            else:
                winner = "tie"

            records.append({"model_a": solver_a, "model_b": solver_b, "winner": winner})

    return pd.DataFrame(records)


def _calculate_kendall_tau(df_pairs, bt_ratings):

    concordant = defaultdict(int)
    discordant = defaultdict(int)
    models = set(bt_ratings.keys())

    for _, row in df_pairs.iterrows():
        ma = row.get("model_a")
        mb = row.get("model_b")
        winner = row.get("winner")

        if winner == "tie" or pd.isna(winner):
            continue

        if ma not in bt_ratings or mb not in bt_ratings:
            continue

        score_a = bt_ratings[ma]
        score_b = bt_ratings[mb]

        is_concordant = False
        is_discordant = False

        if winner == "model_a":
            if score_a > score_b: is_concordant = True
            elif score_a < score_b: is_discordant = True
        elif winner == "model_b":
            if score_b > score_a: is_concordant = True
            elif score_b < score_a: is_discordant = True

        if is_concordant:
            concordant[ma] += 1
            concordant[mb] += 1
        elif is_discordant:
            discordant[ma] += 1
            discordant[mb] += 1

    tau_scores = {}
    for m in models:
        c = concordant[m]
        d = discordant[m]
        total = c + d
        tau_scores[m] = (c - d) / total if total > 0 else np.nan 

    return tau_scores

def report_rankings(raw_scores, pairwise_data=None, target_models=None):
    if pairwise_data is None:
        pairwise_data = {}

    all_settings = set(raw_scores.keys()).union(set(pairwise_data.keys()))
    settings = sorted(list(all_settings))

    TIE_THRESHOLD = 0.1
    N_BOOTSTRAP = 1000
    CI_ALPHA = 0.05
    BOOTSTRAP_SEED = 42

    TIE_METRICS = [
        "spacy_word_count_ratio",
    ]

    NEGATIVE_METRICS = [
        "solution_spacy_word_count",
        "spacy_word_count_ratio",
        "pairwise_computation_tie",
        "pairwise_complexity_tie",
    ]

    for setting in settings:
        do_bootstrap = setting in ["correct", "all"]
        print(f"\n{'='*20} Setting: {setting} {'='*20}\n")

        metrics_scalar = list(raw_scores.get(setting, {}).keys())
        metrics_pairs = list(pairwise_data.get(setting, {}).keys())
        all_metrics = sorted(list(set(metrics_scalar + metrics_pairs)))

        if not all_metrics:
            print("No metrics found.")
            continue

        print(f"--- Table 1.1: Mean Raw Scores ({setting}) ---")
        mean_score_rows = {}

        for metric in all_metrics:
            if metric in metrics_scalar:
                solver_data = raw_scores[setting][metric]
                for solver, problems in solver_data.items():
                    if solver not in mean_score_rows:
                        mean_score_rows[solver] = {}
                    mean_score_rows[solver][metric] = np.mean(
                        [
                            (
                                np.mean(problems[problem])
                                if problem in problems
                                else np.nan
                            )
                            for problem in problems
                        ]
                    )

        df_means = pd.DataFrame.from_dict(mean_score_rows, orient="index")
        df_means = df_means.reindex(columns=all_metrics)
        df_means.index.name = "Solver"
        df_means = df_means.dropna(axis=1, how="all")
        print(df_means.to_markdown(floatfmt=".2f"))
        print("\n")

        if setting.lower() in {"all", "correct"}:
            acc_col = next((c for c in df_means.columns if c.lower() == "accuracy"), None)

            print(f"--- Table 1.1b: Pearson Correlation with Accuracy ({setting}) ---")
            if acc_col is None:
                print("No 'accuracy' column found in scalar metrics.\n")
            else:
                corr_rows = []
                for metric in df_means.columns:
                    if metric == acc_col:
                        continue

                    r, n = _pearson(df_means[acc_col].values, df_means[metric].values)
                    corr_rows.append({
                        "metric": metric,
                        "pearson_r": r,
                        "n_models": n,
                        "abs_r": abs(r) if np.isfinite(r) else np.nan,
                    })

                df_corr = pd.DataFrame(corr_rows)
                if not df_corr.empty:
                    df_corr = df_corr.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"])
                    print(df_corr.to_markdown(index=False, floatfmt=".3f", missingval="-"))
                else:
                    print("No comparable metrics for correlation.")
                print("\n")

        print(
            f"--- Table 1.2: Bradley-Terry Elo Ratings with 95% Bootstrap CI ({setting}) ---\n"
            f"(Scale: *400 + 1200)"
        )
        bt_score_rows = defaultdict(dict)
        bt_ci_rows = defaultdict(dict)

        for metric in all_metrics:
            df_pairs = pd.DataFrame()

            if metric in metrics_scalar:
                solver_data = raw_scores[setting][metric]
                df_pairs = _prepare_bt_dataframe(
                    solver_data,
                    target_models,
                    tie_threshold=TIE_THRESHOLD if metric in TIE_METRICS else None
                )

            elif metric in metrics_pairs:
                data_list = pairwise_data[setting][metric]
                if data_list:
                    df_pairs = pd.DataFrame(data_list)

            if df_pairs.empty:
                continue

            try:
                ratings, _ = compute_bt_ratings(df_pairs, C=1.0)

                is_negative = any(neg in metric.lower() for neg in NEGATIVE_METRICS)

                if do_bootstrap:
                    ci_by_solver = _bootstrap_bt_elo_ci(
                        df_pairs=df_pairs,
                        models=list(ratings.keys()),
                        is_negative=is_negative,
                        n_bootstrap=N_BOOTSTRAP,
                        alpha=CI_ALPHA,
                        C=1.0,
                        random_state=BOOTSTRAP_SEED,
                    )

                for solver, rating in ratings.items():
                    scaled_rating = _scale_bt(rating, is_negative)
                    bt_score_rows[solver][metric] = scaled_rating
                    if do_bootstrap:
                        bt_ci_rows[solver][metric] = ci_by_solver.get(solver, (np.nan, np.nan))

            except Exception as e:
                logger.warning(f"BT Error on {metric}: {e}")
                pass

        # Numeric dataframe for sorting
        df_bt = pd.DataFrame.from_dict(bt_score_rows, orient="index")
        df_bt = df_bt.reindex(columns=all_metrics)
        if not df_bt.empty and len(df_bt.columns) > 0:
            first_col = df_bt.columns[0]
            df_bt = df_bt.sort_values(by=first_col, ascending=False)

        df_bt_display = pd.DataFrame(index=df_bt.index, columns=df_bt.columns, dtype=object)
        for solver in df_bt.index:
            for metric in df_bt.columns:
                val = df_bt.at[solver, metric]
                if pd.isna(val):
                    df_bt_display.at[solver, metric] = "-"
                    continue

                lo, hi = bt_ci_rows.get(solver, {}).get(metric, (np.nan, np.nan))
                if do_bootstrap and np.isfinite(lo) and np.isfinite(hi):
                    df_bt_display.at[solver, metric] = f"{val:.0f} [{lo:.0f}, {hi:.0f}]"
                else:
                    df_bt_display.at[solver, metric] = f"{val:.0f}"

        print(df_bt_display.to_markdown())
        print("\n")

        print("\n" + "-" * 60 + "\n")
