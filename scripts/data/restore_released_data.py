import json
from pathlib import Path

from datasets import load_dataset

REPO_ID = "INSAIT-Institute/ProofRank-outputs"
RESULTS_DIR = Path("data/results")
OUTPUTS_DIR = Path("outputs")
DIVERSITY_SAMPLES_PATH = Path("data/postprocess/matharena_proofs/diversity_samples.json")

CACHED_SETTINGS = [
    "answer_checker",
    "completeness_checker",
    "topic_classifier",
    "verbosity_rephrase",
    "verbosity_rephrase_2",
    "verbosity_rephrase_3",
    "verbosity_verifier",
    "answer_checker_verbosity",
    "completeness_checker_verbosity",
    "answer_checker_verbosity_2",
    "completeness_checker_verbosity_2",
    "technique_verifier",
    "answer_checker_technique",
    "completeness_checker_technique",
    "answer_checker_diversity",
    "completeness_checker_diversity",
    "summary_diversity_clustering_main",
    "summary_diversity_clustering_with_human",
]


def restore_setting(name: str) -> None:
    df = load_dataset(REPO_ID, name, split="train").to_pandas()
    if "extra_metadata" in df.columns:
        df["extra_metadata"] = df["extra_metadata"].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
    for col in ("outputs", "judgements"):
        if col not in df.columns:
            continue
        flag_col = f"_{col}_was_scalar"
        if flag_col in df.columns:
            df[col] = [
                (list(v)[0] if len(v) else None) if scalar else (list(v) if v is not None else [])
                for v, scalar in zip(df[col], df[flag_col])
            ]
            df = df.drop(columns=[flag_col])
        else:
            df[col] = df[col].apply(lambda v: list(v) if v is not None else [])
    out_path = RESULTS_DIR / f"{name}.pkl"
    df.to_pickle(out_path)
    print(f"restored {name}: {len(df)} rows -> {out_path}")


def restore_pairwise_raw() -> None:
    ds = load_dataset(REPO_ID, "pairwise_raw", split="train")
    for row in ds:
        out_path = OUTPUTS_DIR / row["project"] / row["judge"] / row["relpath"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model_id": row["model_id"],
            "problem_id": row["problem_id"],
            "majority_vote": row["majority_vote"],
            "judgements": json.loads(row["judgements"]) if row["judgements"] else None,
            "outputs": json.loads(row["outputs"]) if row["outputs"] else None,
        }
        with open(out_path, "w") as f:
            json.dump(record, f)
    print(f"restored pairwise_raw: {len(ds)} files -> {OUTPUTS_DIR}")


def restore_diversity_samples() -> None:
    ds = load_dataset(REPO_ID, "diversity_samples", split="train")
    DIVERSITY_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DIVERSITY_SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump([json.loads(row["record"]) for row in ds], f)
    print(f"restored diversity_samples: {len(ds)} entries -> {DIVERSITY_SAMPLES_PATH}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in CACHED_SETTINGS:
        restore_setting(name)
    restore_pairwise_raw()
    restore_diversity_samples()
