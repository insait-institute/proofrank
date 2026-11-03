import argparse
import os
import re
import json
import pandas as pd
import numpy as np
from proofrank.analysis.verbosity_prevention_utils import _model_accuracy_from_correctness_map, replacement_elo_change_for_model_swap
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from proofrank.parser import find_last_boxed_content
from proofrank.postprocess import fix_thinking
from proofrank.analysis.verbosity import print_verbosity_stats, calculate_length_timeout
from proofrank.analysis.problem_classification import fix_class
from proofrank.analysis.pairwise_comparison import report_rankings
from proofrank.diversity import DiversityAnalysisParser
from proofrank.technique_diversity import TechniqueDiversityParser
from proofrank.result_utils import parse_outputs
from collections import defaultdict
from proofrank.analysis.diversity_utils import *
from proofrank.analysis.verbosity_prevention_utils import *
parser = argparse.ArgumentParser()
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--configs-folder", type=str, default="configs/")
parser.add_argument(
    "--diversity-samples-path",
    type=str,
    default="data/postprocess/matharena_proofs/diversity_samples.json",
)

TARGET_MODELS = {
    'deepseek/deepseek_v32_think',
    'deepseek/deepseek_v4_pro',
    'gemini/gemini-3-flash',
    'gemini/gemini-31-pro',
    'gemini/gemini-37-flash',
    'openai/gpt-54',
    'openai/gpt-56-sol',
    'stepfun/3.5-flash',
    'glm/glm-5',
    'xai/grok-41-fast-reasoning',
    'moonshot/k25',
    'qwen/qwen35_397b_a17b_high',
    "openai/oss-120b",
}

JUDGE_NAME = "oss-120b"

def get_verbosity(dfs, name, raw_scores, correctness_map, topic_map, unique_topics, answer_correctness_map=None):
    suffix = name.split("_")[-1]
    total_count = 0
    valid_count = 0
    if suffix == "rephrase":
        suffix = None
    if name in dfs and not dfs[name].empty:
        if name.replace("rephrase", "verifier") in dfs and not dfs[name.replace("rephrase", "verifier")].empty:
            merged_df = dfs[name].merge(
                dfs[name.replace("rephrase", "verifier")][["solver_id", "problem_id", "true_grade"]],
                on=["solver_id", "problem_id"],
                suffixes=("", "_verifier"),
                how="left"
            )
            merged_df["is_valid_rephrase"] = merged_df["true_grade_verifier"].fillna("No").astype(str).str.contains("Yes", case=False)
        else:
            merged_df = dfs[name].copy()
            merged_df["is_valid_rephrase"] = True

        for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Verbosity"):
            solver = row['solver_id']
            pid = row['problem_id']

            sol_len_tuple = calculate_length_timeout(row['solution'])
            if sol_len_tuple[0] is None:
                continue
            sol_spacy_count = sol_len_tuple[3]

            if isinstance(row['outputs'], list):
                outputs = row['outputs']
            else:
                outputs = [row['outputs']]


            if not outputs: continue

            rephrase_stats = [calculate_length_timeout(o) for o in outputs]
            rephrase_stats = [s for s in rephrase_stats if s[0] is not None]
            if not rephrase_stats: continue
            avg_rephrase_spacy = sum(s[3] for s in rephrase_stats) / len(rephrase_stats) if rephrase_stats else 0

            ratio = sol_spacy_count / avg_rephrase_spacy if avg_rephrase_spacy > 0 else 0

            raw_scores['all']['spacy_word_count_ratio'][solver][pid] = ratio

            if row["is_valid_rephrase"]:
                raw_scores['valid']['spacy_word_count_ratio'][solver][pid] = ratio
                if correctness_map.get((solver, pid), False):
                    valid_count += 1
                    raw_scores['valid_correct']['spacy_word_count_ratio'][solver][pid] = ratio

            if correctness_map[(solver, pid)]:
                total_count += 1
                raw_scores['correct']['spacy_word_count_ratio'][solver][pid] = ratio
            if answer_correctness_map and answer_correctness_map.get((solver, pid), False):
                raw_scores['answer_correct']['spacy_word_count_ratio'][solver][pid] = ratio

            topic = topic_map.get(pid.split('-')[0])
            if topic and topic in unique_topics:
                raw_scores[topic]['spacy_word_count_ratio'][solver][pid] = ratio
                if correctness_map[(solver, pid)]:
                    raw_scores[f"{topic}_correct"]['spacy_word_count_ratio'][solver][pid] = ratio
    if name.replace("rephrase", "verifier") in dfs:
        print(f"Total valid rephrases for {name}: {valid_count} out of {total_count}")

def get_correctness(dfs, suffix="", include_completeness=True):
    correctness_map = {}
    ans_checker_key = f"answer_checker_{suffix}" if suffix else "answer_checker"
    comp_checker_key = f"completeness_checker_{suffix}" if suffix else "completeness_checker"

    if ans_checker_key in dfs and not dfs[ans_checker_key].empty:
        for i, row in dfs[ans_checker_key].iterrows():
            is_answer_correct = False
            if "judgements" in row:
                corrects = ["incorrect" not in j.lower() for j in row["judgements"]]
                is_answer_correct = sum(corrects) / len(corrects) >= 0.5
            elif isinstance(row["outputs"], list):
                try:
                    corrects = [
                        "incorrect" not in find_last_boxed_content(j)[0].lower()
                        for j in row["outputs"]
                        if find_last_boxed_content(j)[0] is not None
                    ]
                    is_answer_correct = sum(corrects) / len(corrects) >= 0.5
                except:
                    pass
            else:
                is_answer_correct = "incorrect" not in row.get("true_grade", False).lower()
            correctness_map[(row["solver_id"], row["problem_id"])] = is_answer_correct
    if not include_completeness:
        return correctness_map
    if comp_checker_key in dfs and not dfs[comp_checker_key].empty:
        for i, row in dfs[comp_checker_key].iterrows():
            is_complete = False
            if "judgements" in row:
                corrects = ["incomplete" not in j.lower() for j in row["judgements"]]
                is_complete = sum(corrects) / len(corrects) >= 0.5
            elif isinstance(row["outputs"], list):
                try:
                    corrects = [
                        "incomplete" not in find_last_boxed_content(j)[0].lower()
                        for j in row["outputs"]
                        if find_last_boxed_content(j)[0] is not None
                    ]
                    is_complete = sum(corrects) / len(corrects) >= 0.5
                except:
                    pass
            else:
                is_complete = "incomplete" not in row.get("true_grade", False).lower()
            correctness_map[(row["solver_id"], row["problem_id"])] = correctness_map.get((row["solver_id"], row["problem_id"]), False) and is_complete
    
    return correctness_map

def get_clusters_from_output(parser, output_data, model_id, problem_id):
    """Helper to parse diversity outputs using the notebook logic"""

    success = parser.parse(output_data)
    if not success:
        return None

    model_clusters_data = []
    human_clusters_data = []

    MODEL_LIMIT_IDX = 4

    for cluster in parser.clusters:
        cluster_name = cluster["cluster_name"]
        members = cluster["members"]

        model_vars = [m for m in members if m <= MODEL_LIMIT_IDX]
        human_vars = [m for m in members if m > MODEL_LIMIT_IDX]

        if model_vars:
            model_clusters_data.append({"name": cluster_name, "size": len(model_vars)})
        if human_vars:
            human_clusters_data.append({"name": cluster_name, "size": len(human_vars)})

    return {
        "model_clusters": model_clusters_data,
        "human_clusters": human_clusters_data,
    }


if __name__ == "__main__":
    args = parser.parse_args()
    dfs = {}

    diversity_solution_order = load_diversity_solution_order(
        args.diversity_samples_path
    )

    results_cache_dir = "data/results/"
    os.makedirs(results_cache_dir, exist_ok=True)

    eval_settings = ["answer_checker", "completeness_checker", "verbosity_rephrase", "topic_classifier", "verbosity_verifier", "verbosity_rephrase_2", "answer_checker_verbosity", "completeness_checker_verbosity", "answer_checker_verbosity_2", "completeness_checker_verbosity_2", "technique_verifier", "answer_checker_technique", "completeness_checker_technique", "answer_checker_diversity", "completeness_checker_diversity", "summary_diversity_clustering_main" ,"summary_diversity_clustering_with_human", "verbosity_rephrase_2", "verbosity_rephrase_3"]

    for setting in eval_settings:
        cache_path = os.path.join(results_cache_dir, f"{setting}.pkl")

        if os.path.exists(cache_path):
            logger.info(f"Loading cached {setting} from {cache_path}")
            df = pd.read_pickle(cache_path)
            dfs[setting] = df
            continue
        # # ----------------------------

        if not os.path.exists(os.path.join(args.output_folder, setting)):
            logger.warning(
                f"Setting folder {setting} not found and no cache exists, skipping..."
            )
            continue

        logger.info(f"Processing setting from raw logs: {setting}")
        df = parse_outputs(args.output_folder, args.configs_folder, setting, target_models=TARGET_MODELS if setting != 'topic_classifier' else ['human/human'], judge_name=JUDGE_NAME)# if setting != 'verbosity_rephrase' else "grok-41-fast-reasoning")
        df = df[df['is_valid']]
        if not df.empty:
            logger.info(f"Saving {setting} to cache at {cache_path}")
            df.to_pickle(cache_path)

        dfs[setting] = df

    logger.info("Loading problem classifications and solution correctness...")

    topic_map = {}
    if "topic_classifier" in dfs and not dfs["topic_classifier"].empty:
        for i, row in dfs["topic_classifier"].iterrows():
            try:
                if isinstance(row["outputs"], str):
                    row['outputs'] = [row["outputs"]]
                tags = [find_last_boxed_content(output)[0] for output in row["outputs"]]
                tags = [fix_class(tag.replace("\n", " ").strip()) for tag in tags if tag]
                # majority vote
                if tags:
                    tag_counts = defaultdict(int)
                    for tag in tags:
                        tag_counts[tag] += 1
                    tags = sorted(tag_counts.keys(), key=lambda x: tag_counts[x], reverse=True)
                    topic_map[row["problem_id"]] = tags[0]
            except Exception as e:
                topic_map[row["problem_id"]] = "Unknown"
    unique_topics = set(topic_map.values()) - {None, "Unknown"}

    for topic in unique_topics:
        logger.info(f"Topic '{topic}' has {sum(1 for t in topic_map.values() if t == topic)} problems.")

    # Load Correctness
    correctness_map = get_correctness(dfs)
    correctness_map_diversity = get_correctness(dfs, suffix="diversity")
    correctness_map_technique = get_correctness(dfs, suffix="technique")
    correctness_map_verbosity = get_correctness(dfs, suffix="verbosity")
    correctness_map_verbosity_2 = get_correctness(dfs, suffix="verbosity_2")
    answer_correctness_map = get_correctness(dfs, include_completeness=False)
    answer_correctness_map_diversity = get_correctness(dfs, include_completeness=False, suffix="diversity")
    technique_problems = set([tech[1].split('-tech')[0] for tech in correctness_map_technique])

    raw_scores = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    pairwise_data = defaultdict(lambda: defaultdict(list))

    logger.info("Calculating Base Accuracy Metrics...")

    for setting, correctness_dict in zip(
        ["all", "diversity", "technique"],
        [correctness_map, correctness_map_diversity, correctness_map_technique],
    ):
        for (solver_id, prob_id), is_valid in correctness_dict.items():
            score = 1.0 if is_valid else 0.0

            raw_scores[setting]["accuracy"][solver_id][prob_id] = score * 100
            if prob_id in technique_problems:
                raw_scores[setting]["accuracy_tech_problems"][solver_id][prob_id] = score * 100
            if setting == "all":
                t = topic_map.get(prob_id)
                if not t:
                    t = topic_map.get(prob_id.split("-")[0])

                if t and t in unique_topics:
                    raw_scores[t]["accuracy"][solver_id][prob_id] = score * 100
    verbosity_scores_2 = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    verbosity_scores_3 = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    logger.info("Calculating Part 1: Verbosity Metrics...")
    get_verbosity(dfs, "verbosity_rephrase", raw_scores, correctness_map, topic_map, unique_topics, answer_correctness_map)
    get_verbosity(dfs, "verbosity_rephrase_2", verbosity_scores_2, correctness_map_verbosity, topic_map, unique_topics)
    get_verbosity(dfs, "verbosity_rephrase_3", verbosity_scores_3, correctness_map_verbosity_2, topic_map, unique_topics)

    logger.info("Calculating Part 2: Technique Diversity Metrics...")

    if "technique_verifier" in dfs and not dfs["technique_verifier"].empty:
        tech_parser = TechniqueDiversityParser()

        for idx, row in tqdm(dfs["technique_verifier"].iterrows(), total=len(dfs["technique_verifier"]), desc="Technique"):
            solver = row['solver_id']
            pid = row['problem_id']

            verdicts = []
            if isinstance(row['outputs'], str):
                row['outputs'] = [row['outputs']]
            for output in row['outputs']:
                tech_parser.parse(output)
                verdict = getattr(tech_parser, 'verdict', 'unknown')
                verdicts.append(verdict)
            accuracy_score = int(sum([1 for v in verdicts if not "incorrect" in v.lower() and "unknown" not in v.lower()]) / len(verdicts) >= 0.5)
            raw_scores['all']['technique_accuracy'][solver][pid] = int(accuracy_score and correctness_map_technique[(solver, pid)])*100
            base_pid = pid.split('-')[0]
            if correctness_map_technique[(solver, pid)]:
                raw_scores['correct']['technique_accuracy'][solver][pid] = accuracy_score*100

            topic = topic_map.get(base_pid) or topic_map.get(pid)
            if topic and topic in unique_topics:
                raw_scores[topic]['technique_accuracy'][solver][pid] = int(accuracy_score and correctness_map_technique[(solver, pid)])*100
                if correctness_map_technique[(solver, pid)]:
                    raw_scores[f"{topic}_correct"]['technique_accuracy'][solver][pid] = int(accuracy_score and correctness_map_technique[(solver, pid)])*100

    logger.info("Calculating Part 3: Diversity Clustering Metrics...")
    if (
        "summary_diversity_clustering_main" in dfs
        and not dfs["summary_diversity_clustering_main"].empty
    ):
        div_parser = DiversityAnalysisParser()

        for idx, row in tqdm(
            dfs["summary_diversity_clustering_main"].iterrows(),
            total=len(dfs["summary_diversity_clustering_main"]),
            desc="Clustering main",
        ):
            solver = row["solver_id"]
            pid = row["problem_id"]
            base_pid = get_base_problem_id(pid)

            clusters = parse_diversity_clusters(div_parser, row["outputs"])

            if not clusters:
                continue

            entries = get_main_experiment_entries(
                row,
                diversity_solution_order=diversity_solution_order,
            )

            if not entries:
                continue

            # Main experiment has local prompt indices:
            #   1, 2, 3, 4
            all_indices = [entry["idx"] for entry in entries]

            all_counts = cluster_counts_for_indices(
                clusters,
                all_indices,
            )

            entropy_score = entropy_from_counts(all_counts)

            add_diversity_metric(
                raw_scores=raw_scores,
                metric_name="diversity_entropy",
                model_id=solver,
                problem_id=base_pid,
                value=entropy_score,
                topic_map=topic_map,
                unique_topics=unique_topics,
                correctness_bucket=False,
            )

            correct_entries = [
                entry for entry in entries
                if solution_is_correct(entry, correctness_map_diversity)
            ]

            if correct_entries:
                correct_indices = [entry["idx"] for entry in correct_entries]

                correct_counts = cluster_counts_for_indices(
                    clusters,
                    correct_indices,
                )

                correct_entropy_score = entropy_from_counts(correct_counts)

                add_diversity_metric(
                    raw_scores=raw_scores,
                    metric_name="diversity_entropy",
                    model_id=solver,
                    problem_id=base_pid,
                    value=correct_entropy_score,
                    topic_map=topic_map,
                    unique_topics=unique_topics,
                    correctness_bucket=True,
                )

            answer_correct_entries = [
                entry for entry in entries
                if solution_is_correct(entry, answer_correctness_map_diversity)
            ]

            if answer_correct_entries:
                answer_correct_indices = [entry["idx"] for entry in answer_correct_entries]

                answer_correct_counts = cluster_counts_for_indices(
                    clusters,
                    answer_correct_indices,
                )

                answer_correct_entropy_score = entropy_from_counts(answer_correct_counts)

                add_diversity_metric(
                    raw_scores=raw_scores,
                    metric_name="diversity_entropy",
                    model_id=solver,
                    problem_id=base_pid,
                    value=answer_correct_entropy_score,
                    topic_map=topic_map,
                    unique_topics=unique_topics,
                    correctness_bucket=False,
                    answer_correctness_bucket=True,
                )


    if "summary_diversity_clustering_with_human" in dfs and not dfs["summary_diversity_clustering_with_human"].empty:
        div_parser = DiversityAnalysisParser()

        for idx, row in tqdm(
            dfs["summary_diversity_clustering_with_human"].iterrows(),
            total=len(dfs["summary_diversity_clustering_with_human"]),
            desc="Clustering with human",
        ):
            clusters = parse_diversity_clusters(div_parser, row["outputs"])

            if not clusters:
                continue

            entries = get_compiled_solution_entries(row, diversity_solution_order=diversity_solution_order)

            if not entries:
                continue

            human_entries = [entry for entry in entries if is_human_model(entry["model_id"])]

            nonhuman_entries = [entry for entry in entries if not is_human_model(entry["model_id"])]

            if not nonhuman_entries:
                continue

            human_indices = [entry["idx"] for entry in human_entries]
            nonhuman_indices = [entry["idx"] for entry in nonhuman_entries]

            human_cluster_names = cluster_names_for_indices(clusters,human_indices)
            global_nonhuman_counts = cluster_counts_for_indices(clusters,nonhuman_indices,)

            correct_nonhuman_entries = [entry for entry in nonhuman_entries if solution_is_correct(entry, correctness_map_diversity)]

            correct_nonhuman_indices = [entry["idx"] for entry in correct_nonhuman_entries]

            correct_global_nonhuman_counts = cluster_counts_for_indices(clusters, correct_nonhuman_indices)
            entries_by_model = group_entries_by_model(nonhuman_entries, exclude_human=True)

            add_diversity_metric(
                raw_scores=raw_scores,
                metric_name="diversity_novelty",
                model_id="human/human",
                problem_id=base_pid,
                value=(len(human_cluster_names.intersection(global_nonhuman_counts.keys()))/ len(human_cluster_names)) if human_cluster_names else 0.0,
                topic_map=topic_map,
                unique_topics=unique_topics,
                correctness_bucket=False,
            )

            for model_id, model_entries in entries_by_model.items():
                if not model_entries:
                    continue

                base_pid = model_entries[0]["base_problem_id"]
                model_indices = [entry["idx"] for entry in model_entries]

                model_counts = cluster_counts_for_indices(
                    clusters,
                    model_indices,
                )

                if not model_counts:
                    continue

                model_cluster_names = set(model_counts.keys())

                cross_entropy_score = cross_entropy_from_counts(model_counts=model_counts,global_counts=global_nonhuman_counts)

                if cross_entropy_score is None:
                    continue

                entropy_score = entropy_from_counts(model_counts)
                count_score = float(len(model_cluster_names))

                if human_cluster_names:
                    cover_score = (len(model_cluster_names.intersection(human_cluster_names))/ len(human_cluster_names))
                else:
                    cover_score = 0.0

                if model_cluster_names:
                    novelty_score = (len(model_cluster_names - human_cluster_names)/ len(model_cluster_names))
                else:
                    novelty_score = 0.0

                add_diversity_metric(
                    raw_scores=raw_scores,
                    metric_name="diversity_cross_entropy",
                    model_id=model_id,
                    problem_id=base_pid,
                    value=cross_entropy_score,
                    topic_map=topic_map,
                    unique_topics=unique_topics,
                    correctness_bucket=False,
                )

                add_diversity_metric(
                    raw_scores=raw_scores,
                    metric_name="diversity_cover",
                    model_id=model_id,
                    problem_id=base_pid,
                    value=cover_score,
                    topic_map=topic_map,
                    unique_topics=unique_topics,
                    correctness_bucket=False,
                )

                add_diversity_metric(
                    raw_scores=raw_scores,
                    metric_name="diversity_novelty",
                    model_id=model_id,
                    problem_id=base_pid,
                    value=novelty_score,
                    topic_map=topic_map,
                    unique_topics=unique_topics,
                    correctness_bucket=False,
                )

                correct_model_entries = [entry for entry in model_entries if solution_is_correct(entry, correctness_map_diversity)]

                if not correct_model_entries:
                    continue

                correct_model_indices = [entry["idx"] for entry in correct_model_entries]

                correct_model_counts = cluster_counts_for_indices(clusters, correct_model_indices)

                if not correct_model_counts:
                    continue

                correct_model_cluster_names = set(correct_model_counts.keys())

                correct_cross_entropy_score = cross_entropy_from_counts(model_counts=correct_model_counts,global_counts=correct_global_nonhuman_counts)

                if correct_cross_entropy_score is None:
                    continue

                correct_entropy_score = entropy_from_counts(correct_model_counts)
                correct_count_score = float(len(correct_model_cluster_names))

                if human_cluster_names:
                    correct_cover_score = (len(correct_model_cluster_names.intersection(human_cluster_names)) / len(human_cluster_names))
                else:
                    correct_cover_score = 0.0

                if correct_model_cluster_names:
                    correct_novelty_score = (len(correct_model_cluster_names - human_cluster_names) / len(correct_model_cluster_names))
                else:
                    correct_novelty_score = 0.0

                add_diversity_metric(
                    raw_scores=raw_scores,
                    metric_name="diversity_cross_entropy",
                    model_id=model_id,
                    problem_id=base_pid,
                    value=correct_cross_entropy_score,
                    topic_map=topic_map,
                    unique_topics=unique_topics,
                    correctness_bucket=True,
                )

                add_diversity_metric(
                    raw_scores=raw_scores,
                    metric_name="diversity_cover",
                    model_id=model_id,
                    problem_id=base_pid,
                    value=correct_cover_score,
                    topic_map=topic_map,
                    unique_topics=unique_topics,
                    correctness_bucket=True,
                )

                add_diversity_metric(
                    raw_scores=raw_scores,
                    metric_name="diversity_novelty",
                    model_id=model_id,
                    problem_id=base_pid,
                    value=correct_novelty_score,
                    topic_map=topic_map,
                    unique_topics=unique_topics,
                    correctness_bucket=True,
                )

    logger.info("Processing Part 4: Pairwise Analysis...")

    pairwise_projects = [
        "pairwise_computation_tie",
        "pairwise_complexity_tie",
    ]

    judge = "openai_oss-120b"
    for project in pairwise_projects:
        loaded_cache = True

        base_path = Path(args.output_folder) / project / judge.replace("_", "/", 1)
        if not base_path.exists():
            continue

        json_files = list(base_path.rglob("*.json"))

        for file_path in tqdm(json_files, desc=f"Loading {project}", leave=False):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                model_a = raw["model_id"]

                try:
                    parts = raw["problem_id"].split("-part")
                    actual_pid = parts[0]
                    model_b = parts[1].replace("_", "/", 1)
                except:
                    continue

                if model_a not in TARGET_MODELS or model_b not in TARGET_MODELS:
                    continue

                winner = "tie"
                if raw["majority_vote"] == "1":
                    winner = "model_a"
                elif raw["majority_vote"] == "2":
                    winner = "model_b"

                record = {"model_a": model_a, "model_b": model_b, "winner": winner}

                cA = correctness_map[(model_a, actual_pid)]
                cB = correctness_map[(model_b, actual_pid)]
                both_correct = cA and cB

                aA = answer_correctness_map.get((model_a, actual_pid), False)
                aB = answer_correctness_map.get((model_b, actual_pid), False)
                both_answer_correct = aA and aB
                topic = topic_map.get(actual_pid)

                pairwise_data["all"][project].append(record)

                if both_correct:
                    pairwise_data['correct'][project].append(record)

                if both_answer_correct:
                    pairwise_data['answer_correct'][project].append(record)

                if topic:
                    pairwise_data[topic][project].append(record)
                    if both_correct:
                        pairwise_data[f"{topic}_correct"][project].append(record)

            except Exception as e:
                continue
        for setting in pairwise_data:
            cache_path = os.path.join(
                results_cache_dir, f"{project}_{setting}_{judge}.json"
            )
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(pairwise_data[setting][project], f, indent=4)

    logger.info("Raw scores loaded successfully for specified settings and models.")
    print(pairwise_data["all"].keys())
    print(raw_scores["all"].keys())
    report_rankings(raw_scores, pairwise_data, TARGET_MODELS)

    print("\n" + "=" * 25 + " Replacement Elo Impact (correct) " + "=" * 25)
    metric = "spacy_word_count_ratio"
    df_base_acc = _model_accuracy_from_correctness_map(correctness_map, target_models=TARGET_MODELS)
    base_acc_map = dict(zip(df_base_acc["model"], df_base_acc["accuracy_pct"]))

    anti_variants = [
        ("verbosity_rephrase_2", verbosity_scores_2, correctness_map_verbosity),
        ("verbosity_rephrase_3", verbosity_scores_3, correctness_map_verbosity_2),
    ]

    for anti_name, anti_scores, anti_corr_map in anti_variants:
        print(f"\n--- {anti_name}: replace one model at a time in 'correct' for metric '{metric}' ---")

        df_anti_acc = _model_accuracy_from_correctness_map(anti_corr_map, target_models=TARGET_MODELS)
        anti_acc_map = dict(zip(df_anti_acc["model"], df_anti_acc["accuracy_pct"]))

        df_delta = replacement_elo_change_for_model_swap(
            base_scores=raw_scores,
            anti_scores=anti_scores,
            metric=metric,
            setting="correct",
            target_models=TARGET_MODELS,
            baseline_acc_map=base_acc_map,
            anti_acc_map=anti_acc_map,
        )

        if df_delta.empty:
            print("No data available.\n")
            continue

        print(df_delta.to_markdown(index=False, floatfmt=".1f", missingval="-"))
