# Not All Proofs Are Equal: Evaluating LLM Proof Quality Beyond Correctness

This repository contains the code for the NeurIPS 2026 submission **Not All Proofs Are Equal: Evaluating LLM Proof Quality Beyond Correctness**.


# Installation

We offer 2 separate ways of installing the `proofrank` package, dependent on your whether you want to run the benchmark only using an API, or by locally running vLLM instances.

### API Inference

If you plan to only use API calls, you can create an environment and run:

```
pip install .
```

### Local Inference

If you plan to use local vLLM for inference, run:

```
conda install -f environment.yml -n proofrank
```

# Data Setup

### Preparing the raw data

The raw data is a list of objects that should reside within `data/raw/<project>/sample.json`. The only condition is that each entry should include a `problem`, and `problem_id` tags. 

Before running the solvers, you must define a `project` and a `solver` configuration inside `configs/`. Use the included examples as reference. 

### Run the solvers

First, prepare the data using:

```
python ./scripts/process.py --project <project>
```

Then, run the solver with:

```
python ./scripts/run.py --project <project>
```

After the script has finished, finalize the run by postprocessing the solutions:

```
python ./scripts/postprocess.py --project <project>
```

### Running a processor

We run our LLM-as-a-judge through processors, whose configs are specified in `configs/processors/`. Each processor can be run on top of a processed solver run, or on top of another processor.

To do so, run the following command:

```
python ./scripts/judge.py --checker_configs <list of models to use as judges/processors> --setting_config <processor_name>
```

# Running ProofRank

### Conciseness
1. **Prerequisites**
    - Complete the data preparation steps for the `matharena_proofs` (main) project.
2. Run the correctness verification using the `answer_checker` and `completeness_checker` processors.
3. Run the rephraser using the `verbosity_rephraser` processor.

### Computational Ease
1. **Prerequisites**:
    - Complete the data preparation steps for the `matharena_proofs` (main) project.
    - Process the pairs using the ```python ./scripts/data/create_pairwise_judgements.py --project <project>```
2. Run the correctness verification using the `answer_checker` and `completeness_checker` processors.
3. Run the pairwise judge using the `pairwise_computation_tie` processor.

### Cognitive Simplicity
The intstructions here are the same as above, but using the `pairwise_complexity_tie` processor.

### Diversity
1. **Prerequisites**
    - Complete the data preparation steps for the `diversity_samples` project.
    - (Optional) Merge the samples with those from the previous metrics using ```python ./scripts/data/merge_diversity_samples.py```
    - Run the `core_idea` processor on the solutions.
    - Merge them into the data using ```python ./scripts/data/parse_core_ideas.py -i <input_file> -p <processor> -o <output_file>```
2. Run the clustering using the `summary_diversity_clustering_main` processor.
3. Run the correctness verification with the processors `answer_checker_diversity`, `completeness_checker_diversity`


### Adaptivity

1. **Prerequisites**
    - Complete the data preparation steps for the `technique_adaptivity` project.
2. Run the correctness verification with the processors `answer_checker_technique`, `completeness_checker_technique`, and `technique_verifier`.

### Collating Results

You can see the available results by running the following script:

```
python ./scripts/results/eval_package.py
```

# Ablation Studies and Additional Experiments

### Conciseness Calibration (Sec. 4.2)
For this experiment, simply follow the conciseness calibration steps from above, generating the solutions from the `anti_verbosity` and `anti_verbosity_2` projects.

### Clustering Consistency Validation (App. B.3)
1. Prepare the diversity project, as described above.
2. Sample subsets using ```python ./scripts/data/generate_clustering_subsets.py -i <postprocessed_solutions> -o <output_file>```
3. Run the `sample_summary_diversity_clustering` processor.

### Rephrasing Validity (App. C.2)
Simply run the `verbosity_verifier` processor on top of the `verbosity_rephrase` results.

### Novelty of LLM Solutions (App. C.3)
For this, after preparing the diversity project, use the `summary_diversity_clustering_with_human` processor.


# Replicating our Figures

Inside the `notebooks/` folder, you can use the `plots.ipynb` notebook to re-generate the figures from our paper.

# Citation

If you use find the code or dataset useful, please cite us accordingly:

```
@inproceedings{petrov2026proofrank,
  title={Not All Proofs Are Equal: Evaluating {LLM} Proof Quality Beyond Correctness},
  author={Petrov, Ivo and Dekoninck, Jasper and Dimitrov, Dimitar I. and Vechev, Martin},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year={2026},
  url={https://openreview.net/forum?id=Zf9xKiTAom}
}
```

