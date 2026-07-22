# Sequential-EQA

Sequential-EQA evaluates whether embodied question-answering systems can reuse scene knowledge across a sequence of questions. The episodic protocol clears memory before every question. The sequential protocol presents the same ordered questions and preserves only designated scene memory between them.

## Repository structure

```text
benchmark/    Benchmark construction and the released Open-EQA HM3D sequences
evaluation/   Local answer grading and paired benchmark metrics
adapters/     Common adapter contract and four model integrations
scripts/      Setup and episodic/sequential launch commands
configs/      Reproducible model settings and upstream revision lock
external/     Ignored location for third-party repositories
```

Generated outputs, HM3D data, checkpoints, credentials, and third-party repositories are excluded from version control.

## Benchmark data

The canonical release is under `benchmark/sequence_files/openeqa_hm3d/`: 50 HM3D scenes and 498 questions in their paper order. `manifest.json` records counts, construction settings, and SHA-256 hashes. The original dataset revision and seed were unavailable in the research workspace and are marked accordingly.

Validate or rebuild a collection with:

```bash
python benchmark/build_sequences.py \
  --input benchmark/sequence_files/openeqa_hm3d \
  --output /tmp/sequential-eqa-validation \
  --keep-duplicates
```

See `benchmark/README.md` for the source schema and filtering behavior.

## Installation

Install the lightweight orchestration dependencies:

```bash
python -m pip install -r requirements.txt
```

Set the common data paths:

```bash
export HM3D_DATA_ROOT=/path/to/hm3d-val
export HM3D_INITIAL_POSES=/path/to/scene_init_poses.csv
```

Then install the desired upstream system. Exact immutable revisions are recorded in `configs/upstreams.yaml`:

```bash
bash scripts/setup_exploreeqa.sh
bash scripts/setup_memoryeqa.sh
bash scripts/setup_3d_mem.sh
bash scripts/setup_uninavid.sh
```

Each setup command clones or fast-forwards the official repository to its current default branch. The inspected commit is recorded in `configs/upstreams.yaml`, while rerunning setup deliberately pulls newer upstream changes. Each model has substantial CUDA, Habitat, checkpoint, and API requirements; finish its upstream installation instructions and set the corresponding checkpoint environment variable.

## Running the protocols

Run one scene first to verify paths and GPU dependencies:

```bash
scripts/run_episodic.sh exploreeqa --output-dir outputs --scene 00801-HaxA7YrQdEC
scripts/run_sequential.sh exploreeqa --output-dir outputs --scene 00801-HaxA7YrQdEC
```

Multiple models can be supplied before the options. Add `--dry-run` to print the isolated Conda commands without launching inference.

Each completed run writes native model artifacts and a canonical `results.jsonl`. The standardized fields include model/mode identifiers, scene and question IDs, prediction, reference answer, path length, navigation time, step count, status, and error metadata.

## Grading and metrics

Install `transformers>=4.56`, `torch`, and `accelerate` in a grading environment, then run:

```bash
python evaluation/grade_answers.py \
  --input outputs/episodic/results.jsonl \
  --output outputs/episodic/graded.jsonl

python evaluation/compute_metrics.py \
  --episodic outputs/episodic/graded.jsonl \
  --sequential outputs/sequential/graded.jsonl \
  --output-dir outputs/metrics
```

The default grader is `Qwen/Qwen3-8B-FP8`. Metrics include normalized Open-EQA score, binary success, memory advantage, and path/time/step advantages, with scene and question-position breakdowns. See `evaluation/README.md` for definitions.

## Reproducibility notes

- All four adapters import code directly from the official repositories under `external/`; no sibling research checkout is used.
- Setup tracks the current upstream branch, while the last inspected commit is recorded for compatibility diagnosis.
- The four model environments remain isolated because their dependencies conflict.
- The adapter state table in `adapters/README.md` is the protocol definition: scene memory may persist, but question-specific state and initial pose are reset.
- Do not commit HM3D assets, model checkpoints, generated outputs, API keys, or cloned upstream repositories.
