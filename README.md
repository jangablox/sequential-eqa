<div align="center">

# Sequential-EQA

### Beyond Episodic Evaluation: Memory Architectural Bottlenecks in Sequential Embodied Question Answering

[![Benchmark](https://img.shields.io/badge/Benchmark-Sequential--EQA-4d5eff?style=for-the-badge)](#benchmark)
[![Code](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](#quick-start)
[![Paper](https://img.shields.io/badge/Paper-IROS%202026-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](iros26_camera_ready_beyond_episodic_1.pdf)

</div>

<!--
TODO: Replace the local/section badge targets with the public project page,
paper, benchmark, and repository URLs when they are available.
-->

## News

- **Coming soon:** Public project, paper, benchmark, and repository links will be added here.

## Resources

- **Benchmark:** [50 HM3D scene sequences with 498 questions](benchmark/sequence_files/openeqa_hm3d)
- **Code:** [benchmark construction](benchmark), [evaluation](evaluation), and [model adapters](adapters)
- **Paper:** [IROS 2026 camera-ready manuscript](iros26_camera_ready_beyond_episodic_1.pdf)
- **Reproducibility:** [model configurations and pinned upstream revisions](configs)

## Overview

Embodied question answering is typically evaluated **episodically**: the agent answers one question, then its internal state is cleared. This setup does not reflect continuous deployment, where a robot should accumulate scene knowledge and reuse it across tasks.

**Sequential-EQA** converts an existing episodic benchmark into a continuous, multi-query evaluation. Questions from the same scene are presented in sequence, the environment and model weights remain fixed, and only designated scene memory persists between questions. Paired episodic and sequential runs reveal whether an architecture truly accumulates reusable knowledge—or merely keeps state around.

<p align="center">
  <img src="assets/readme/episodic-vs-sequential.png" width="100%" alt="Comparison of episodic and sequential embodied question answering evaluation">
</p>

<p align="center"><em>Episodic evaluation resets memory after every question; sequential evaluation preserves scene memory for later queries.</em></p>

### Key findings

**1. Persistence is not accumulation.** Simply retaining internal state does not reliably improve answer quality. Occupancy maps remember where an agent traveled but not necessarily what it saw, while short-horizon latent histories can become out-of-distribution across long, multi-query sequences.

**2. Spatial grounding breaks the accuracy-efficiency tradeoff.** Among the evaluated architectures, 3D-Mem is the only method that substantially improves both answer accuracy and navigation efficiency, achieving **+33.3% memory advantage** and **+53.3% step advantage**.

**3. The bottleneck extends beyond simulation.** Experiments on a Unitree Go2 across indoor and outdoor environments confirm that structured, spatially grounded visual memory is important for continuous operation under sensing and actuation noise.

## Main Results

<p align="center">
  <img src="assets/readme/main-results.png" width="100%" alt="Accuracy and navigation efficiency results for four embodied question answering agents">
</p>

Performance on Sequential-EQA. `SR` is episodic success rate, `SRmem` is sequential success rate, `MA` is memory advantage, `PL` is path length, and `SA` is step advantage.

| Method | SR ↑ | SRmem ↑ | MA ↑ | PL ↓ | PLmem ↓ | SA ↑ |
|:--|--:|--:|--:|--:|--:|--:|
| ExploreEQA | 43.8 | 46.5 | +2.7 | 84.3 | 84.3 | 0.0 |
| MemoryEQA | 61.0 | 62.4 | +1.4 | 43.6 | 43.9 | -0.5 |
| **3D-Mem** | 25.5 | **58.8** | **+33.3** | 5.6 | **2.6** | **+53.3** |
| UniNavid | 36.4 | 37.3 | +0.9 | 12.2 | 12.4 | -1.5 |

## Benchmark

The released benchmark is derived from the OpenEQA HM3D split and contains **50 scenes** and **498 questions** in reproducible scene-level sequences. Episodic and sequential evaluations use the same questions, order, initial poses, environments, and frozen model weights; only memory persistence changes.

The canonical files are located in [`benchmark/sequence_files/openeqa_hm3d/`](benchmark/sequence_files/openeqa_hm3d). See the [benchmark documentation](benchmark/README.md) for the schema, validation rules, and sequence-building options.

## Supported Agents

| Agent | Paradigm | Persistent memory | Retrieval |
|:--|:--|:--|:--|
| [ExploreEQA](https://github.com/Stanford-ILIAD/explore-eqa) | VLM agent | Occupancy and explored-space map | VLM frontier ranking |
| [MemoryEQA](https://github.com/memory-eqa/MemoryEQA) | VLM agent | RGB, poses, descriptions, and embeddings | Entropy-based adaptive retrieval |
| [3D-Mem](https://github.com/UMass-Embodied-AGI/3D-Mem) | VLM agent | 3D reconstruction and spatial visual features | Spatially indexed 3D lookup |
| [Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) | VLA | Frame history and latent visual context | Implicit attention |

The common lifecycle and the exact state retained or reset for each agent are documented in [`adapters/README.md`](adapters/README.md).

## Quick Start

### Environment preparation

Clone the repository and install the lightweight orchestration dependencies:

```bash
git clone <SEQUENTIAL_EQA_REPOSITORY_URL>
cd sequential-eqa
python -m pip install -r requirements.txt
```

Set the common HM3D paths:

```bash
export HM3D_DATA_ROOT=/path/to/hm3d-val
export HM3D_INITIAL_POSES=/path/to/scene_init_poses.csv
```

Install one or more upstream agents. Each agent remains in its own environment because their CUDA, Habitat, and model dependencies conflict.

```bash
bash scripts/setup_exploreeqa.sh
bash scripts/setup_memoryeqa.sh
bash scripts/setup_3d_mem.sh
bash scripts/setup_uninavid.sh
```

Exact inspected revisions are recorded in [`configs/upstreams.yaml`](configs/upstreams.yaml). Complete any checkpoint and API-key setup required by the corresponding upstream project.

### Evaluation

Start with one scene to validate the simulator, data paths, and GPU dependencies:

```bash
scripts/run_episodic.sh exploreeqa \
  --output-dir outputs \
  --scene 00801-HaxA7YrQdEC

scripts/run_sequential.sh exploreeqa \
  --output-dir outputs \
  --scene 00801-HaxA7YrQdEC
```

Multiple model names may be supplied before the options. Add `--dry-run` to inspect the isolated Conda commands without launching inference. Each completed run writes the native model artifacts and a canonical `results.jsonl`.

### Grading and metrics

Install `transformers>=4.56`, `torch`, and `accelerate` in the grading environment, then run:

```bash
python evaluation/grade_answers.py \
  --input outputs/episodic/results.jsonl \
  --output outputs/episodic/graded.jsonl

python evaluation/grade_answers.py \
  --input outputs/sequential/results.jsonl \
  --output outputs/sequential/graded.jsonl

python evaluation/compute_metrics.py \
  --episodic outputs/episodic/graded.jsonl \
  --sequential outputs/sequential/graded.jsonl \
  --output-dir outputs/metrics
```

The evaluator reports normalized OpenEQA score, binary success, memory advantage, and path/time/step advantages, including scene-level and question-position breakdowns. See [`evaluation/README.md`](evaluation/README.md) for metric definitions.

## Repository Structure

```text
benchmark/    Benchmark construction and released OpenEQA-HM3D sequences
evaluation/   Local answer grading and paired benchmark metrics
adapters/     Common adapter contract and four model integrations
scripts/      Setup and episodic/sequential launch commands
configs/      Reproducible model settings and upstream revision lock
external/     Ignored location for third-party repositories
```

Generated outputs, HM3D assets, checkpoints, credentials, and third-party repositories are excluded from version control.

## Acknowledgements

This project builds on [OpenEQA](https://open-eqa.github.io/), [Habitat](https://aihabitat.org/), and the four agent implementations listed above. We thank their authors for making their benchmarks and code available to the research community.

## Citation

If you find Sequential-EQA useful, please cite:

```bibtex
@inproceedings{cai2026beyond,
  title     = {Beyond Episodic Evaluation: Memory Architectural Bottlenecks
               in Sequential Embodied Question Answering},
  author    = {Cai, Zikui and Janga, Kaushal and Dao, Tan Dat and Lee, Seungjae
               and Dass, Shivin and Seo, Mingyo and Yue, Kaiyu and Kang, Mintong
               and Pillai, Nandhu and Hoover, Monte and Palnitkar, Aadi
               and Rawal, Ruchit and Zheng, Ruijie and Li, Bo and Zhu, Yuke
               and Mart{\'i}n-Mart{\'i}n, Roberto and Goldstein, Tom
               and Huang, Furong},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots
               and Systems (IROS)},
  year      = {2026}
}
```
