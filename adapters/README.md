# Model adapter contract

The outer runner launches one model-side worker per scene in the model's own Conda environment. The worker adds the freshly cloned repository to `sys.path`, imports its Python modules, and drives the imported model objects through the common lifecycle. This avoids copying upstream code and prevents incompatible CUDA, Habitat, and model packages from sharing one process.

Every integration follows the same lifecycle:

1. Load model weights once for the scene process.
2. Clear all state when entering a new scene.
3. Restore the configured initial simulator pose and clear question-specific targets, action queues, prompts, and planner state before each question.
4. In episodic mode, also clear every map, reconstruction, visual history, embedding, and language memory before each question.
5. In sequential mode, retain only the model-specific scene memory listed below.
6. Write predictions and navigation statistics; the outer adapter normalizes them to `results.jsonl`.

| Adapter | Retained between questions | Always reset between questions |
| --- | --- | --- |
| ExploreEQA | Occupancy/semantic geometry and explored-space memory | Frontier target, stopping state, question prompt, initial pose |
| MemoryEQA | Images, poses, descriptions, embeddings, and hierarchical retrieval memory | Active query, planner target, stopping state, initial pose |
| 3D-Mem | 3D reconstruction, spatial visual features, and scene snapshots | Query retrieval state, navigation target, initial pose |
| Uni-NaVid | Frame history and permitted latent visual context | Pending actions, prompt/task tokens, initial pose |

`base.py` defines the canonical configuration, lifecycle, and result types. `worker.py` runs that lifecycle inside the selected Conda environment. Each model adapter imports the current upstream implementation directly:

- ExploreEQA imports `run_vlm_exp`, caches its VLM, and controls its `TSDFPlanner` factory.
- MemoryEQA imports `src.modeling.memory_eqa.MemoryEQA` and conditionally suppresses `StructuredMemory.clear()`.
- 3D-Mem imports `run_aeqa_evaluation`, then controls its `Scene` and `TSDFPlanner` factories.
- Uni-NaVid imports `offline_eval_uninavid.UniNaVid_Agent` and controls its online feature-cache reset.

The adapters do not edit upstream files. Setup scripts clone a missing repository or fast-forward an existing clone, so rerunning setup intentionally targets the current upstream default branch. The resolved commits used during adapter development are recorded in `configs/upstreams.yaml` for debugging compatibility changes.
