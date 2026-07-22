# Evaluation

Raw adapter output is normalized to JSONL before grading. Required fields are `model`, `mode`, `scene_id`, `question_id`, `question_index`, `question`, `reference_answer`, and `prediction`; navigation statistics are nullable.

```bash
python evaluation/grade_answers.py --input outputs/raw.jsonl --output outputs/graded.jsonl
python evaluation/compute_metrics.py \
  --episodic outputs/episodic.jsonl \
  --sequential outputs/sequential.jsonl \
  --output-dir outputs/metrics
```

The local grader uses a configurable Qwen model and a 1–5 rubric. It retries malformed responses once and never silently substitutes a score. The primary Open-EQA score is `mean((score - 1) / 4) * 100`; binary success (`score >= 4` by default), memory advantage, path advantage, time advantage, and step advantage are also reported. Inputs are paired strictly by `question_id`.

