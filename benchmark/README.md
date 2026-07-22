# Benchmark construction

The released Open-EQA HM3D split is in `sequence_files/openeqa_hm3d/`. It contains 50 scene-level sequences and 498 questions. Each question has the stable identifier `<scene_id>:qNNN`, where the index follows the released row order.

The files are the canonical artifact used by this repository. The original source-dataset revision and sequence-generation seed were not preserved in the working research directory, so the manifest records those fields as unavailable rather than guessing them.

To convert another episodic CSV collection into scene sequences:

```bash
python benchmark/build_sequences.py \
  --input /path/to/csv-or-directory \
  --output /tmp/sequences \
  --minimum-sequence-length 1
```

Input CSVs require `scene`, `question`, and `answer`; `floor` is optional and defaults to `0`. Source row order is preserved unless both `--shuffle` and `--seed` are supplied. The builder removes exact duplicates by default; pass `--keep-duplicates` when reproducing an artifact in which repeated questions are intentional. Invalid rows are always rejected.
