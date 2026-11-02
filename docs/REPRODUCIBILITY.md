# Reproducibility and environment

## What is preserved

The primary Python implementation, eleven notebooks (cell outputs removed), final-run and sweep configuration files, reports, threshold sweeps, validation summaries, and other aggregate model metrics. Historical absolute Colab paths and model IDs are preserved in original configs for research provenance; paths have **not** been silently rewritten to pretend out-of-the-box reproducibility.

## What still needs to be provided

- An approved local copy of the annotated train/validation/test sets in the expected original or mapped schemas.
- Compatible Python/package versions; GLiNER experiments originally referenced `gliner==0.2.29` and Transformers 4.51.3.
- A suitable GPU and/or Colab environment for full fine-tuning.
- Model checkpoints: excluded from the lightweight public snapshot; training code can create new checkpoints when the data is provided.

## Commands

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/gliner/train_gliner.py --config configs/gliner/finetune_run_1.json --check-config
```

The final command requires editing `train_data`, `val_data` and `output_dir` in the config (paths copied from Google Colab). `--check-config` validates the source files; it does not download weights or train.

GLiNER evaluation uses exact inclusive token boundaries and exact label identity. Calibration selects thresholds on validation, not the held-out test. Option B's added candidate labels are unverified, so these results are exploratory. ModernBERT, DeBERTa and OpenMed experiments have separate dependencies and may not work with the GLiNER-only requirements file.

## Notebook notes

The `.ipynb` files are valid Jupyter notebooks after removing stored execution outputs and execution counts; only outputs and execution counts were cleared. Some notebook cells reference their original Drive paths and require adaptation. Clearing outputs avoids committing verbose logs, embedded result data and potentially sensitive text.

## Limitations of this archive

The snapshot was assembled from selected Drive source code, reports, configurations and summary artifacts. It is *not* a byte-for-byte reconstruction of all project history. Nothing in this archive indicates that the experiments were rerun or revalidated by the packaging process.
