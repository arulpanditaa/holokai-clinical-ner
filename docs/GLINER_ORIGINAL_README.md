# Using finetune

This directory contains GLiNER data, configs, scripts, the Colab notebook, and run artifacts. Run the commands below from the project root. New run paths are examples; create the corresponding config before using them.

## 1. Directory and data

```text
finetune/
├── README.md
├── Holokai_Finetune_Run_1.ipynb    # Complete Colab workflow
├── requirements.txt              # Local dependencies
├── requirements-colab.txt        # Colab dependencies; keeps runtime-provided PyTorch
├── .gitignore                    # Ignores virtual environments, caches, and outputs
├── data/                         # Only mapped files are listed here
│   ├── train_gliner_mapping.json
│   ├── val_gliner_mapping.json
│   └── corpus_V4_gliner_mapping.json
├── configs/
│   ├── train_gliner.json
│   ├── finetune_run_1.json
│   ├── smoke_test_mapping.json
│   └── smoke_test.json
├── scripts/
│   ├── train_gliner.py
│   ├── evaluate_gliner.py
│   ├── tune_per_label_thresholds.py
│   ├── map_recommended_labels.py
│   ├── validate_gliner_data.py
│   └── label_schemas.py
└── outputs/
    └── finetune_run_1/            # Detailed layout in Section 5
```

`.gitkeep` preserves empty directories. `.DS_Store` and `__pycache__/` require no action.

| File to place in `data/` | Purpose |
| --- | --- |
| `train_gliner_mapping.json` | Formal training input |
| `val_gliner_mapping.json` | Input for baseline evaluation, training validation, model selection, and threshold tuning |
| `corpus_V4_gliner_mapping.json` | Mapped corpus; not directly used for training or validation by the current notebook |

Training requires only the first two files, with `label_schema: recommended`. Each dataset is a JSON array of records containing tokens and entity annotations:

```json
[
  {
    "tokenized_text": ["Patient", "takes", "aspirin", "."],
    "ner": [[2, 2, "medication, drug, or prescription"]]
  }
]
```

Each `ner` entry is `[start_token, end_token, mapped_label]`. Indices are zero-based and the end token is **inclusive**; records without entities use `[]`. Mapping changes label names only, preserving tokens and spans. See `scripts/label_schemas.py` for the 11 recommended labels and their mappings.

## 2. Environment and workflow

For local use, choose a stable Python 3.10 or 3.11 environment and install:

```bash
python -m pip install -r finetune/requirements.txt
python -m pip check
```

In Colab, select a GPU runtime, mount Drive, change to the project root, and install `finetune/requirements-colab.txt`. Restart the runtime if relevant packages were already imported before installation. Training supports one process and one selected GPU; `device: auto` uses CUDA when available, otherwise CPU.

Run the notebook in this order:

**Environment and paths → mapped data → config check → baseline → smoke test → full training → model evaluation and selection → comparison → threshold tuning → run summary.**

The smoke test uses the first 100 mapped training records and first 30 validation records. Model selection, comparison files, and the run summary are produced by notebook cells, not separate scripts.

If mapped data already exists, skip the mapping-generation command in Step 2: the script refuses to overwrite it. If you have only mapped files, also skip the two `validate_mapping(...)` calls that require original files. Keep the cell's imports, paths, label variables, and `load_json` definition for later steps. Still run `--check-config` before training to check data structure, labels, and overlapping train/validation token sequences.

## 3. Starting a new experiment and changing training paths

### How to use the configs

| File in `configs/` | Usage |
| --- | --- |
| `train_gliner.json` | Base config read by the current notebook. Uses relative paths and defaults to `outputs/full_finetune_run1`, which differs from the notebook's Run 1 directory |
| `finetune_run_1.json` | Run 1 config. Copy its complete field structure for a new run; its existing paths are absolute Colab paths |
| `smoke_test_mapping.json` | Notebook-generated mapped-data smoke config, using absolute Colab paths |
| `smoke_test.json` | Older smoke config; defaults to the original schema and references input paths that are currently absent. Do not use for new experiments |

To change hyperparameters, copy the Run 1 config to a new file such as `configs/finetune_run_2.json`. Preserve the complete field structure and edit settings such as `learning_rate`, `batch_size`, and `epochs` in that file. `eval_steps` controls validation frequency, `logging_steps` controls logging frequency, and `threshold` controls validation prediction filtering.

Also update the new config's paths. These are replacement fields, **not a complete config**:

```json
{
  "label_schema": "recommended",
  "train_data": "../data/train_gliner_mapping.json",
  "val_data": "../data/val_gliner_mapping.json",
  "output_dir": "../outputs/finetune_run_2/training"
}
```

Relative config paths resolve against the **config file's directory**; command-line paths resolve against the current working directory. Use a fresh output directory and preserve earlier runs.

### Running training directly

Point `--config` to the new file in both commands:

```bash
python finetune/scripts/train_gliner.py --config finetune/configs/finetune_run_2.json --check-config
python finetune/scripts/train_gliner.py --config finetune/configs/finetune_run_2.json
```

`--check-config` does not load a model, train, or write output files. It also rejects a destination that already contains run artifacts. Training exports models at completed epochs and creates `final_model/`; resuming optimizer/scheduler state is not supported.

### Running a new experiment through the notebook

Copy the notebook for the new experiment, then update these locations:

| Notebook location | Required changes |
| --- | --- |
| Step 1 | Update the Drive working directory in `%cd` |
| Step 2 | Update `PROJECT` and, if needed, `TRAIN_DATA` / `VAL_DATA`. Set `RUN_ROOT` to `PROJECT / "finetune/outputs/finetune_run_2"` |
| Step 3 | Read the new config you prepared. Replace the block that reads `BASE_CONFIG`, asserts default hyperparameters, and writes `RUN_CONFIG`, so it does not overwrite your settings |
| Steps 3 and 6 | Change `--config` in the check/training commands to the new path, or use `--config "{RUN_CONFIG}"` consistently |
| Step 5 | Give `SMOKE_CONFIG` a unique name, such as `smoke_test_mapping_run_2.json`. Change both smoke commands to `--config "{SMOKE_CONFIG}"` |
| Step 10 | Change the summary's `"run"` value to `"finetune_run_2"` |

Replace the Step 3 config-generation cell with the following. It reads the new config and resolves relative paths without rewriting the file:

```python
import sys
sys.path.insert(0, str(PROJECT / "finetune/scripts"))
from train_gliner import load_config

RUN_CONFIG = PROJECT / "finetune/configs/finetune_run_2.json"
cfg = load_config(RUN_CONFIG)
assert cfg["label_schema"] == "recommended"
assert Path(cfg["train_data"]) == TRAIN_DATA.resolve()
assert Path(cfg["val_data"]) == VAL_DATA.resolve()
assert Path(cfg["output_dir"]) == TRAIN_OUT.resolve()
print(json.dumps(cfg, indent=2))
```

`TRAIN_OUT`, `BASELINE_OUT`, `MODEL_EVAL_OUT`, `COMPARISON_OUT`, and `THRESHOLD_OUT` derive from `RUN_ROOT`. Re-execute their assignments after changing it; there is no need to edit each destination separately. The config's `output_dir` must match `TRAIN_OUT`.

Keep threshold 0.4 for comparable standard runs. If deliberately changing it, also update the notebook's fixed values for baseline evaluation, model evaluation, smoke testing, and summary generation; changing the config alone is insufficient.

## 4. After training and script quick reference

**In the same notebook session, with files still in place, no further path changes are normally needed.** Step 7 finds models under `TRAIN_OUT` and sets `BEST_MODEL_PATH`; subsequent evaluation, tuning, and summaries use the current run's variables.

When running scripts separately, set `--model` to the new run's model directory and `--output-dir` to a separate results directory. These examples use `final_model`; replace it with the selected `epoch-N` when following notebook model selection:

```bash
python finetune/scripts/evaluate_gliner.py --label-schema recommended \
  --model finetune/outputs/finetune_run_2/training/final_model \
  --data finetune/data/val_gliner_mapping.json --threshold 0.4 \
  --output-dir finetune/outputs/finetune_run_2/model_eval/final_model

python finetune/scripts/tune_per_label_thresholds.py --label-schema recommended \
  --model finetune/outputs/finetune_run_2/training/final_model \
  --data finetune/data/val_gliner_mapping.json \
  --output-dir finetune/outputs/finetune_run_2/threshold_tuning
```

For the baseline, use the same evaluator with `--model knowledgator/gliner-pii-base-v1.0` and the run's `baseline/` as the output directory. Evaluation and tuning default to the original schema, so explicitly select `recommended` for mapped data. Tuning currently requires 171 validation records covering all 11 labels; its results are validation-calibrated performance.

After restarting the notebook, restore the run's paths/config and load existing metrics to recover the variables needed by later cells. Do not rerun all cells from the beginning just to restore state. If copying Colab files locally, adjust `PROJECT` and the config or CLI paths in use. Historical Colab paths in saved outputs do not update automatically and need not be rewritten to run a new evaluation.

| Script / entry point | Inputs and purpose | Outputs |
| --- | --- | --- |
| `train_gliner.py` | Config + mapped train/validation data; check or train | Config, manifest, history, and models under `training/`; commands in Section 3 |
| `evaluate_gliner.py` | Hub model ID or complete model directory + data + threshold | `metrics.json`, `per_label_metrics.csv`, `predictions.json` |
| `tune_per_label_thresholds.py` | Selected model + full validation data | Threshold sweep, selected thresholds, and calibrated validation metrics |
| `map_recommended_labels.py` | Optional preparation tool; reads `train_gliner.json`, `val_gliner.json`, and `corpus_V4_gliner.json` | Three `_mapping.json` files and console checks; unnecessary when mapped files already exist |
| `validate_gliner_data.py` | Optional source-data check; requires `train.json`, `val.json`, and corresponding mapped files | JSON validation report and PASS/FAIL; does not generate or convert data |
| `label_schemas.py` | Shared label definitions and model/schema compatibility checks | No standalone CLI output |
| Notebook Steps 5 / 7 / 8 / 10 | Smoke test / model selection / comparison / summary | Files within the corresponding run |

Optional preparation commands follow; do not run them with only mapped files available. The mapping destination must not already contain files with the same mapped names. Validation can replace an existing report, so use a separate report path:

```bash
python finetune/scripts/map_recommended_labels.py --data-dir finetune/data \
  --output-dir finetune/outputs/finetune_run_2/mapped_data
python finetune/scripts/validate_gliner_data.py --label-schema recommended \
  --data-dir finetune/data --report finetune/outputs/finetune_run_2/data_validation.json
```

Training rejects existing run artifacts; evaluation and tuning reject existing result files with the same names. However, notebook writes to configs, smoke subsets, comparisons, and summaries can overwrite files. Use both a new config name and a new `RUN_ROOT` for each experiment.

## 5. Output files

The current layout is shown below, with repeated contents of `epoch-1/`, `epoch-2/`, `epoch-3/`, and `final_model/` grouped together:

```text
outputs/finetune_run_1/
├── baseline/                     # metrics.json, per_label_metrics.csv, predictions.json
├── smoke_test/
│   ├── data/                     # Mapped train/validation subsets
│   └── run/                      # Training records + epoch-1/, final_model/
├── training/                     # Training records + epoch-1/, epoch-2/, epoch-3/, final_model/
├── model_eval/                   # Evaluation files for each epoch and final_model
├── comparison/
│   ├── pretrained_vs_finetuned_overall.json
│   └── pretrained_vs_finetuned_per_label.csv
├── threshold_tuning/
│   ├── threshold_sweep.csv
│   ├── best_thresholds.json
│   ├── calibrated_metrics.json
│   └── calibrated_per_label_metrics.csv
├── best_model.json
├── run_summary.json
└── HoloKai_GLiNER_Fine-tuning_Report_for_Run_1.docx
```

| File | Contents / when to use it |
| --- | --- |
| `metrics.json` / `per_label_metrics.csv` | Overall and per-label P/R/F1, TP/FP/FN, and support; JSON also records provenance and settings. Matching requires exact token spans and labels within a sample; overall metrics are micro-averaged |
| `predictions.json` | Sample index, tokens, `gold_ner`, and `predicted_entities` with start/end/label/score; use for inspecting prediction errors |
| `training_config.json` / `data_manifest.json` | Resolved training config; dataset hashes, sample counts, and schema |
| `training_history.json` | Training and validation logs by step/epoch |
| `train_loss.json` / `validation_metrics.json` | Loss entries / exact-match validation entries extracted from history |
| Model directories | `gliner_config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, and `spm.model`. Complete exports can be loaded by the evaluator; they do not contain training-resume state |
| `threshold_sweep.csv` / `best_thresholds.json` | Per-label candidate-threshold metrics; selected thresholds and selection rules. Grid: 0.10–0.90 in steps of 0.05, with provider recall constrained to ≥ 0.70 |
| `calibrated_metrics.json` / `calibrated_per_label_metrics.csv` | Overall/per-label validation metrics after per-label threshold filtering; not independent test metrics |
| `best_model.json` / comparison files | Selected model location and selection rule; fixed-threshold baseline/selected-model metrics and differences |
| `run_summary.json` / DOCX | Notebook-generated run settings and results; human-readable report. Current scripts do not generate the DOCX |

Currently, `training/epoch-1/`, `training/epoch-2/`, and `smoke_test/run/final_model/` lack weight files. These directories and their saved evaluation results alone are insufficient to reload a model. `.gitignore` excludes `outputs/*` except `.gitkeep`; preserve complete artifacts separately when sharing runs. The `scripts/check_environment.py` and `tests/` paths mentioned in the previous README are currently absent.
