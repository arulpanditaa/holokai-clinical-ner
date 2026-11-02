# Holokai Clinical NER — HIPAA-Oriented Entity Recognition

**Johns Hopkins University · Industry Project Practicum · 2026**

A research and experimentation repository for clinical/privacy-oriented **named entity recognition (NER)**. The work investigates detecting unstructured personally identifying and clinical entities using fine-tuned **GLiNER**, alongside **ModernBERT**, **DeBERTa-GlobalPointer**, and **OpenMed** model baselines. The broader practicum considered efficient, on-device detection for regulatory/privacy workflows.

> **Scope:** This is a research artifact and experiment archive, **not** a HIPAA-certified product or a validated clinical de-identification service. The underlying project was collaborative; results and code are presented as team research, not claimed as one person's work.

## Highlights

| Experiment | Evaluation | Micro F1 | Interpretation |
| --- | --- | ---: | --- |
| Fine-tuned GLiNER, original annotations, fixed threshold 0.40 | Held-out V4 (648 records; 1,320 gold entities) | **88.25%** | Primary reference result |
| Same GLiNER, validation-selected thresholds | Same original V4 | **88.30%** | Marginal test-set gain after calibration |
| GLiNER Option B, fixed threshold 0.40 | Revised V4 annotations (648 records; 1,413 gold entities) | **87.74%** | Different gold annotations; not a same-test comparison |
| GLiNER Option B, calibrated | Same revised Option B V4 | **88.61%** | Option B candidate additions remain automated/unverified |
| Fine-tuned OpenMed DiseaseDetect 184M | Original V4 (648 records; 1,320 gold entities) | **87.38%** | Different model architecture, original test annotations |

**Validation performance:** original GLiNER achieved **87.46%** F1 with the fixed 0.40 threshold and **89.44%** with validation-fitted per-label thresholds. These validation figures **must not** be confused with held-out V4 results. The best final-run configuration used learning rate `1.2e-4`, batch size `16`, eight epochs, weight decay `0.1`, seed `42`, and the `knowledgator/gliner-pii-base-v1.0` starting checkpoint.

The ModernBERT experiments and OpenMed alternatives are documented separately in `reports/` and `results/`. Differences between the original and Option B annotations are changes in the evaluation target, not clean estimates of model improvement.

## The 11 modeled entities

`DIAGNOSIS`, `FACILITY`, `GENDER_IDENTITY`, `LAB_RESULT`, `LOCATION_ADDRESS`, `MEDICATION`, `PERSON`, `PROCEDURE`, `PROVIDER`, `SEX`, and `UNIQUE_ID`.

The project focuses on entities that require contextual NER rather than simple deterministic pattern matching. The full original-to-recommended prompt mapping is recorded in [`src/gliner/label_schemas.py`](src/gliner/label_schemas.py).

## Repository layout

```text
.
├── src/
│   ├── gliner/             # Dataset validation, training, evaluation, threshold tuning
│   ├── modernbert/         # Frozen encoder baselines and ONNX export research
│   ├── openmed/
│   │   ├── disease/        # DiseaseDetect preparation, fine-tuning and evaluation
│   │   └── pharma/         # PharmaDetect preparation, fine-tuning and evaluation
│   ├── analysis/           # Weak-label error analysis and V4 annotation audit
│   └── legacy/             # Earlier exploratory scripts
├── notebooks/              # Experiment notebooks, with cell outputs cleared
├── configs/
│   ├── gliner/            # Final-run, LR sweeps, seed/weight-decay configs
│   └── openmed/           # DiseaseDetect and PharmaDetect configs
├── results/                # Aggregate metrics, threshold sweeps and comparison plots
├── reports/                # Experiment reports and selected comparison artifacts
├── docs/                   # Experiment interpretation and data/reproduction notes
├── data/                   # Only a small entirely fictional demo fixture
├── tests/                  # Lightweight tests without downloading models
└── requirements.txt
```

## Quick start

The original GLiNER implementation was developed for Google Colab and has environment-specific paths in historical configs. **Change those paths before reproducing training.**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For the original GLiNER dataset format, each record contains a list of `tokenized_text` tokens and `ner` annotations of form `[start_token, end_token, label]`; indices are zero-based and the end token is **inclusive**. A tiny **fictional** example lives at `data/demo_gliner.json`. It is illustrative, not a training benchmark.

After placing separately approved training/validation data in a private local directory and updating the desired JSON config:

```bash
# Validate config and data before training; does not load a model.
python src/gliner/train_gliner.py --config configs/gliner/finetune_run_1.json --check-config

# Full training is GPU-intensive; run only with the intended dataset and paths.
python src/gliner/train_gliner.py --config configs/gliner/finetune_run_1.json
```

For Colab-first instructions and the original file layout, consult [`docs/GLINER_ORIGINAL_README.md`](docs/GLINER_ORIGINAL_README.md). Other architectures have their own scripts and notebooks and may require extra dependencies beyond the GLiNER environment.

## Methods and evaluation notes

- **Exact entity matching:** correct label and exact inclusive token boundaries per example; micro precision/recall/F1; no partial-credit spans.
- **Validation calibration:** threshold values are chosen on validation data and frozen before test evaluation; a validation F1 improvement is not guaranteed to transfer.
- **Model sensitivity:** random-seed and weight-decay studies are archived in `results/sensitivity/`. Some seed scores came from mixed environments and should not be presented as fully controlled replications.
- **Option B:** automated candidate annotations were added to training, validation and V4; they are **unverified**. Its V4 metrics are evaluated against revised gold labels.
- **Deployment:** an inference result is not proof of deployability, privacy compliance, or real-time laptop inference performance. ONNX work is exploratory.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data and model checkpoints

Raw clinical-like records, patient-like identifiers, per-record predictions/error dumps, and heavyweight model checkpoints are **not packaged** in this public-repository snapshot. The source Drive contains examples with patient-like names and record numbers; their synthetic/de-identified status was not established by this packaging process. This is a precaution about **public redistribution**, not a claim that the source Drive is confidential. Refer to [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md). The original sponsor presentation, labeled proprietary/confidential, is likewise not included.

## Provenance and project context

Based on research artifacts in the team's Holokai Industry Project Practicum Google Drive workspace (2026); retained output reports, notebooks and configs are in this repository. Final-figure statements are sourced to the corresponding included reports and aggregate metric JSONs. Notebook cell outputs were removed from this snapshot; underlying source cells are preserved. No model training was re-run during packaging.

**Licensing:** A software license is intentionally not assigned here because ownership and downstream redistribution rights for a sponsored collaborative practicum were not verified. Anyone interested in reuse should obtain appropriate permission.
