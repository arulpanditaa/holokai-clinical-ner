# Experiments and findings

This note consolidates the included recorded results; it is not a fresh experiment or independent verification of the original training runs.

## GLiNER original final run

- Base: `knowledgator/gliner-pii-base-v1.0`.
- Original 11-entity mapped schema; 1,509 train records, 171 validation records, and 648 held-out V4 records.
- Original entity counts: 3,947 training, 453 validation, 1,320 V4.
- Final selected configuration: learning rate `0.00012`, batch 16, eval batch 8, 8 epochs, 0.1 warmup, 0.1 weight decay, seed 42, threshold 0.4.
- Validation micro F1: 87.46% at fixed threshold and 89.44% after tuning 11 per-label thresholds on validation.
- Held-out original V4 micro F1: 88.25% fixed / 88.30% calibrated.
- Limitation: procedure and condition recognition were among the weaker labels, and validation calibration provided only marginal held-out benefit.

Supporting files: `reports/Holokai_GLiNER_Final_Results_Report.docx`, `results/gliner_original/`, `configs/gliner/finetune_final_run.json`.

## Option B annotation expansion

- Automated candidate additions increased entity counts by 631 train / 57 validation / 93 V4 to 4,578 / 510 / 1,413.
- Fixed-threshold (0.4) validation micro F1: 87.92%; tuned validation: 90.24%.
- Revised V4 fixed-threshold micro F1: 87.74%; calibrated revised V4: 88.61%.
- **Caveat:** Option B candidate labels are automated and unverified; revised V4 is a different target from original V4. Don't interpret a cross-version score difference as a pure architecture or training gain.

Supporting files: `reports/Holokai_Option_B_Training_Evaluation_Report.docx`, `docs/OPTION_B_ANNOTATION_POLICY.md`, `results/gliner_option_b/`.

## OpenMed DiseaseDetect 184M

- Zero-shot output schema did not cover all 11 target types. Fine-tuning replaced the disease-only head with a 23-class BIO head and updated the encoder.
- Best selected learning rate: `5e-5` after eight epochs (seed 42), based on validation F1 90.68%.
- Held-out **original V4** micro F1: **87.38%**.
- Strong on some clinical entity types but weaker than original GLiNER overall, particularly for PROCEDURE. Original V4 GLiNER comparison: 88.25% at fixed threshold.

Supporting files: `src/openmed/disease/`, `results/openmed_disease/`, `reports/OpenMed_DiseaseDetect_ZeroShot_and_FineTuning_Report.docx`.

## ModernBERT and OpenMed baseline investigations

- ModernBERT Large frozen-encoder baseline validation micro F1 was about 46.98%; BioClinical variant about 57.27%, per the project ModernBERT report. These are **validation** figures, not directly comparable to the held-out V4 GLiNER figures.
- OpenMed privacy-filter baseline, PharmaDetect, and DeBERTa-GlobalPointer were also evaluated. The repository contains code, notebooks, model comparisons and aggregate metric artifacts where retrieved.

Supporting files: `src/modernbert/`, `src/openmed/pharma/`, `results/modernbert_*`, `results/openmed_*`, `results/deberta/`, and related `reports/` files.

## Hyperparameter sensitivity

The five-seed comparison recorded validation F1 of approximately 90.68%, 88.98%, 87.46%, 87.04% and 89.19%, with a mean about 88.67% and standard deviation about 1.46 percentage points. **These values include mixed execution environments**, so the variation cannot be attributed to seed alone. Separate same-environment weight-decay studies favored 0.10 among the recorded candidates. See `results/sensitivity/` and the seed/weight-decay report for provenance.

## Weak-label analysis

Error-analysis work highlighted inconsistent span boundaries, class ambiguity (especially medications and procedures), and distribution differences between validation and V4. The project assessed possible annotation adjustments; their outputs are not automatically verified human ground truth. See `src/analysis/`, associated notebooks, and the weak-label reports.
