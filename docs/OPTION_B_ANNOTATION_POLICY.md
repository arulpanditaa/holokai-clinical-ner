# Holokai GLiNER — Option B annotation policy (proposed)

## Scope
Retain all existing 11 label names, original tokenization, and original annotations. Expand the clinical scope of three existing labels. All spans use inclusive token offsets.

### `lab result or test finding` (LAB_RESULT)
Include reported blood-pressure readings (e.g., BP 126/74), heart rate, respiratory rate, SpO₂, temperature, and other explicitly measured vital-sign values. Annotate the clinical measurement and value together when adjacent; do not tag dates, employee IDs, or unrelated numerical ratios. A qualitative observation without a measurement needs manual adjudication.

### `medical procedure or test performed` (PROCEDURE)
Include named physical/neurological examinations, structured clinical assessments, and named clinical counseling or patient education services. Differentiate actual, recommended, planned, and hypothetical services consistently; these automated files primarily target explicit named phrases and cannot fully resolve status.

### `condition` (DIAGNOSIS)
Include explicitly documented patient symptoms and signs (e.g., chest pain, hives, hypoxia, wheezing, seizures), whether or not a formal disease diagnosis is present. Avoid labeling unrelated uses of common words, negated findings, future warning signs, and hypothetical scenarios without review.

## Caveats and approval
These are machine-generated candidate updates, NOT adjudicated ground truth. The rules are pattern-based, do not exhaust all possible paraphrases, and may create incorrect spans. All newly added entities are marked AUTOMATED_UNVERIFIED in the CSV audit. Human review is necessary before retraining or publishing new metrics. Do not tune against V4; use V4 only after policy and model selection are frozen. Retain original test metrics for comparison.

## Files
`train_gliner_mapping_option_B.json`, `val_gliner_mapping_option_B.json`, `corpus_V4_gliner_mapping_option_B.json`: original records plus proposed additions.
`option_B_additions_audit.csv`: traceable new entities and context.
`option_B_summary.json`: counts and hashes of original sources.
