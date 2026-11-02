# OpenMed DiseaseDetect SuperClinical 184M: fine-tuning on all 11 HoloKai entities

Same method and code as the teammate's PharmaDetect-434M run (week9 on Drive), adapted to
`OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-184M` (revision `1d72b9a9bf3e755fef613337a7d6b17e1b7ef48f`).

- Keep the encoder, replace the 1-class head (O / B-DISEASE / I-DISEASE) with a 23-tag BIO head
  (O + B/I for the 11 entities), full fine-tuning.
- Data: original week6 `train_gliner_mapping.json` / `val_gliner_mapping.json` / `corpus_V4_gliner_mapping.json` (no Option B).
- Best epoch per group chosen by validation exact-match micro F1. V4 is scored once at the end,
  only for the validation-selected group (`final_v4_eval.py`).

## Learning rates

| Group | LR | Why |
|---|---|---|
| learning_rate_3e-5 | 3e-5 | Best of the PharmaDetect run (89.0 val F1); anchor |
| learning_rate_5e-5 | 5e-5 | PharmaDetect peaked at its highest LR; 184M is half the size of 434M and tolerates a higher LR |
| learning_rate_8e-5 | 8e-5 | Our GLiNER base models kept improving up to 8e-5–1.2e-4 at effective batch 16 |

Shared: 8 epochs (PharmaDetect's best was its last of 5 epochs; our best GLiNER used 8), effective batch 16
(batch 4 × accumulation 4), weight decay 0.01, warmup 0.1, linear schedule, max grad norm 1, seed 42,
gradient checkpointing, fp16 on T4.

## Run on Colab

1. Upload this whole folder to `MyDrive/Holokai_Project/week9/` (next to the PharmaDetect folder).
2. Open `finetune_colab.ipynb` in Colab, set **Runtime → T4 GPU**, run the cells top to bottom.
3. Results land in `outputs/`:
   - `outputs/<group>/validation_results.json`: best epoch and per-label validation metrics
   - `outputs/<group>/best_model/`: the best checkpoint (~740 MB each)
   - `outputs/final_v4_results.json`: the one-time V4 score of the selected model

## Files

| File | Role |
|---|---|
| `prepare_data.py` | GLiNER spans → BIO chunks with integrity checks (writes `data/`) |
| `evaluate.py` | Exact original-token entity scoring; never reads V4 |
| `train.py` | Full fine-tuning of the three groups, keeps each group's best epoch |
| `final_v4_eval.py` | One-time V4 evaluation of the validation-selected model |
| `configs/*.json` | The three learning-rate configurations |
| `finetune_colab.ipynb` | Colab driver |
