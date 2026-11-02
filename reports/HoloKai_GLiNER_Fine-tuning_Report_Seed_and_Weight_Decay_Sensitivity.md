# HoloKai GLiNER Fine-tuning Report

# Random-Seed and Weight-Decay Sensitivity Analysis

# 1\. Objective

Evaluate how random initialization/data-order seed and AdamW weight decay affect exact-match named-entity recognition performance of the optimized B22 HoloKai GLiNER fine-tuning configuration. This report consolidates the completed five-seed study and five-setting weight-decay sweep. The two studies answer different questions: how much the outcome changes with seed at WD=0.01, and which WD setting performed best with seed=42. These are validation results, not independent test results.

# 2\. Data, model and evaluation

Base model: knowledgator/gliner-pii-base-v1.0. Task: HoloKai medical/PII entity extraction with the 11 recommended GLiNER label names. Training split: 1,509 samples; mapped validation split: 171 samples containing 453 gold entities. Exact matching requires the predicted label and token-span boundaries to match the gold entity. The reported overall precision, recall and F1 are micro-aggregated across labels. All runs in this report use prediction threshold 0.4. No held-out V4 test results were produced for these new sensitivity experiments.

# 3\. Experimental configuration

Optimized B22 anchor: learning rate 0.00012; training batch size 16; 8 epochs; AdamW; weight decay 0.01 unless explicitly varied; random seed 42 unless explicitly varied; linear learning-rate scheduler; warmup ratio 0.1; gradient accumulation 1; Adam β1=0.9, β2=0.999; evaluation every 100 steps and at training completion. The best validation F1 across these scheduled evaluations is the primary comparison metric. Experiment configs and epoch metrics are stored in Google Drive under Holokai\_Project/finetune\_final/. The initial Run 1 used different hyperparameters and obtained 79.06% validation F1; this report uses the optimized B22 configuration as the comparison anchor, rather than varying only one factor from the original Run 1\. This departure should be identified when interpreting results against the assignment brief.

# 4\. Study A — random-seed sensitivity (WD=0.01)

Seed 42: best validation F1 90.6800% (original teammate B22 run).  
Seed 7: best validation F1 88.9831%.  
Seed 123: best validation F1 87.4598%.  
Seed 2026: best validation F1 87.0370%.  
Seed 99: best validation F1 89.1920%.

Five-seed mean: 88.6704%. Sample standard deviation: 1.4613 percentage points. Observed minimum: 87.0370%; maximum: 90.6800%; range: 3.6430 percentage points. The mean and standard deviation summarize observed results, not a confidence interval or proof of an expected population effect. Importantly, seed 42 came from a teammate's earlier environment while seeds 7, 123, 2026 and 99 were run in the current user's environment. Accordingly, this five-seed spread can reflect both seed and environment/run-to-run differences; it is not a fully controlled seed-only estimate. A combined five\_seed\_statistics.json exists, but individual per-seed backup artifacts are not complete for all five seeds.

# 5\. Study B — weight-decay sensitivity (seed=42)

WD 0.00: best validation F1 87.2917%; best epoch 7.37; final F1 86.6530%.  
WD 0.01: best validation F1 88.0753%; best epoch 7.37; final F1 87.3181%.  
WD 0.05: best validation F1 87.2651%; best epoch 7.37; final F1 86.5916%.  
WD 0.10: best validation F1 89.5075%; best epoch 7.37; final F1 88.7949%.  
WD 0.20: best validation F1 88.2845%; best epoch 8.00; final F1 88.2845%.

These five runs used the same stated seed, model, dataset, threshold, LR, batch size, and number of epochs, changing weight decay and run output directory. They were conducted in the current Colab environment. WD=0.10 was the highest-scoring tested setting, \+1.4322 percentage points relative to the same-environment WD=0.01 run. Increasing WD from 0.10 to 0.20 lowered best validation F1 by 1.2230 points in this seed-42 sweep. The pattern is not monotonic, and one seed per WD does not establish that 0.10 will outperform alternatives on average.

# 6\. Best observed weight-decay run — error counts

WD=0.10, seed=42, epoch approximately 7.37: precision 86.9023%, recall 92.2737%, F1 89.5075%; TP 418, FP 63, FN 35 (453 gold entities). The corresponding WD=0.01 replication at its best epoch had precision 83.6978%, recall 92.9360%, F1 88.0753%; TP 421, FP 82, FN 32\. Thus, the WD=0.10 run produced 19 fewer false positives but 3 more false negatives and 3 fewer true positives, with a higher micro F1. These are aggregate exact-match counts; no example-level error adjudication was performed here.

# 7\. Cross-study comparison and interpretation

The teammate's original B22 seed-42 result was 90.68%, while the current-environment WD=0.01 seed-42 replication was 88.08%, a difference of about 2.60 percentage points despite matching the main documented settings. Consequently, the original 90.68% figure must not be treated as an interchangeable controlled baseline for the weight-decay sweep. The fair within-environment comparison is 89.51% at WD=0.10 versus 88.08% at WD=0.01. The seed-study variation (1.46-point sample SD, with mixed environments) is comparable in magnitude to the observed 1.43-point WD improvement. Therefore, results support 'best observed WD at seed 42' rather than a statistically established regularization gain.

# 8\. Reproducibility and artifacts

Google Drive project root: Holokai\_Project/finetune\_final/  
weight\_decay\_experiments/: all 15 expected files verified (five settings × epoch\_metrics.csv, summary.json and config.json).  
configs/: five WD configuration JSON files and B22\_seed7.json verified.  
seed\_experiments/: five\_seed\_statistics.json and complete individual summary/metrics/config sets for seeds 123, 2026 and 99 verified. Seed 7's per-run summary and epoch metrics were not found in this folder; seed 42's individual seed-study files were also not found there. The original seed-42 run belongs to the teammate's earlier work and may be archived elsewhere.  
best\_models/: B22\_seed7\_best\_model.zip verified. No WD=0.10 best-model ZIP was verified in this folder; checkpoint availability elsewhere is not established.

Recorded current Colab package versions: GLiNER 0.2.29, transformers 4.51.3, PyTorch 2.11.0+cu130 and accelerate 1.15.0. The training output folders were kept in local /content/holokai\_runs/; Colab-local checkpoints are not durable after runtime reset unless separately backed up. Validation metrics and configs, however, are on Drive.

# 9\. Limitations

(1) The seed-study data mix two execution environments; treat its dispersion as descriptive. (2) Each WD setting was evaluated at only seed 42; there is no multi-seed WD effect estimate or statistical significance test. (3) Best-of-eight checkpoint selection uses the same validation set and can favor optimistic maxima. (4) Validation comprises only 171 samples and 453 gold entities; individual labels have substantially smaller support. (5) WD sweep was conducted on optimized B22, not the original Run 1 configuration; direct compliance with a Run-1-only assignment requires supervisor confirmation. (6) No independent held-out V4 evaluation of the new WD configurations is claimed. (7) Individual seed-study backup artifacts and a WD=0.10 checkpoint archive remain incomplete or unverified in the inspected Drive folder.

# 10\. Conclusions and recommended handoff

The assigned exploratory work is complete: five seeds at WD=0.01 and five WD values at seed=42. Observed seed-study F1 averaged 88.67% (sample SD 1.46 points). Among the tested WD values, WD=0.10 achieved the highest seed-42 validation F1, 89.51%, improving by 1.43 points over the same-environment WD=0.01 replication. These findings should be presented as sensitivity-analysis evidence, not proof of a stable generalization gain. For the team handoff, retain this report alongside all original per-run metrics, note the mixed-environment caveat, and, if choosing a final production candidate, validate the selected checkpoint on an untouched held-out set rather than selecting by that set.  
