# **ModernBERT NER Baseline Report**

## **1\. Objective & Models**

Compare ModernBERT Large (**answerdotai/ModernBERT-large**) and its BioClinical variant (**thomas-sounack/BioClinical-ModernBERT-large**) as frozen encoders for HoloKai NER.

## **2\. Dataset & Method**

Week 8 Option B: **1,509 Train** and **171 Validation** records; 11 entity labels. Both encoders remain frozen; only identical linear BIO heads (1,024 → 23 tags) are trained. Strict entity-level micro evaluation requires exact word-span boundaries and label. Checkpoints and configurations are selected by Validation F1.

## **3\. Experimental Setup**

Shared: seed 42, batch size 8, AdamW, zero weight decay, gradient clipping 1.0, FP32. Initial baseline: 10 epochs. Head tuning: at most 20 epochs, early-stopping patience 3\.

Best saved configurations across the initial baseline and all head-tuning runs: [ModernBERT](https://drive.google.com/drive/u/0/folders/15M0d1RXGR8UIfid5Kkw-eZB-APOdHZ_c) — LR **1e-3**, epoch **9** (initial); [BioClinical ModernBERT](https://drive.google.com/drive/u/0/folders/1_oq93cABgrc6Zfpp2ES6uvyS7UYUXryz) — LR **3e-3**, epoch **11** (tuned).

## **4\. Best Results & Findings**

| Model | Precision | Recall | F1 | TP | FP | FN |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- |
| ModernBERT Large | 0.4401 | 0.5039 | 0.4698 | 257 | 327 | 253 |
| BioClinical ModernBERT Large | 0.5281 | 0.6255 | **0.5727** | 319 | 285 | 191 |

• BioClinical ModernBERT leads by **10.29 F1 percentage points**.

• Head tuning improves BioClinical F1 by **0.47 points**; the initial ModernBERT baseline remains its best result.  
