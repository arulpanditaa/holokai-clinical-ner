"""Combined baseline test on the team's merged 1320-row corpus
(merged_candidates_FINAL_CONTEXT_CLEAN_120_EACH.json), all 11 LLM-type entities,
swept across thresholds 0.3/0.4/0.5/0.6 -- per Arul's request in the group chat
to run a combined test (not split per person) as the baseline before
separating work back out. Same relaxed-overlap matching methodology as the
team's other reports.
"""
import json
import time
from collections import defaultdict
from gliner import GLiNER

THRESHOLDS = [0.3, 0.4, 0.5, 0.6]
DATA_FILE = "merged_candidates_FINAL_UNIQUE_CONTEXT_120_EACH.json"

LABEL_TO_PROMPT = {
    "PROVIDER": "healthcare provider name",
    "FACILITY": "healthcare facility name",
    "LAB_RESULT": "lab result or test finding",
    "UNIQUE_ID": "unique identifier or record number",
    "SEX": "sex assigned at birth",
    "MEDICATION": "medication, drug, or prescription",
    "DIAGNOSIS": "medical diagnosis or condition",
    "PROCEDURE": "medical procedure or test performed",
    "PERSON": "person name",
    "LOCATION_ADDRESS": "home or physical address",
    "GENDER_IDENTITY": "gender identity",
}
PROMPT_TO_LABEL = {v: k for k, v in LABEL_TO_PROMPT.items()}
PROMPTS = list(LABEL_TO_PROMPT.values())


def overlaps(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


def score(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return p, r, f1


def main():
    t_load_start = time.time()
    print("Loading model...")
    model = GLiNER.from_pretrained("knowledgator/gliner-pii-base-v1.0")
    # Arul's fix: give the tokenizer the same input limit GLiNER uses, so the
    # "no maximum length" truncation warning stops firing.
    tokenizer = model.data_processor.transformer_tokenizer
    tokenizer.model_max_length = model.config.max_len
    print(f"Model loaded in {time.time() - t_load_start:.1f}s\n")

    rows = json.load(open(DATA_FILE, encoding="utf-8"))
    print(f"Loaded {len(rows)} documents\n")

    t_infer_start = time.time()
    raw_preds = []
    for i, row in enumerate(rows):
        preds = model.predict_entities(row["text"], PROMPTS, threshold=0.0)
        raw_preds.append(preds)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_infer_start
            print(f"  scored {i + 1}/{len(rows)}  ({elapsed:.1f}s elapsed, {elapsed/(i+1):.3f}s/doc)")
    infer_time = time.time() - t_infer_start
    print(f"\nInference done: {infer_time:.1f}s total, {infer_time/len(rows):.3f}s/doc average\n")

    overall_rows = []
    per_label_rows = defaultdict(list)

    for thresh in THRESHOLDS:
        stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for row, preds in zip(rows, raw_preds):
            gold_by_label = defaultdict(list)
            for g in row["entities"]:
                gold_by_label[g["label"]].append((g["start"], g["end"]))
            pred_by_label = defaultdict(list)
            for p in preds:
                if p["score"] < thresh:
                    continue
                label = PROMPT_TO_LABEL.get(p["label"])
                if label:
                    pred_by_label[label].append((p["start"], p["end"]))

            for label in LABEL_TO_PROMPT:
                g_spans = gold_by_label.get(label, [])
                p_spans = pred_by_label.get(label, [])
                matched_gold, matched_pred = set(), set()
                for pi, (ps, pe) in enumerate(p_spans):
                    for gi, (gs, ge) in enumerate(g_spans):
                        if gi in matched_gold:
                            continue
                        if overlaps(ps, pe, gs, ge):
                            matched_gold.add(gi); matched_pred.add(pi)
                            break
                tp = len(matched_gold)
                fn = len(g_spans) - len(matched_gold)
                fp = len(p_spans) - len(matched_pred)
                stats[label]["tp"] += tp
                stats[label]["fn"] += fn
                stats[label]["fp"] += fp

        total_tp = sum(s["tp"] for s in stats.values())
        total_fp = sum(s["fp"] for s in stats.values())
        total_fn = sum(s["fn"] for s in stats.values())
        p, r, f1 = score(total_tp, total_fp, total_fn)
        overall_rows.append((thresh, total_tp, total_fp, total_fn, p, r, f1))

        for label in LABEL_TO_PROMPT:
            s = stats[label]
            lp, lr, lf1 = score(s["tp"], s["fp"], s["fn"])
            per_label_rows[label].append((thresh, s["tp"], s["fp"], s["fn"], lp, lr, lf1))

    print("=" * 80)
    print("OVERALL THRESHOLD COMPARISON (all 11 entities combined)")
    print("=" * 80)
    print(f"{'Thresh':<8}{'TP':<6}{'FP':<6}{'FN':<6}{'Precision':<12}{'Recall':<10}{'F1':<8}")
    for thresh, tp, fp, fn, p, r, f1 in overall_rows:
        print(f"{thresh:<8}{tp:<6}{fp:<6}{fn:<6}{p*100:<11.2f}%{r*100:<9.2f}%{f1*100:<7.2f}%")

    print()
    print("=" * 80)
    print("PER-LABEL THRESHOLD COMPARISON")
    print("=" * 80)
    best_per_label = {}
    for label, rows_ in per_label_rows.items():
        print(f"\n-- {label} --")
        print(f"{'Thresh':<8}{'TP':<6}{'FP':<6}{'FN':<6}{'Precision':<12}{'Recall':<10}{'F1':<8}")
        best = max(rows_, key=lambda x: x[-1])
        best_per_label[label] = best
        for thresh, tp, fp, fn, p, r, f1 in rows_:
            flag = "  <- best" if thresh == best[0] else ""
            print(f"{thresh:<8}{tp:<6}{fp:<6}{fn:<6}{p*100:<11.2f}%{r*100:<9.2f}%{f1*100:<7.2f}%{flag}")

    print()
    print("=" * 80)
    print("SUMMARY: best threshold per label")
    print("=" * 80)
    for label, (thresh, tp, fp, fn, p, r, f1) in best_per_label.items():
        print(f"{label:<20} best threshold={thresh}  F1={f1*100:.2f}%")


if __name__ == "__main__":
    main()
