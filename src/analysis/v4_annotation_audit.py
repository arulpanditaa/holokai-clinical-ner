"""
How do train, val and V4 *gold annotations* treat the ambiguous categories behind the weak labels?
No model is loaded and nothing is predicted: V4 is read only for its annotation conventions, to set
labeling rules (vital signs, qualitative findings, exams, symptoms, medication dose, list items).

Colab: open run_v4_annotation_audit.ipynb in week7/weak_label_error_analysis, Run all (~1 min).
Outputs: outputs/v4_audit/ next to this script.
"""
import argparse, collections, glob, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from error_analysis_weak_labels import (ROOT, FOCUS, SHORT, discover, load_records, overlap,
                                        span_stats)
import pandas as pd

# Category probes: text patterns whose gold treatment we compare across splits.
PROBES = {
    "vital: blood pressure": r"\b\d{2,3} ?/ ?\d{2,3}(?: ?mm ?hg)?\b",
    "vital: SpO2": r"\b(?:spo2|sao2|o2 sat\w*|saturation|saturating)\b\D{0,15}\d{2,3} ?%",
    "vital: temperature": r"\b\d{2}(?: ?\. ?\d)? ?° ?[cf]\b|\btemp(?:erature)?\b\D{0,10}\d{2}(?: ?\. ?\d)?",
    "vital: heart/resp rate": r"\b(?:heart rate|hr|pulse|respiratory rate|rr)\b\D{0,10}\d{2,3}\b",
    "qualitative finding": r"\b(?:normal|negative|unremarkable|within normal limits|positive|elevated|reduced|clear)\b",
    "exam / assessment": r"\b(?:\w+ )?(?:examination|exam|assessment|screening|checks?)\b",
    "scale / score": r"\b(?:glasgow coma scale|gcs|apgar|nihss)\b",
    "counseling / education": r"\bcounsell?ing\b|\beducation\b",
    "symptom / sign": r"\b(?:hypoxia|seizures?|dehydration|jaundice|hives|fever|nausea|vomiting|dyspnea|"
                      r"shortness of breath|chest pain|headache|fatigue|tachycardia)\b",
    "lab test name": r"\b(?:cbc|fbc|urinalysis|cmp|bmp|panel|hba1c|ferritin|culture|pcr|troponin|acr|lipid profile)\b",
    "surgical procedure": r"\b\w*(?:ectomy|otomy|ostomy|plasty)\b|\b(?:repair|resection|biopsy|surgery)\b",
}

# Span-composition checks on gold spans of one label: share of spans matching each pattern.
COMPOSITION = {
    "medication, drug, or prescription": {
        "has dose (number + unit)": r"\d ?(?:mg|mcg|g|ml|units?|%|iu)\b",
        "has frequency": r"\b(?:daily|twice|once|bid|tid|qid|qd|prn|weekly|hourly|every|nightly|as needed)\b",
        "has route": r"\b(?:oral|orally|iv|intravenous|po|subcutaneous|sc|im|topical|inhaled|nebulized)\b",
    },
    "lab result or test finding": {
        "contains comma (multi-finding span)": r",",
        "value only (starts with a number)": r"^\s*[\d<>]",
        "contains qualitative word": PROBES["qualitative finding"],
        "contains vital-sign pattern": "|".join(PROBES[k] for k in PROBES if k.startswith("vital")),
    },
    "medical procedure or test performed": {
        "is exam / assessment": PROBES["exam / assessment"],
        "is lab test name": PROBES["lab test name"],
        "is surgical": PROBES["surgical procedure"],
        "is scale / counseling": PROBES["scale / score"] + "|" + PROBES["counseling / education"],
    },
    "condition": {
        "is symptom / sign": PROBES["symptom / sign"],
        "has modifier (acute/chronic/severe/mild)": r"\b(?:acute|chronic|severe|mild|moderate|recurrent)\b",
    },
}


def find_v4(root):
    for n in ["corpus_V4_gliner_mapping.json", "*V4*gliner*mapping*.json", "*V4*gliner*.json"]:
        hits = sorted(glob.glob(os.path.join(root, "**", n), recursive=True))
        if hits:
            return hits[0]
    return None


def probe_counts(splits):
    rows = []
    for name, recs in splits.items():
        for probe, pat in PROBES.items():
            rx = re.compile(pat, re.I)
            c = collections.Counter()
            for r in recs:
                for m in rx.finditer(r["text"]):
                    labs = sorted({SHORT.get(l, l) for s, e, l in r["gold"] if overlap((m.start(), m.end()), (s, e))})
                    c["+".join(labs) or "unlabeled"] += 1
            n = sum(c.values())
            rows.append({"probe": probe, "split": name, "matches": n,
                         "treatment": "; ".join(f"{k} {100 * v / n:.0f}% ({v})" for k, v in c.most_common(4)) if n else ""})
    return pd.DataFrame(rows)


def composition(splits):
    rows = []
    for name, recs in splits.items():
        for lab, checks in COMPOSITION.items():
            spans = [r["text"][s:e] for r in recs for s, e, l in r["gold"] if l == lab]
            for check, pat in checks.items():
                hit = sum(bool(re.search(pat, s, re.I)) for s in spans)
                rows.append({"label": SHORT[lab], "check": check, "split": name, "spans": len(spans),
                             "share_%": round(100 * hit / len(spans), 1) if spans else None})
    return pd.DataFrame(rows)


def samples(splits, k=25, seed=0):
    rows = []
    for name, recs in splits.items():
        for lab in FOCUS:
            pool = [(r["text"], s, e) for r in recs for s, e, l in r["gold"] if l == lab]
            for t, s, e in random.Random(seed).sample(pool, min(k, len(pool))):
                rows.append({"split": name, "label": SHORT[lab], "span": t[s:e],
                             "context": (t[max(0, s - 60):s] + "{{" + t[s:e] + "}}" + t[e:e + 60]).replace("\n", " ")})
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--v4")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "v4_audit"))
    a = ap.parse_args()

    _, t, v = discover(a.root)
    a.train, a.val, a.v4 = a.train or t, a.val or v, a.v4 or find_v4(a.root)
    print(f"train: {a.train}\nval:   {a.val}\nV4:    {a.v4}")
    if not (a.train and a.val and a.v4):
        sys.exit("Could not locate train/val/V4. Pass --train --val --v4 explicitly.")
    os.makedirs(a.out, exist_ok=True)

    splits = {"train": load_records(a.train), "val": load_records(a.val), "V4": load_records(a.v4)}
    for n, recs in splits.items():
        print(f"{n}: {len(recs)} records / {sum(len(r['gold']) for r in recs)} entities")

    pc, comp, ss, smp = probe_counts(splits), composition(splits), span_stats(splits, FOCUS), samples(splits)
    pc.to_csv(f"{a.out}/probe_treatment.csv", index=False)
    comp.to_csv(f"{a.out}/span_composition.csv", index=False)
    ss.to_csv(f"{a.out}/span_stats_train_val_v4.csv", index=False)
    smp.to_csv(f"{a.out}/gold_span_samples.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 110)
    lines = []
    def say(x=""):
        print(x); lines.append(str(x))
    say("=" * 100)
    say("How each split's GOLD labels treat ambiguous text (share of pattern matches by gold label; 'unlabeled' = no gold span)")
    for probe, g in pc.groupby("probe", sort=False):
        say(f"\n[{probe}]")
        say(g[["split", "matches", "treatment"]].to_string(index=False))
    say("\n" + "=" * 100)
    say("Composition of gold spans (share of spans of that label matching the check)")
    say(comp.pivot_table(index=["label", "check"], columns="split", values="share_%", sort=False)
        .reindex(columns=list(splits)).to_string())
    say("\n" + "=" * 100)
    say("Span length / first and last words")
    say(ss.to_string(index=False))
    with open(f"{a.out}/v4_audit_summary.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\nAll outputs written to {a.out}")


if __name__ == "__main__":
    main()
