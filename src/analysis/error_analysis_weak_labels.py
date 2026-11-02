"""
HoloKai GLiNER error analysis for weak labels (validation + train; V4 untouched).

Colab usage: upload this folder to Drive Holokai_Project/, open run_error_analysis.ipynb, run all.
Or, after mounting Drive:
    !python /content/drive/MyDrive/Holokai_Project/weak_label_error_analysis/error_analysis_weak_labels.py

Paths are auto-discovered under Holokai_Project/ (train = 1509 records, val = 171 records).
Outputs go to outputs/ next to this script. Override with --model / --train / --val / --out.
"""
import argparse, collections, glob, json, os, re, sys

# PyTorch only: keep transformers from loading Colab's TensorFlow (and its absl/CUDA log noise)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd

ROOT = "/content/drive/MyDrive/Holokai_Project"
# Drive layout: Final Run model and its train/val live under week6/ (see final_model/configs/finetune_final_run.json)
MODEL_REL = "week6/final_model/outputs/training/best_model"
TRAIN_REL = "week6/finetune/data/train_gliner_mapping.json"
VAL_REL = "week6/finetune/data/val_gliner_mapping.json"

# original corpus label -> recommended GLiNER label name (verified by per-label support counts)
LABEL_MAP = {
    "PERSON": "name",
    "PROVIDER": "name medical professional",
    "FACILITY": "organization medical facility",
    "LOCATION_ADDRESS": "location address",
    "UNIQUE_ID": "personal identifier number",
    "SEX": "sex assigned at birth",
    "GENDER_IDENTITY": "gender identity",
    "MEDICATION": "medication, drug, or prescription",
    "DIAGNOSIS": "condition",
    "LAB_RESULT": "lab result or test finding",
    "PROCEDURE": "medical procedure or test performed",
}
LABELS = list(LABEL_MAP.values())
FOCUS = ["condition", "lab result or test finding", "medical procedure or test performed",
         "medication, drug, or prescription"]
SHORT = {v: k for k, v in LABEL_MAP.items()}  # for compact display

THRESHOLD = 0.4
LOW_THRESHOLD = 0.1
FLAT_NER = False
CTX = 100  # chars of context on each side


# ---------------------------------------------------------------- data

def norm_label(lab):
    return LABEL_MAP.get(lab, LABEL_MAP.get(lab.upper(), lab))


def load_records(path):
    """Return list of {text, offs (token char offsets or None), gold: set[(s,e,label)]} in char space."""
    with open(path) as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(l) for l in raw.splitlines() if l.strip()]
    recs = []
    for d in data:
        if "tokenized_text" in d:
            toks = d["tokenized_text"]
            offs, pos = [], 0
            for t in toks:
                offs.append((pos, pos + len(t)))
                pos += len(t) + 1
            text = " ".join(toks)
            gold = {(offs[s][0], offs[e][1], norm_label(l)) for s, e, l in d.get("ner", [])}
        else:
            text, offs = d["text"], None
            gold = {(e["start"], e["end"], norm_label(e["label"])) for e in d.get("entities", [])}
        recs.append({"text": text, "offs": offs, "gold": gold})
    return recs


def discover(root):
    """Find model and train/val by name (as in final_model/configs/finetune_final_run.json), at any depth
    (Drive is reorganized into weekN/ folders)."""
    fixed = [os.path.join(root, p) for p in (MODEL_REL, TRAIN_REL, VAL_REL)]
    if all(os.path.exists(p) for p in fixed):
        return tuple(fixed)

    def find(*names):
        for n in names:
            hits = sorted(glob.glob(os.path.join(root, "**", n), recursive=True))
            if hits:
                return hits[0]
        return None
    return (find("final_model/outputs/training/best_model"),
            find("train_gliner_mapping.json", "train_gliner.json"),
            find("val_gliner_mapping.json", "val_gliner.json"))


# ---------------------------------------------------------------- prediction

def snap(s, e, offs):
    """Snap a char span to whole tokens, so comparisons match token-level evaluation."""
    if offs is None:
        return s, e
    i = next(k for k, (a, b) in enumerate(offs) if b > s)
    j = max(k for k, (a, b) in enumerate(offs) if a < e)
    return offs[i][0], offs[j][1]


def predict(model, recs, threshold):
    out = []
    for k, r in enumerate(recs, 1):
        if k % 100 == 0 or k == len(recs):
            print(f"  {k}/{len(recs)} @ threshold {threshold}", flush=True)
        ents = model.predict_entities(r["text"], LABELS, threshold=threshold, flat_ner=FLAT_NER)
        out.append({(*snap(p["start"], p["end"], r["offs"]), p["label"]): p["score"] for p in ents})
    return out


# ---------------------------------------------------------------- analysis

def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def boundary_kind(p, g):
    if p[0] <= g[0] and p[1] >= g[1]:
        return "too_long"
    if p[0] >= g[0] and p[1] <= g[1]:
        return "too_short"
    return "shifted"


def ctx(text, spans):
    """Context window; gold marked {{ }}, prediction marked [[ ]]."""
    lo = max(0, min(s for s, _ in spans) - CTX)
    hi = min(len(text), max(e for _, e in spans) + CTX)
    marks = []
    for k, (s, e) in enumerate(spans):
        o, c = ("{{", "}}") if k == 0 else ("[[", "]]")
        marks += [(s, o), (e, c)]
    out, pos = [], lo
    for at, m in sorted(marks, key=lambda x: (x[0], x[1] in ("}}", "]]"))):
        out.append(text[pos:at]); out.append(m); pos = at
    out.append(text[pos:hi])
    return "".join(out).replace("\n", " ")


def analyze(recs, preds, low_preds, split):
    rows, conf = [], collections.Counter()
    for i, (r, P, L) in enumerate(zip(recs, preds, low_preds)):
        text, G = r["text"], r["gold"]
        exact = G & set(P)
        for g in exact:
            conf[(g[2], g[2])] += 1
        un_g = G - exact
        paired = set()
        for p in sorted(set(P) - exact):
            cands = sorted(un_g, key=lambda g: -overlap(p, g))
            g = cands[0] if cands and overlap(p, cands[0]) > 0 else None
            if g is None:
                kind = "spurious_fp"
            elif (g[0], g[1]) == (p[0], p[1]):
                kind = "label_confusion"
            elif g[2] == p[2]:
                kind = "boundary_" + boundary_kind(p, g)
            else:
                kind = "boundary_and_label"
            if g:
                paired.add(g)
            conf[(g[2] if g else "O", p[2])] += 1
            rows.append({
                "split": split, "rec": i, "type": kind,
                "gold_label": SHORT.get(g[2], g[2]) if g else "",
                "pred_label": SHORT.get(p[2], p[2]),
                "gold_text": text[g[0]:g[1]] if g else "",
                "pred_text": text[p[0]:p[1]], "pred_score": round(P[p], 3),
                "context": ctx(text, [g[:2], p[:2]] if g else [p[:2], p[:2]]),
            })
        for g in sorted(un_g - paired):
            same = L.get(g)
            near = max((v for k, v in L.items() if k[2] == g[2] and overlap(k, g) > 0), default=None)
            score = same if same is not None else near
            conf[(g[2], "O")] += 1
            rows.append({
                "split": split, "rec": i,
                "type": "missed_fn_low_score" if score is not None else "missed_fn_unseen",
                "gold_label": SHORT.get(g[2], g[2]), "pred_label": "",
                "gold_text": text[g[0]:g[1]], "pred_text": "",
                "pred_score": round(score, 3) if score is not None else "",
                "context": ctx(text, [g[:2]]),
            })
    return pd.DataFrame(rows), conf


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 100 * p, 100 * r, (200 * p * r / (p + r) if p + r else 0.0)


def scores(recs, preds):
    """Per-label exact and relaxed (overlap, same label) P/R/F1."""
    c = collections.defaultdict(lambda: collections.Counter())
    for r, P in zip(recs, preds):
        G, Ps = r["gold"], set(P)
        for lab in LABELS + ["__ALL__"]:
            g = {x for x in G if lab == "__ALL__" or x[2] == lab}
            p = {x for x in Ps if lab == "__ALL__" or x[2] == lab}
            c[lab]["tp"] += len(g & p); c[lab]["fp"] += len(p - g); c[lab]["fn"] += len(g - p)
            c[lab]["rtp_p"] += sum(any(x[2] == y[2] and overlap(x, y) for y in g) for x in p)
            c[lab]["rtp_g"] += sum(any(x[2] == y[2] and overlap(x, y) for y in p) for x in g)
            c[lab]["np"] += len(p); c[lab]["ng"] += len(g)
    out = []
    for lab, k in c.items():
        P, R, F = prf(k["tp"], k["fp"], k["fn"])
        rp = 100 * k["rtp_p"] / k["np"] if k["np"] else 0.0
        rr = 100 * k["rtp_g"] / k["ng"] if k["ng"] else 0.0
        rf = 2 * rp * rr / (rp + rr) if rp + rr else 0.0
        out.append({"label": "ALL" if lab == "__ALL__" else SHORT.get(lab, lab), "support": k["ng"],
                    "TP": k["tp"], "FP": k["fp"], "FN": k["fn"],
                    "P": round(P, 2), "R": round(R, 2), "F1_exact": round(F, 2),
                    "F1_overlap": round(rf, 2), "gap": round(rf - F, 2)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- train-set annotation checks (no model)

def words(s):
    return re.findall(r"\w+(?:[-_/.]\w+)*|\S", s)


def consistency(train, focus):
    surf = collections.defaultdict(collections.Counter)
    for r in train:
        for s, e, l in r["gold"]:
            surf[r["text"][s:e].lower().strip()][l] += 1
    multi = [{"surface": k, "labels": "; ".join(f"{SHORT.get(l, l)}={n}" for l, n in v.most_common())}
             for k, v in surf.items() if len(v) > 1 and any(l in focus for l in v)]

    unl = []
    for k, v in surf.items():
        lab = v.most_common(1)[0][0]
        if lab not in focus or len(k) < 4:
            continue
        pat = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)", re.I)
        labeled = sum(v.values())
        unlabeled = 0
        for r in train:
            for m in pat.finditer(r["text"]):
                if not any(overlap((m.start(), m.end()), g) > 0 for g in r["gold"]):
                    unlabeled += 1
        if unlabeled:
            unl.append({"surface": k, "label": SHORT.get(lab, lab), "labeled": labeled, "unlabeled": unlabeled})
    return (pd.DataFrame(multi).sort_values("surface") if multi else pd.DataFrame(columns=["surface", "labels"]),
            pd.DataFrame(unl).sort_values("unlabeled", ascending=False) if unl
            else pd.DataFrame(columns=["surface", "label", "labeled", "unlabeled"]))


def span_stats(splits, focus):
    rows = []
    for name, recs in splits.items():
        for lab in focus:
            spans = [words(r["text"][s:e]) for r in recs for s, e, l in r["gold"] if l == lab]
            if not spans:
                continue
            lens = pd.Series([len(w) for w in spans])
            first = collections.Counter(w[0].lower() for w in spans).most_common(8)
            last = collections.Counter(w[-1].lower() for w in spans).most_common(8)
            rows.append({"split": name, "label": SHORT[lab], "n": len(spans),
                         "len_mean": round(lens.mean(), 2), "len_median": lens.median(),
                         "len_p90": lens.quantile(0.9), "len_max": lens.max(),
                         "has_digit_%": round(100 * sum(any(c.isdigit() for c in "".join(w)) for w in spans) / len(spans), 1),
                         "top_first_words": ", ".join(f"{w}({n})" for w, n in first),
                         "top_last_words": ", ".join(f"{w}({n})" for w, n in last)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress live under Colab's !python
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--train"); ap.add_argument("--val")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()

    if not (a.model and a.train and a.val):
        m, t, v = discover(a.root)
        a.model, a.train, a.val = a.model or m, a.train or t, a.val or v
    print(f"model: {a.model}\ntrain: {a.train}\nval:   {a.val}")
    if not (a.model and a.train and a.val and os.path.isdir(a.model)):
        sys.exit("Could not locate model/train/val. Pass --model --train --val explicitly.")
    os.makedirs(a.out, exist_ok=True)

    train, val = load_records(a.train), load_records(a.val)
    unknown = {l for r in train + val for *_, l in r["gold"]} - set(LABELS)
    if unknown:
        sys.exit(f"Gold labels not in LABEL_MAP: {unknown}")
    print(f"train {len(train)} recs / {sum(len(r['gold']) for r in train)} ents; "
          f"val {len(val)} recs / {sum(len(r['gold']) for r in val)} ents")

    import torch  # imported here so v4_annotation_audit.py can reuse this module without gliner/torch
    from gliner import GLiNER
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(a.model).to(dev)
    model.eval()
    print("device:", dev)

    results = {}
    for name, recs in [("val", val), ("train", train)]:
        print(f"predicting {name} ...")
        P = predict(model, recs, THRESHOLD)
        L = predict(model, recs, LOW_THRESHOLD)
        errs, conf = analyze(recs, P, L, name)
        sc = scores(recs, P)
        results[name] = (errs, conf, sc)
        errs.to_csv(f"{a.out}/{name}_errors.csv", index=False)
        sc.to_csv(f"{a.out}/{name}_scores.csv", index=False)
        cm = pd.DataFrame(0, index=[SHORT[l] for l in LABELS] + ["O"], columns=[SHORT[l] for l in LABELS] + ["O"])
        for (g, p), n in conf.items():
            cm.loc[SHORT.get(g, g), SHORT.get(p, p)] += n
        cm.index.name = "gold \\ pred"
        cm.to_csv(f"{a.out}/{name}_confusion.csv")
        with open(f"{a.out}/{name}_predictions.jsonl", "w") as f:
            for i, (r, p) in enumerate(zip(recs, L)):
                f.write(json.dumps({"rec": i, "gold": sorted(r["gold"]),
                                    "pred_low": [[*k, round(v, 4)] for k, v in sorted(p.items())]}) + "\n")

    multi, unl = consistency(train, FOCUS)
    multi.to_csv(f"{a.out}/train_multi_label_surfaces.csv", index=False)
    unl.to_csv(f"{a.out}/train_unlabeled_occurrences.csv", index=False)
    ss = span_stats({"train": train, "val": val}, FOCUS)
    ss.to_csv(f"{a.out}/span_stats.csv", index=False)

    # ------------------------------------------------------------ summary
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 120)
    lines = []
    def say(x=""):
        print(x); lines.append(str(x))

    errs, conf, sc = results["val"]
    allrow = sc[sc.label == "ALL"].iloc[0]
    say("=" * 100)
    say(f"SANITY CHECK val @0.4: TP {allrow.TP} FP {allrow.FP} FN {allrow.FN} F1 {allrow.F1_exact}"
        f"   (Final report: TP 415 FP 81 FN 38 F1 87.46)")
    for name in ["val", "train"]:
        say(f"\n--- {name}: exact vs overlap F1 ---")
        say(results[name][2].sort_values("F1_exact").to_string(index=False))
    focus_short = [SHORT[l] for l in FOCUS]
    for name in ["val", "train"]:
        e = results[name][0]
        e = e[e.gold_label.isin(focus_short) | e.pred_label.isin(focus_short)]
        say(f"\n--- {name}: error types involving focus labels (row = one FP or one unmatched FN) ---")
        if len(e):
            e = e.assign(label=e.pred_label.where(e.pred_label != "", e.gold_label))
            say(pd.crosstab(e.label, e.type, margins=True).to_string())
    say("\n--- val: confusion (gold rows x pred cols, boundary errors counted on their label) ---")
    say(pd.read_csv(f"{a.out}/val_confusion.csv", index_col=0).to_string())
    say("\n--- span length / composition ---")
    say(ss.to_string(index=False))
    say(f"\n--- train: {len(multi)} surfaces with >1 label (focus); top 30 ---")
    say(multi.head(30).to_string(index=False))
    say(f"\n--- train: surfaces labeled somewhere but left unlabeled elsewhere; top 30 of {len(unl)} ---")
    say(unl.head(30).to_string(index=False))
    with open(f"{a.out}/summary.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\nAll outputs written to {a.out}")


if __name__ == "__main__":
    main()
