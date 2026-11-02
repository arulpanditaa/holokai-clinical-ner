"""Frozen ModernBERT linear NER baseline; all dataset offsets remain inclusive."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

LABELS = ["condition", "organization medical facility", "gender identity",
          "lab result or test finding", "location address", "medication, drug, or prescription",
          "name", "medical procedure or test performed", "name medical professional",
          "sex assigned at birth", "personal identifier number"]
TAGS = ["O"] + [f"{p}-{label}" for label in LABELS for p in ("B", "I")]
FILES = {"train": "train_gliner_mapping_option_B.json",
         "validation": "val_gliner_mapping_option_B.json"}
COUNTS = {"train": 1509, "validation": 171}
MODELS = {"modernbert_large": "answerdotai/ModernBERT-large",
          "bioclinical_modernbert_large": "thomas-sounack/BioClinical-ModernBERT-large"}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def load_split(root, split):
    path = Path(root) / FILES[split]
    raw = path.read_bytes()
    rows = json.loads(raw)
    assert len(rows) == COUNTS[split], (split, len(rows))
    seen = set()
    for r in rows:
        assert r["tokenized_text"] and all(isinstance(w, str) and w for w in r["tokenized_text"])
        occupied = set()
        for start, end, label in r["ner"]:
            assert label in LABELS and 0 <= start <= end < len(r["tokenized_text"])
            positions = set(range(start, end + 1))
            assert not occupied & positions, "Overlapping entities cannot use a single BIO head"
            occupied |= positions
            seen.add(label)
    assert seen == set(LABELS), (split, seen)
    return rows, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "records": len(rows)}


def word_labels(row):
    labels = [0] * len(row["tokenized_text"])
    for start, end, label in row["ner"]:
        base = 1 + 2 * LABELS.index(label)
        labels[start] = base
        labels[start + 1:end + 1] = [base + 1] * (end - start)
    return torch.tensor(labels, dtype=torch.long)


def decode(ids):
    """An orphan/type-changing I tag starts a new entity, identically everywhere."""
    spans, active, start = [], None, None
    for i, tag_id in enumerate(list(ids) + [0]):
        tag = TAGS[int(tag_id)]
        label = None if tag == "O" else tag[2:]
        if active is not None and (label != active or tag.startswith("B-")):
            spans.append([start, i - 1, active])
            active = None
        if label is not None and active is None:
            active, start = label, i
    return spans


def score(rows, predictions):
    totals = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for row, predicted in zip(rows, predictions):
        gold, pred = set(map(tuple, row["ner"])), set(map(tuple, predicted))
        for field, spans in (("tp", gold & pred), ("fp", pred - gold), ("fn", gold - pred)):
            for _, _, label in spans:
                totals[label][field] += 1

    def finish(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        return dict(c, precision=tp / (tp + fp) if tp + fp else 0.0,
                    recall=tp / (tp + fn) if tp + fn else 0.0,
                    f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"overall": finish({k: sum(c[k] for c in totals.values()) for k in ("tp", "fp", "fn")}),
            "per_label": {label: finish(c) for label, c in totals.items()}}


def tokenize(tokenizer, rows, limit=8192):
    enc = tokenizer([r["tokenized_text"] for r in rows], is_split_into_words=True,
                    truncation=False, padding=False, return_attention_mask=True)
    features = []
    for i, row in enumerate(rows):
        ids = enc["input_ids"][i]
        assert len(ids) <= limit, f"Record {i} exceeds encoder capacity; no truncation permitted"
        first, prev = [], None
        for pos, word in enumerate(enc.word_ids(i)):
            if word is not None and word != prev:
                first.append(pos)
            prev = word
        assert len(first) == len(row["tokenized_text"]), "Tokenizer lost original words"
        features.append({"input_ids": ids, "attention_mask": enc["attention_mask"][i], "first": first})
    return features


def encode(encoder, tokenizer, rows, batch_size, device):
    encoded = tokenize(tokenizer, rows, encoder.config.max_position_embeddings)
    cached = []
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            part = encoded[offset:offset + batch_size]
            batch = tokenizer.pad([{k: r[k] for k in ("input_ids", "attention_mask")} for r in part],
                                  padding=True, return_tensors="pt").to(device)
            hidden = encoder(**batch).last_hidden_state
            for j, row in enumerate(part):
                cached.append(hidden[j, row["first"]].float().cpu().clone())
            if offset % (batch_size * 20) == 0:
                print(f"  encoded {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
    return cached, max(len(r["input_ids"]) for r in encoded)


def predict(head, cached, device):
    with torch.no_grad():
        return [decode(head(x.to(device)).argmax(-1).cpu().tolist()) for x in cached]


def train(args):
    learning_rates = [3e-4, 1e-3, 3e-3] if args.lr_sweep else [args.lr]
    max_epochs = 20 if args.lr_sweep else args.epochs
    patience = 3 if args.lr_sweep else 0
    assert max_epochs > 0 and all(lr > 0 for lr in learning_rates)
    if args.lr_sweep:
        # Keep experiments separate from the original baseline and OpenMed results.
        assert Path(args.output).parent.name == "head_lr_sweep", "Use output/head_lr_sweep/<model>"
    existing = Path(args.output) / "metrics.json"
    if existing.exists():
        saved = json.loads(existing.read_text())
        assert saved["model_id"] == MODELS[args.model], "Existing model differs"
        assert bool(saved.get("sweep")) == args.lr_sweep, "Existing experiment mode differs"
        expected = {"seed": args.seed, "epochs": max_epochs, "batch_size": args.batch_size}
        if args.lr_sweep:
            assert saved["sweep"]["learning_rates"] == learning_rates
            assert saved["sweep"]["patience"] == patience
        else:
            expected["lr"] = args.lr
        for key, value in expected.items():
            assert saved["hyperparameters"][key] == value, f"Existing {key} differs"
        if args.revision:
            assert saved["revision"] == args.revision, "Existing encoder revision differs"
        for split in FILES:
            _, manifest = load_split(args.data, split)
            assert saved["data"][split] == manifest, "Existing results used different data"
        artifact = (Path(args.output) / "model.onnx" if saved["onnx"]["verified"]
                    else Path(args.scratch) / "best_head.pt")
        assert artifact.is_file(), "Selected model artifact is unavailable; refusing to retrain"
        print(f"REUSING_SELECTED_MODEL {args.model} validation_f1={saved['validation']['overall']['f1']:.6f}", flush=True)
        return
    assert torch.cuda.is_available(), "A Colab GPU runtime is required"
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output, scratch = Path(args.output), Path(args.scratch)
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    assert not (output / "metrics.json").exists(), "Refusing to overwrite existing experiment results"
    train_rows, train_manifest = load_split(args.data, "train")
    val_rows, val_manifest = load_split(args.data, "validation")
    train_text = {tuple(r["tokenized_text"]) for r in train_rows}
    assert not train_text & {tuple(r["tokenized_text"]) for r in val_rows}, "Train/validation leakage"
    model_id = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision, use_fast=True)
    encoder = AutoModel.from_pretrained(model_id, revision=args.revision, torch_dtype=torch.float32,
                                       attn_implementation="sdpa", reference_compile=False).to(device).eval()
    assert encoder.config.hidden_size == 1024
    encoder.requires_grad_(False)
    versions = [p._version for p in encoder.parameters()]
    revision = encoder.config._commit_hash
    print(f"{model_id} revision={revision}; frozen encoder; GPU={torch.cuda.get_device_name()}", flush=True)
    x_train, max_train = encode(encoder, tokenizer, train_rows, args.batch_size, device)
    x_val, max_val = encode(encoder, tokenizer, val_rows, args.batch_size, device)
    y_train = [word_labels(r) for r in train_rows]
    # Reuse frozen features for every LR; reset the same linear head and data order.
    checkpoint = scratch / "best_head.pt"
    trials, best_f1, initial_hash = [], -1.0, None
    for lr in learning_rates:
        torch.manual_seed(args.seed)
        head = nn.Linear(1024, len(TAGS)).to(device)
        trial_hash = hashlib.sha256(head.weight.detach().cpu().numpy().tobytes()).hexdigest()
        if initial_hash is None:
            initial_hash = trial_hash
        assert trial_hash == initial_hash, "Head initialization must match across LRs"
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
        generator = torch.Generator().manual_seed(args.seed)
        trial_history, trial_f1, trial_epoch, stale = [], -1.0, None, 0
        for epoch in range(1, max_epochs + 1):
            head.train()
            order = torch.randperm(len(train_rows), generator=generator).tolist()
            total_loss, total_words = 0.0, 0
            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                x = torch.cat([x_train[i] for i in indices]).to(device)
                y = torch.cat([y_train[i] for i in indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(head(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y)
                total_words += len(y)
            head.eval()
            metrics = score(val_rows, predict(head, x_val, device))
            trial_history.append({"epoch": epoch, "train_loss": total_loss / total_words,
                                  "validation": metrics["overall"]})
            print(json.dumps({"lr": lr, **trial_history[-1]}), flush=True)
            if metrics["overall"]["f1"] > trial_f1:
                trial_f1, trial_epoch, stale = metrics["overall"]["f1"], epoch, 0
                trial_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                trial_metrics = metrics["overall"]
            else:
                stale += 1
            if patience and stale >= patience:
                break
        trials.append({"lr": lr, "selected_epoch": trial_epoch, "epochs_run": len(trial_history),
                       "early_stopped": bool(patience and stale >= patience),
                       "validation": trial_metrics})
        # Strict improvement keeps the earliest epoch/LR on a tie.
        if trial_f1 > best_f1:
            best_f1, best_epoch, selected_lr = trial_f1, trial_epoch, lr
            history = trial_history
            torch.save({"head": trial_state, "model_id": model_id, "revision": revision,
                        "tags": TAGS, "epoch": best_epoch, "lr": selected_lr}, checkpoint)
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["head"])
    head.eval()
    val_predictions = predict(head, x_val, device)
    validation = score(val_rows, val_predictions)
    assert validation["overall"]["f1"] == best_f1
    frozen_ok = versions == [p._version for p in encoder.parameters()] and all(
        not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert frozen_ok
    result = {"model_id": model_id, "revision": revision, "labels": LABELS, "tags": TAGS,
              "annotation_status": "Option B automated, unverified additions",
              "hyperparameters": {"lr": selected_lr, "seed": args.seed, "epochs": max_epochs,
                                  "batch_size": args.batch_size, "optimizer": "AdamW",
                                  "weight_decay": 0.0, "max_grad_norm": 1.0, "precision": "float32"},
              "head_initial_weight_sha256": initial_hash, "encoder_frozen_verified": frozen_ok,
              "selected_epoch": best_epoch, "selection_metric": "validation exact-span micro F1",
              "data": {"train": train_manifest, "validation": val_manifest},
              "max_subtokens": {"train": max_train, "validation": max_val},
              "environment": {k: importlib.metadata.version(k) for k in
                              ("torch", "transformers", "numpy", "onnx", "onnxruntime")},
              "gpu": torch.cuda.get_device_name(), "history": history,
              "validation": validation, "onnx": {"verified": False}}
    if args.lr_sweep:
        result["sweep"] = {"learning_rates": learning_rates, "max_epochs": max_epochs,
                           "patience": patience, "min_delta": 0.0, "trials": trials,
                           "selected_lr": selected_lr,
                           "tie_policy": "earliest best epoch; first LR in listed order",
                           "features_cached_once": True}
    write_json(output / "metrics.json", result)
    with (output / "per_label_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        for split, metrics in (("validation", validation),):
            for label, values in {"overall_micro": metrics["overall"], **metrics["per_label"]}.items():
                writer.writerow({"split": split, "label": label, **values})
    write_json(output / "predictions.json", {
        split: [{"sample_index": i, "tokens": r["tokenized_text"], "gold_ner": r["ner"],
                 "predicted_ner": p} for i, (r, p) in enumerate(zip(rows, predictions))]
        for split, rows, predictions in (("validation", val_rows, val_predictions),)})
    tokenizer.save_pretrained(output)
    print(f"TRAIN_EVAL_COMPLETE {args.model} validation_f1={best_f1:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-sweep", action="store_true",
                        help="Frozen-head experiment: LRs 3e-4/1e-3/3e-3; max 20 epochs; patience 3")
    parser.add_argument("--revision", help="Optional pinned pretrained encoder/tokenizer revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    train(parser.parse_args())
