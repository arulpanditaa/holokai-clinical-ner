"""OpenMed pretrained NER: validation inference, conservative projection, and ONNX."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoModelForTokenClassification, AutoTokenizer
from train_eval import LABELS, load_split, score, tokenize, write_json

MODEL_ID = "OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1"
NATIVE_MAPPING = {
    "first_name": "name", "last_name": "name",
    "street_address": "location address", "city": "location address",
    "county": "location address", "state": "location address",
    "country": "location address", "postcode": "location address",
    "ssn": "personal identifier number", "unique_id": "personal identifier number",
    "employee_id": "personal identifier number",
    "customer_id": "personal identifier number", "medical_record_number": "personal identifier number",
    "health_plan_beneficiary_number": "personal identifier number",
    "certificate_license_number": "personal identifier number", "tax_id": "personal identifier number",
    "blood_type": "lab result or test finding",
}
UNSUPPORTED_REASONS = {
    "condition": "No native diagnosis or symptom label.",
    "organization medical facility": "Generic company_name does not identify medical facilities.",
    "gender identity": "Generic gender cannot reliably identify gender identity.",
    "medication, drug, or prescription": "No native medication label.",
    "medical procedure or test performed": "No native procedure or test label.",
    "name medical professional": "Generic first/last names do not distinguish clinical providers.",
    "sex assigned at birth": "Generic gender does not establish sex assigned at birth.",
}
PARTIAL_REASONS = {
    "name": "First/last name components; patient versus provider roles are not distinguished.",
    "location address": "Address/geographic components; exact full-address boundaries can differ.",
    "personal identifier number": "Only the listed personal, clinical, and license ID subtypes.",
    "lab result or test finding": "Blood type only; other lab findings and vital signs are unsupported.",
}


def native_spans(ids, labels):
    spans, active, start = [], None, None
    for i, tag_id in enumerate(list(ids) + [None]):
        tag = "O" if tag_id is None else labels[int(tag_id)]
        label = None if tag == "O" else tag[2:]
        if active is not None and (label != active or tag.startswith("B-")):
            spans.append([start, i - 1, active])
            active = None
        if label is not None and active is None:
            start, active = i, label
    return spans


def project(native, words, mapping):
    """Merge adjacent name parts and address components across commas, without gold labels."""
    mapped = []
    for start, end, source in native:
        target = mapping.get(source)
        if target is None:
            continue
        if mapped and mapped[-1][2] == target:
            previous_end = mapped[-1][1]
            gap = words[previous_end + 1:start]
            adjacent = start == previous_end + 1
            address_comma = target == "location address" and 0 < len(gap) <= 2 and all(w == "," for w in gap)
            if (target in ("name", "location address") and adjacent) or address_comma:
                mapped[-1][1] = end
                continue
        mapped.append([start, end, target])
    return mapped


class LogitsOnly(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.pretrained = pretrained

    def forward(self, input_ids, attention_mask):
        return self.pretrained(input_ids=input_ids, attention_mask=attention_mask).logits


def batches(tokenizer, encoded, indices, batch_size):
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset:offset + batch_size]
        batch = tokenizer.pad([{k: encoded[i][k] for k in ("input_ids", "attention_mask")}
                               for i in selected], padding=True, return_tensors="pt")
        yield selected, batch


def run(args):
    output = Path(args.output)
    assert torch.cuda.is_available(), "OpenMed inference requires the selected Colab GPU"
    if (output / "metrics.json").exists():
        saved = json.loads((output / "metrics.json").read_text())
        if saved.get("onnx", {}).get("verified"):
            print("OPENMED_ALREADY_COMPLETE: reusing verified OpenMed results", flush=True)
            return
        assert saved["model_id"] == MODEL_ID and saved["revision"] == args.revision
        print("RESUMING_INCOMPLETE_OPENMED: rerunning inference/export for this checkpoint only", flush=True)
    rows, manifest = load_split(args.data, "validation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=args.revision, use_fast=True)
    pretrained, loading = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, revision=args.revision, torch_dtype=torch.float32, attn_implementation="eager",
        reference_compile=False, output_loading_info=True)
    assert not loading["missing_keys"] and not loading.get("mismatched_keys"), loading
    pretrained.requires_grad_(False).eval()
    labels = {int(k): v for k, v in pretrained.config.id2label.items()}
    assert len(labels) == pretrained.config.num_labels and "O" in labels.values()
    assert all(t == "O" or t.startswith(("B-", "I-")) for t in labels.values())
    types = sorted({t[2:] for t in labels.values() if t != "O"})
    mapping = {k: v for k, v in NATIVE_MAPPING.items() if k in types}
    supported = sorted(set(mapping.values()))
    coverage = {label: {"status": "partial" if label in supported else "unsupported",
                        "native_labels": sorted(k for k, v in mapping.items() if v == label),
                        "reason": PARTIAL_REASONS[label] if label in supported else UNSUPPORTED_REASONS[label]}
                for label in LABELS}
    print("Native labels:", json.dumps(labels), flush=True)
    print("Conservative mapping:", json.dumps(mapping), flush=True)
    print("Coverage:", json.dumps(coverage), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = LogitsOnly(pretrained).to("cuda").eval()
    versions = [p._version for p in model.parameters()]
    encoded = tokenize(tokenizer, rows, pretrained.config.max_position_embeddings)
    word_logits, predictions, native_predictions = [None] * len(rows), [None] * len(rows), [None] * len(rows)
    ignored_counts = Counter()
    with torch.no_grad():
        for selected, batch in batches(tokenizer, encoded, list(range(len(rows))), args.batch_size):
            logits = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda()).cpu().numpy()
            for j, index in enumerate(selected):
                word_logits[index] = logits[j, encoded[index]["first"]].copy()
                native = native_spans(word_logits[index].argmax(-1), labels)
                native_predictions[index] = native
                predictions[index] = project(native, rows[index]["tokenized_text"], mapping)
                ignored_counts.update(s[2] for s in native if s[2] not in mapping)
            print(f"OpenMed inferred {selected[-1] + 1}/{len(rows)}", flush=True)
    metrics = score(rows, predictions)
    frozen_ok = versions == [p._version for p in model.parameters()] and all(
        not p.requires_grad and p.grad is None for p in model.parameters())
    assert frozen_ok
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    pretrained.config.to_json_file(output / "config.json")
    result = {"model_id": MODEL_ID, "revision": pretrained.config._commit_hash,
              "mode": "pretrained NER inference only; original head unchanged",
              "data": {"validation": manifest}, "labels": LABELS, "native_id2label": labels,
              "mapping": mapping, "label_coverage": coverage,
              "covered_target_labels": supported, "unsupported_target_labels": [x for x in LABELS if x not in supported],
              "label_coverage_fraction": len(supported) / len(LABELS),
              "unmapped_native_labels": [x for x in types if x not in mapping],
              "ignored_native_prediction_counts": dict(ignored_counts),
              "evaluation_policy": "Exact inclusive word-span and label; micro overall across all 11 labels. Unsupported gold entities count as FN. Native out-of-schema predictions are ignored and retained for audit.",
              "merge_policy": "Adjacent mapped name components; adjacent address components or components separated only by one/two commas. No gold-dependent rules.",
              "annotation_status": "Option B automated, unverified additions",
              "max_subtokens": max(len(e["input_ids"]) for e in encoded),
              "batch_size": args.batch_size, "precision": "float32", "model_frozen_verified": frozen_ok,
              "gpu": torch.cuda.get_device_name(),
              "environment": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "onnx", "onnxruntime")},
              "validation": metrics, "onnx": {"verified": False}}
    write_json(output / "metrics.json", result)
    with (output / "per_label_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "coverage", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerow({"label": "overall_micro_all_11", "coverage": f"{len(supported)}/11 partial", **metrics["overall"]})
        for label in LABELS:
            writer.writerow({"label": label, "coverage": coverage[label]["status"], **metrics["per_label"][label]})
    write_json(output / "predictions.json", {"validation": [
        {"sample_index": i, "tokens": r["tokenized_text"], "gold_ner": r["ner"],
         "predicted_ner": predictions[i], "native_entities": native_predictions[i]}
        for i, r in enumerate(rows)]})
    print("OPENMED_VALIDATION_COMPLETE", json.dumps(metrics["overall"]), flush=True)
    model = model.cpu().eval()
    torch.set_num_threads(2)
    dummy = tokenizer(["Example clinical note.", "Blood pressure 120/80."], padding=True, return_tensors="pt")
    onnx_path = output / "model.onnx"
    with torch.no_grad():
        model(dummy["input_ids"], dummy["attention_mask"])
        torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), str(onnx_path),
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                        "attention_mask": {0: "batch", 1: "sequence"},
                                        "logits": {0: "batch", 1: "sequence"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(onnx_path), full_check=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads, opts.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    order = sorted(range(len(rows)), key=lambda i: len(encoded[i]["input_ids"]))
    cases = [[order[0]], [order[0], order[-1]], [order[len(order)//2], order[-2], order[1]]]
    checks = []
    with torch.no_grad():
        for selected in cases:
            _, batch = next(batches(tokenizer, encoded, selected, len(selected)))
            reference = model(batch["input_ids"], batch["attention_mask"]).numpy()
            actual = session.run(["logits"], {k: batch[k].numpy().astype(np.int64) for k in ("input_ids", "attention_mask")})[0]
            mask = batch["attention_mask"].numpy().astype(bool)
            assert reference.shape == actual.shape and np.isfinite(actual).all()
            np.testing.assert_allclose(actual[mask], reference[mask], rtol=1e-3, atol=1e-3)
            for j, index in enumerate(selected):
                positions = encoded[index]["first"]
                assert np.array_equal(actual[j, positions].argmax(-1), reference[j, positions].argmax(-1))
                assert project(native_spans(actual[j, positions].argmax(-1), labels), rows[index]["tokenized_text"], mapping) == predictions[index]
            checks.append({"batch_size": len(selected), "sequence_length": int(actual.shape[1]),
                           "max_abs_error": float(np.max(np.abs(actual[mask] - reference[mask]))), "identical_tags_entities": True})
            print("OpenMed ONNX verification:", checks[-1], flush=True)
    ort_predictions, native_tag_differences = [None] * len(rows), 0
    for selected, batch in batches(tokenizer, encoded, list(range(len(rows))), args.batch_size):
        actual = session.run(["logits"], {k: batch[k].numpy().astype(np.int64) for k in ("input_ids", "attention_mask")})[0]
        for j, index in enumerate(selected):
            native_ids = actual[j, encoded[index]["first"]].argmax(-1)
            native_tag_differences += int(np.sum(native_ids != word_logits[index].argmax(-1)))
            ort_predictions[index] = project(native_spans(native_ids, labels), rows[index]["tokenized_text"], mapping)
        print(f"OpenMed ONNX validated {selected[-1] + 1}/{len(rows)}", flush=True)
    assert ort_predictions == predictions, "ONNX mapped predictions differ on full validation set"
    assert score(rows, ort_predictions) == metrics
    _, after = load_split(args.data, "validation")
    assert after == manifest
    with onnx_path.open("rb") as f:
        onnx_hash = hashlib.file_digest(f, "sha256").hexdigest()
    result["onnx"] = {"verified": True, "opset": 17, "provider": "CPUExecutionProvider", "atol": 1e-3,
                      "rtol": 1e-3, "checks": checks, "full_validation_records": len(rows),
                      "full_validation_mapped_predictions_identical": True,
                      "full_validation_native_tag_differences": native_tag_differences,
                      "sha256": onnx_hash, "bytes": onnx_path.stat().st_size}
    write_json(output / "inference_config.json", {"model_id": MODEL_ID, "revision": result["revision"],
               "native_id2label": labels, "mapping": mapping, "label_coverage": coverage,
               "preprocessing": "AutoTokenizer from this directory; is_split_into_words=True; first subtoken per original word; no truncation",
               "postprocessing": "Native BIO decode then project() in openmed_baseline.py; inclusive word offsets",
               "merge_policy": result["merge_policy"], "onnx_inputs": "int64 input_ids and attention_mask [batch, sequence]",
               "onnx_output": f"float32 logits [batch, sequence, {len(labels)}]"})
    write_json(output / "metrics.json", result)
    print("OPENMED_ONNX_VERIFIED: full validation predictions and metrics identical", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    run(parser.parse_args())
