"""Export the selected frozen encoder plus NER head and verify ONNX Runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from train_eval import TAGS, decode, load_split, tokenize, write_json


class NER(nn.Module):
    def __init__(self, model_id, revision, state):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=torch.float32,
                                                 attn_implementation="eager", reference_compile=False)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(TAGS))
        self.head.load_state_dict(state)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        return self.head(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)


def export(args):
    output, scratch = Path(args.output), Path(args.scratch)
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert not metrics["onnx"]["verified"], "Export already verified; refusing to overwrite"
    checkpoint = torch.load(scratch / "best_head.pt", map_location="cpu", weights_only=True)
    assert checkpoint["tags"] == TAGS and checkpoint["epoch"] == metrics["selected_epoch"]
    assert checkpoint["revision"] == metrics["revision"]
    tokenizer = AutoTokenizer.from_pretrained(output, use_fast=True)
    model = NER(checkpoint["model_id"], checkpoint["revision"], checkpoint["head"]).eval()
    onnx_path = output / "model.onnx"
    dummy = tokenizer(["Example clinical note.", "Blood pressure 120/80."], padding=True,
                      return_tensors="pt")
    # Only standard ONNX operators: no FlashAttention, compiled kernels, or quantization.
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
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    assert [x.name for x in session.get_inputs()] == ["input_ids", "attention_mask"]
    rows, manifest = load_split(args.data, "validation")
    assert manifest["sha256"] == metrics["data"]["validation"]["sha256"]
    encoded = tokenize(tokenizer, rows)
    order = sorted(range(len(rows)), key=lambda i: len(encoded[i]["input_ids"]))
    cases = [[order[0]], [order[0], order[-1]], [order[len(order)//2], order[-2], order[1]]]
    checks = []
    stored = json.loads((output / "predictions.json").read_text())["validation"]
    with torch.no_grad():
        for indices in cases:
            batch = tokenizer.pad([{k: encoded[i][k] for k in ("input_ids", "attention_mask")}
                                   for i in indices], padding=True, return_tensors="pt")
            reference = model(batch["input_ids"], batch["attention_mask"]).numpy()
            actual = session.run(["logits"], {k: v.numpy().astype(np.int64) for k, v in batch.items()
                                              if k in ("input_ids", "attention_mask")})[0]
            assert actual.shape == reference.shape and np.isfinite(actual).all()
            mask = batch["attention_mask"].numpy().astype(bool)
            np.testing.assert_allclose(actual[mask], reference[mask], rtol=1e-3, atol=1e-3)
            for j, index in enumerate(indices):
                positions = encoded[index]["first"]
                pt_ids, ort_ids = reference[j, positions].argmax(-1), actual[j, positions].argmax(-1)
                assert np.array_equal(pt_ids, ort_ids), "ONNX word-level tag mismatch"
                entities = decode(ort_ids.tolist())
                assert entities == stored[index]["predicted_ner"], "ONNX differs from evaluated checkpoint"
            checks.append({"batch_size": len(indices), "sequence_length": int(actual.shape[1]),
                           "records": indices, "max_abs_error": float(np.max(np.abs(actual[mask] - reference[mask]))),
                           "tags_and_entities_identical": True})
            print(f"ONNX verification: {checks[-1]}", flush=True)
    write_json(output / "inference_config.json", {
        "model_id": checkpoint["model_id"], "revision": checkpoint["revision"], "tags": TAGS,
        "inputs": {"input_ids": "int64 [batch, sequence]", "attention_mask": "int64 [batch, sequence]"},
        "output": "float32 logits [batch, sequence, 23]", "max_sequence_length": 8192,
        "preprocessing": "AutoTokenizer.from_pretrained(this_directory); is_split_into_words=True; no truncation",
        "postprocessing": "Take first subtoken per original word, argmax; train_eval.decode; inclusive token offsets",
        "orphan_I_policy": "start a new entity", "onnx_opset": 17})
    metrics["onnx"] = {"verified": True, "opset": 17, "provider": "CPUExecutionProvider",
                       "precision": "float32", "atol": 1e-3, "rtol": 1e-3, "checks": checks,
                       "sha256": hashlib.file_digest(onnx_path.open("rb"), "sha256").hexdigest(),
                       "bytes": onnx_path.stat().st_size}
    # Read the final graph from Drive again before declaring completion.
    assert onnx_path.stat().st_size > 100_000_000
    write_json(metrics_path, metrics)
    (scratch / "best_head.pt").unlink()
    print(f"ONNX_EXPORT_VERIFIED {checkpoint['model_id']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch", required=True)
    export(parser.parse_args())
