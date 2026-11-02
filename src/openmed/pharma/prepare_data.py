#!/usr/bin/env python3
"""Prepare Drive GLiNER datasets for OpenMed; never loads model weights.

Colab setup (after mounting Drive):
    %pip install transformers tokenizers sentencepiece
    %run /content/drive/MyDrive/Holokai_project/week9/OpenMed-NER-PharmaDetect-SuperClinical-434M/prepare_data.py

Defaults read the three named Drive inputs and create prepared/*.jsonl,
label_map.json and preparation_manifest.json beside this script's intended
Drive location. Each JSONL line represents one ORIGINAL record, containing
its unchanged source object, canonical word-level BIO tags and a list of
model-ready chunks. Flatten ``chunks`` later for training; reconstruct
predictions using token_start/token_end_exclusive and record_id.

Run --self-test for synthetic integrity tests without dependencies, Drive,
network, or V4 access. No output files are created by self-tests.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


MODEL_ID = "OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M"
TOKENIZER_REVISION = "d4259d05b0e924ad786f5c342ddedf46fe956331"
INPUT_DIR = Path("/content/drive/MyDrive/Holokai_Project/week6/finetune/data")
OUTPUT_DIR = Path("/content/drive/MyDrive/Holokai_Project/week9")
MAX_LENGTH = 512
SPLITS = (
    ("train", "train_gliner_mapping.json"),
    ("validation", "val_gliner_mapping.json"),
    ("v4_test", "corpus_V4_gliner_mapping.json"),
)
# Fixed order, independent of label frequencies or V4 contents.
LABEL_MAPPING = {
    "condition": "DIAGNOSIS",
    "organization medical facility": "FACILITY",
    "gender identity": "GENDER_IDENTITY",
    "lab result or test finding": "LAB_RESULT",
    "location address": "LOCATION_ADDRESS",
    "medication, drug, or prescription": "MEDICATION",
    "name": "PERSON",
    "medical procedure or test performed": "PROCEDURE",
    "name medical professional": "PROVIDER",
    "sex assigned at birth": "SEX",
    "personal identifier number": "UNIQUE_ID",
}
BIO_LABELS = ["O"] + [f"{prefix}-{label}" for label in LABEL_MAPPING.values()
                      for prefix in ("B", "I")]
LABEL2ID = {label: index for index, label in enumerate(BIO_LABELS)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def decode_bio(tags):
    """Return inclusive canonical spans, rejecting malformed BIO sequences."""
    entities = []
    for index, tag in enumerate(tags):
        if tag == "O":
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B":
            entities.append([index, index, label])
        else:
            require(prefix == "I" and index > 0 and
                    tags[index - 1] in (f"B-{label}", f"I-{label}"),
                    f"Invalid BIO continuation at token {index}")
            entities[-1][1] = index
    return entities


def validate_record(record):
    require(isinstance(record, dict), "Record must be an object")
    tokens, annotations = record.get("tokenized_text"), record.get("ner")
    require(isinstance(tokens, list) and bool(tokens) and
            all(isinstance(t, str) and bool(t.strip()) for t in tokens),
            "tokenized_text must contain nonblank strings")
    require(isinstance(annotations, list), "ner must be a list")
    tags, seen = ["O"] * len(tokens), set()
    canonical = []
    for entity in annotations:
        require(isinstance(entity, list) and len(entity) == 3,
                "Expected [start, inclusive_end, mapped_label]")
        start, end, label = entity
        require(type(start) is int and type(end) is int and
                0 <= start <= end < len(tokens), "Invalid inclusive token span")
        require(isinstance(label, str) and label in LABEL_MAPPING,
                f"Unknown GLiNER mapped label: {label!r}")
        require(tuple(entity) not in seen, "Duplicate entity annotation")
        seen.add(tuple(entity))
        require(all(tag == "O" for tag in tags[start:end + 1]),
                "Overlapping entities cannot be represented losslessly by flat BIO")
        canonical.append([start, end, LABEL_MAPPING[label]])
        tags[start] = "B-" + LABEL_MAPPING[label]
        tags[start + 1:end + 1] = ["I-" + LABEL_MAPPING[label]] * (end - start)
    require(sorted(decode_bio(tags)) == sorted(canonical), "BIO round-trip failed")
    return tokens, tags, canonical


def encode_words(tokenizer, tokens):
    encoded = tokenizer(tokens, is_split_into_words=True, truncation=False,
                        padding=False, return_attention_mask=True,
                        return_special_tokens_mask=True, verbose=False)
    words = encoded.word_ids()
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    special = encoded["special_tokens_mask"]
    require(len(ids) == len(mask) == len(words) == len(special),
            "Tokenizer returned inconsistent lengths")
    require(all(type(w) is int and 0 <= w < len(tokens)
                for w in words if w is not None), "Invalid tokenizer word IDs")
    observed = [w for w in words if w is not None]
    require(observed == sorted(observed) and set(observed) == set(range(len(tokens))),
            "Tokenizer lost or reordered original tokens")
    require(all((w is None) == bool(s) for w, s in zip(words, special)),
            "Unexpected special-token alignment")
    return ids, mask, words


def safe_end(start, end, entities):
    for entity_start, entity_end, _ in entities:
        if entity_start < end <= entity_end:
            end = entity_start
    require(end > start,
            "A token or complete entity cannot fit within 512 positions; nothing was truncated")
    return end


def prepare_record(record, record_id, split, tokenizer):
    tokens, tags, entities = validate_record(record)
    full_ids, _, full_words = encode_words(tokenizer, tokens)
    counts = Counter(w for w in full_words if w is not None)
    special_count = len(full_ids) - sum(counts.values())
    chunks, start = [], 0
    recovered, reconstructed_tokens, reconstructed_tags = [], [], []
    while start < len(tokens):
        end, length = start, special_count
        while end < len(tokens) and length + counts[end] <= MAX_LENGTH:
            length += counts[end]
            end += 1
        end = safe_end(start, end, entities)
        # Verify the actual chunk encoding, including any context-dependent effects.
        while True:
            ids, mask, words = encode_words(tokenizer, tokens[start:end])
            if len(ids) <= MAX_LENGTH:
                break
            end = safe_end(start, end - 1, entities)
        chunk_tags = tags[start:end]
        aligned, seen_words = [], set()
        for word in words:
            if word is None or word in seen_words:
                aligned.append(-100)
            else:
                aligned.append(LABEL2ID[chunk_tags[word]])
                seen_words.add(word)
        require(len(ids) == len(mask) == len(aligned) <= MAX_LENGTH,
                "Aligned tensor lengths or sequence limit failed")
        supervised = [BIO_LABELS[tag] for tag in aligned if tag != -100]
        require(supervised == chunk_tags, "Subtoken-to-word round-trip failed")
        recovered.extend([[a + start, b + start, label]
                          for a, b, label in decode_bio(supervised)])
        reconstructed_tokens.extend(tokens[start:end])
        reconstructed_tags.extend(chunk_tags)
        chunks.append({
            "chunk_id": f"{record_id}:{len(chunks)}",
            "token_start": start, "token_end_exclusive": end,
            "tokens": tokens[start:end], "bio_tags": chunk_tags,
            "input_ids": ids, "attention_mask": mask, "labels": aligned,
            "word_ids": [None if w is None else w + start for w in words],
        })
        start = end
    require(reconstructed_tokens == tokens and reconstructed_tags == tags,
            "Chunks lost, duplicated or reordered original tokens")
    require(sorted(recovered) == sorted(entities), "Chunk entity reconstruction failed")
    return {"record_id": record_id, "split": split, "original": record,
            "bio_tags": tags, "chunks": chunks}


def prepare_split(records, split, tokenizer):
    require(isinstance(records, list) and bool(records),
            f"{split}: expected a nonempty JSON array")
    prepared, before, after = [], Counter(), Counter()
    for index, record in enumerate(records):
        record_id = f"{split}:{index}"
        try:
            result = prepare_record(record, record_id, split, tokenizer)
        except ValueError as exc:
            raise ValueError(f"{record_id}: {exc}") from exc
        require(result["original"] == record, f"{record_id}: source annotation changed")
        before.update(LABEL_MAPPING[e[2]] for e in record["ner"])
        for chunk in result["chunks"]:
            after.update(e[2] for e in decode_bio(chunk["bio_tags"]))
        prepared.append(result)
    require(before == after, f"{split}: entity counts changed")
    summary = {
        "role": "held_out_test" if split == "v4_test" else split,
        "records": len(records), "entities": sum(before.values()),
        "entity_free_records": sum(not r["ner"] for r in records),
        "chunks": sum(len(r["chunks"]) for r in prepared),
        "label_counts": {label: before[label] for label in LABEL_MAPPING.values()},
        "absent_labels": [label for label in LABEL_MAPPING.values() if not before[label]],
        "integrity_checks_passed": True,
    }
    return prepared, summary


def write_exclusive(path, content):
    with path.open("xb") as handle:
        handle.write(content)


def run(input_dir, output_dir):
    sources = [(split, input_dir / name) for split, name in SPLITS]
    targets = [output_dir / "prepared" / f"{split}.jsonl" for split, _ in SPLITS]
    targets += [output_dir / "label_map.json", output_dir / "preparation_manifest.json"]
    source_paths = {p.resolve() for _, p in sources}
    for path in targets:
        require(path.resolve() not in source_paths, f"Output conflicts with input: {path}")
        require(not path.exists() and not path.is_symlink(), f"Refusing overwrite: {path}")
    for _, path in sources:
        require(path.is_file(), f"Missing Drive input: {path}")
    try:
        import transformers
        import tokenizers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("Install dependencies in Colab: %pip install transformers tokenizers sentencepiece") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=TOKENIZER_REVISION, use_fast=True, trust_remote_code=False)
    require(tokenizer.is_fast, "A fast tokenizer with word_ids() is required")
    payloads, summaries, source_hashes = [], {}, {}
    # Settings never derive from any split. V4 is processed last for integrity only.
    for (split, source), target in zip(sources, targets[:3]):
        raw = source.read_bytes()
        records = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys,
                             parse_constant=reject_constant)
        prepared, summary = prepare_split(records, split, tokenizer)
        content = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                          for r in prepared).encode("utf-8")
        payloads.append((target, content))
        source_hashes[split] = hashlib.sha256(raw).hexdigest()
        summaries[split] = dict(summary, source=str(source),
                               source_sha256=source_hashes[split], output=str(target),
                               output_sha256=hashlib.sha256(content).hexdigest())
        print(f"{split}: integrity checks passed ({len(records)} records)")
    # Detect sources changing during a run before creating final outputs.
    for split, source in sources:
        require(hashlib.sha256(source.read_bytes()).hexdigest() == source_hashes[split],
                f"Source changed during preparation: {source}")
    label_map = {
        "mapped_to_original": LABEL_MAPPING,
        "original_to_mapped": {v: k for k, v in LABEL_MAPPING.items()},
        "label2id": LABEL2ID, "id2label": dict(enumerate(BIO_LABELS)),
    }
    manifest = {
        "schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {"model_id": MODEL_ID, "revision": TOKENIZER_REVISION,
                      "class": type(tokenizer).__name__,
                      "transformers_version": transformers.__version__,
                      "tokenizers_version": tokenizers.__version__},
        "settings": {"max_length": MAX_LENGTH, "includes_special_tokens": True,
                     "subtoken_supervision": "first_only", "ignore_index": -100,
                     "padding": False, "truncation": False,
                     "chunking": "nonoverlapping_token_boundaries_move_before_entities",
                     "chunking_uses_gold_entity_boundaries": True,
                     "overlaps_duplicates_unknown_labels": "reject",
                     "oversize_entity_or_token": "reject_without_truncation",
                     "v4_policy": "held_out_integrity_only_no_decision_tuning"},
        "output_layout": "one_original_record_per_line_with_nested_model_ready_chunks",
        "splits": summaries,
        "completion_policy": "Outputs are usable together only when this manifest exists",
    }
    for path, obj in zip(targets[3:], (label_map, manifest)):
        payloads.append((path, (json.dumps(obj, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n").encode("utf-8")))
    # Validate every split before writing; exclusive creation prevents overwrites.
    # Manifest is written last. Interrupted/partial outputs are never auto-replaced.
    for path, content in payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_exclusive(path, content)
        require(path.read_bytes() == content, f"Output read-back verification failed: {path}")
    print(f"Prepared outputs verified: {output_dir}")


def self_test():
    """Synthetic tests only; deliberately never discovers or opens real datasets."""
    import copy
    import unittest

    class Encoding(dict):
        def word_ids(self):
            return self.words

    class FakeTokenizer:
        def __call__(self, words, **kwargs):
            ids, alignment, special = [1], [None], [1]
            for index, word in enumerate(words):
                pieces = 511 if word == "OVERSIZE" else 2
                ids.extend([10] * pieces)
                alignment.extend([index] * pieces)
                special.extend([0] * pieces)
            ids.append(2)
            alignment.append(None)
            special.append(1)
            result = Encoding(input_ids=ids, attention_mask=[1] * len(ids),
                              special_tokens_mask=special)
            result.words = alignment
            return result

    class Tests(unittest.TestCase):
        def test_all_labels_and_roundtrip(self):
            annotations = [[i, i, label] for i, label in enumerate(LABEL_MAPPING)]
            record = {"tokenized_text": ["word"] * 11, "ner": annotations,
                      "metadata": {"preserved": True}}
            snapshot = copy.deepcopy(record)
            for split, _ in SPLITS:
                result, summary = prepare_split([record], split, FakeTokenizer())
                self.assertEqual(result[0]["original"], snapshot)
                self.assertEqual(record, snapshot)
                self.assertEqual(summary["entities"], 11)
                self.assertEqual(len(result[0]["chunks"][0]["labels"]), 24)
                self.assertEqual(set(summary["label_counts"].values()), {1})

        def test_entity_boundary_and_empty_record(self):
            label = next(iter(LABEL_MAPPING))
            record = {"tokenized_text": ["word"] * 300, "ner": [[250, 260, label]]}
            result = prepare_record(record, "train:0", "train", FakeTokenizer())
            self.assertEqual([c["token_start"] for c in result["chunks"]], [0, 250])
            self.assertEqual(result["chunks"][0]["token_end_exclusive"], 250)
            empty = prepare_record(dict(record, ner=[]), "train:1", "train", FakeTokenizer())
            self.assertTrue(all(t == "O" for t in empty["bio_tags"]))
            exact = prepare_record({"tokenized_text": ["word"] * 255, "ner": []},
                                   "train:2", "train", FakeTokenizer())
            self.assertEqual(len(exact["chunks"][0]["input_ids"]), 512)

        def test_rejections(self):
            label = next(iter(LABEL_MAPPING))
            for ner in ([[0, 0, label], [0, 0, label]],
                        [[0, 1, label], [1, 1, label]], [[0, 2, label]],
                        [[True, 1, label]], [[0, 0, "unknown"]]):
                with self.subTest(ner=ner), self.assertRaises(ValueError):
                    validate_record({"tokenized_text": ["a", "b"], "ner": ner})
            for tokens, ner in ((["OVERSIZE"], []),
                                (["word"] * 256, [[0, 255, label]])):
                with self.assertRaises(ValueError):
                    prepare_record({"tokenized_text": tokens, "ner": ner},
                                   "train:0", "train", FakeTokenizer())

        def test_exclusive_outputs(self):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "output.json"
                write_exclusive(path, b"original")
                with self.assertRaises(FileExistsError):
                    write_exclusive(path, b"replacement")
                self.assertEqual(path.read_bytes(), b"original")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    require(unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(),
            "Synthetic integrity tests failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            run(args.input_dir, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
