#!/usr/bin/env python3
"""Create recommended-label copies of the three GLiNER datasets, without models.

Run from any directory: python /path/to/finetune/scripts/map_recommended_labels.py
Defaults to finetune/data for both input and output. Existing outputs are refused.
Only each ner entry's label (index 2) changes; JSON formatting may be normalized.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from label_schemas import RECOMMENDED_LABEL_MAPPING as LABEL_MAPPING

INPUT_NAMES = ('train_gliner.json', 'val_gliner.json', 'corpus_V4_gliner.json')


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f'Invalid JSON constant: {value}')


def map_records(records):
    if len(LABEL_MAPPING) != 11 or len(set(LABEL_MAPPING.values())) != 11:
        raise ValueError('The label mapping must be one-to-one and contain 11 labels')
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty JSON array of samples')
    mapped = deepcopy(records)
    for index, row in enumerate(mapped):
        if not isinstance(row, dict):
            raise ValueError(f'Sample {index} must be an object')
        tokens, entities = row.get('tokenized_text'), row.get('ner')
        if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t for t in tokens):
            raise ValueError(f'Sample {index} has invalid tokenized_text')
        if not isinstance(entities, list):
            raise ValueError(f'Sample {index} has invalid ner')
        for entity_index, entity in enumerate(entities):
            where = f'Sample {index}, entity {entity_index}'
            if not isinstance(entity, list) or len(entity) != 3:
                raise ValueError(f'{where}: expected [start, end, label]')
            start, end, label = entity
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(tokens):
                raise ValueError(f'{where}: invalid inclusive token span')
            if not isinstance(label, str) or label not in LABEL_MAPPING:
                raise ValueError(f'{where}: unexpected source label {label!r}')
            entity[2] = LABEL_MAPPING[label]
    return mapped


def validate_mapping(original, mapped):
    """Prove complete label coverage and reversibility of all non-label content."""
    if len(original) != len(mapped):
        raise ValueError('Sample count changed')
    before = Counter(e[2] for row in original for e in row['ner'])
    after = Counter(e[2] for row in mapped for e in row['ner'])
    if set(before) != set(LABEL_MAPPING):
        raise ValueError(f'Input must cover all 11 labels; missing {sorted(set(LABEL_MAPPING) - set(before))}')
    if set(after) != set(LABEL_MAPPING.values()):
        raise ValueError('Output contains missing or unexpected recommended labels')
    if after != Counter({LABEL_MAPPING[label]: count for label, count in before.items()}):
        raise ValueError('Per-label entity counts changed')
    inverse = {value: key for key, value in LABEL_MAPPING.items()}
    restored = deepcopy(mapped)
    for source, row in zip(original, restored):
        if len(source['ner']) != len(row['ner']):
            raise ValueError('Per-sample entity count changed')
        for entity in row['ner']:
            entity[2] = inverse[entity[2]]
    if original != restored:
        raise ValueError('Content other than NER labels changed (tokens, spans, order or metadata)')
    return {'samples': len(original), 'entities': sum(before.values()),
            'empty_ner_samples': sum(not row['ner'] for row in original),
            'source_label_counts': dict(sorted(before.items())),
            'mapped_label_counts': dict(sorted(after.items()))}


def generate(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    pairs = [(data_dir / name, output_dir / f'{Path(name).stem}_mapping.json') for name in INPUT_NAMES]
    for source, target in pairs:
        if target.exists() or target.is_symlink():
            raise ValueError(f'Output already exists; refusing to overwrite: {target}')
        if target.resolve() in {p.resolve() for p, _ in pairs}:
            raise ValueError('Output must not replace an input')
    prepared = []
    # Validate every input before creating any output files.
    for source, target in pairs:
        raw = source.read_bytes()
        original = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_keys, parse_constant=reject_constant)
        mapped = map_records(original)
        summary = validate_mapping(original, mapped)
        summary.update(input=str(source), output=str(target), source_sha256=hashlib.sha256(raw).hexdigest())
        prepared.append((source, target, raw, original, mapped, summary))
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for source, target, raw, original, mapped, summary in prepared:
        with target.open('x', encoding='utf-8') as handle:
            json.dump(mapped, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        saved = target.read_bytes()
        validate_mapping(original, json.loads(saved))
        summary['output_sha256'] = hashlib.sha256(saved).hexdigest()
        summaries.append(summary)
    for source, _, raw, _, _, _ in prepared:
        if source.read_bytes() != raw:
            raise ValueError(f'Original file changed during generation: {source}')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--output-dir', type=Path, help='Default: same directory as the input files')
    args = parser.parse_args()
    try:
        summaries = generate(args.data_dir, args.output_dir or args.data_dir)
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        print('PASS: only NER label strings changed; all 11 labels covered in every output; originals unchanged')
        return 0
    except (OSError, ValueError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
