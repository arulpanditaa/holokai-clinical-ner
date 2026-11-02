#!/usr/bin/env python3
"""Read-only validation of character annotations against GLiNER token spans."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from label_schemas import SOURCE_LABEL_MAP as LABEL_MAP, ORIGINAL_LABELS as LABELS
from label_schemas import SCHEMAS, get_labels, source_label_map, schema_metadata
FINETUNE = Path(__file__).resolve().parents[1]


def reject_constant(value):
    raise ValueError(f'Non-JSON numeric constant: {value}')


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON object key: {key}')
        result[key] = value
    return result


class Validator:
    def __init__(self, label_schema='original'):
        self.label_schema = label_schema
        self.labels = get_labels(label_schema)
        self.label_map = source_label_map(label_schema)
        self.errors = []
        self.warnings = []
        self.files = {}
        self.summaries = {}
        self.indices = {}

    def error(self, code, location, message):
        self.errors.append(dict(code=code, location=location, message=message))

    def read(self, path):
        try:
            raw = path.read_bytes()
            self.files[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            data = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_keys,
                              parse_constant=reject_constant)
            if not isinstance(data, list):
                raise ValueError('Top level must be an array of samples')
            if not data:
                self.error('empty_dataset', str(path), 'Dataset contains no samples')
            return data
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            self.error('invalid_json', str(path), str(exc))
            return None

    def validate_split(self, split, original, converted):
        counts = {'original_samples': len(original) if original is not None else None,
                  'gliner_samples': len(converted) if converted is not None else None,
                  'original_entities': 0, 'gliner_entities': 0,
                  'original_empty_ner_samples': 0, 'gliner_empty_ner_samples': 0,
                  'exactly_mapped_entities': 0}
        label_counts = {'original': Counter(), 'gliner': Counter()}
        indices = {key: defaultdict(list) for key in ('id', 'text', 'tokens', 'source')}
        source_info, gliner_info = [], []
        for i, row in enumerate(original or []):
            loc = f'{split}.json[{i}]'
            info = {'text': None, 'entities': [], 'count': None}
            source_info.append(info)
            if not isinstance(row, dict):
                self.error('sample_schema', loc, 'Sample must be an object')
                continue
            for key in ('id', 'source', 'text'):
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    self.error('sample_schema', loc, f'{key} must be a nonempty string')
                else:
                    indices[key][' '.join(value.split()) if key == 'text' else value].append(i)
            text = row.get('text')
            if isinstance(text, str):
                info['text'] = text
            entities = row.get('entities')
            if not isinstance(entities, list):
                self.error('entity_schema', loc, 'entities must be an array')
                continue
            info['count'] = len(entities)
            counts['original_entities'] += len(entities)
            counts['original_empty_ner_samples'] += not entities
            seen = set()
            spans = {}
            for j, entity in enumerate(entities):
                eloc = f'{loc}.entities[{j}]'
                if not isinstance(entity, dict):
                    self.error('entity_schema', eloc, 'Entity must be an object')
                    continue
                label = entity.get('label')
                if isinstance(label, str):
                    label_counts['original'][self.label_map.get(label, label)] += 1
                if not isinstance(label, str) or label not in self.label_map:
                    self.error('unexpected_label', eloc, f'Unexpected source label: {label!r}')
                start, end, value = entity.get('start'), entity.get('end'), entity.get('value')
                if (type(start) is not int or type(end) is not int or
                        not isinstance(text, str) or not 0 <= start < end <= len(text)):
                    self.error('character_span', eloc, 'Expected 0 <= start < end <= text length (exclusive end)')
                    continue
                if not isinstance(value, str) or not value.strip() or text[start:end] != value:
                    self.error('entity_value', eloc, 'value must be nonempty and equal text[start:end]')
                if not isinstance(label, str):
                    continue
                key = (start, end, self.label_map.get(label, label))
                if key in seen:
                    self.error('duplicate_entity', eloc, f'Duplicate entity {key}')
                seen.add(key)
                if (start, end) in spans and spans[start, end] != label:
                    self.error('conflicting_entity', eloc, 'Same span has different labels')
                spans[start, end] = label
                info['entities'].append(key)

        for i, row in enumerate(converted or []):
            suffix = '_mapping' if self.label_schema == 'recommended' else ''
            loc = f'{split}_gliner{suffix}.json[{i}]'
            info = {'tokens': None, 'entities': [], 'count': None}
            gliner_info.append(info)
            if not isinstance(row, dict):
                self.error('sample_schema', loc, 'Sample must be an object')
                continue
            tokens, entities = row.get('tokenized_text'), row.get('ner')
            if (not isinstance(tokens, list) or not tokens or
                    any(not isinstance(t, str) or not t or any(c.isspace() for c in t) for t in tokens)):
                self.error('token_schema', loc, 'tokenized_text must be a nonempty array of nonempty, whitespace-free strings')
            else:
                info['tokens'] = tokens
                indices['tokens'][tuple(tokens)].append(i)
            if not isinstance(entities, list):
                self.error('entity_schema', loc, 'ner must be an array')
                continue
            info['count'] = len(entities)
            counts['gliner_entities'] += len(entities)
            counts['gliner_empty_ner_samples'] += not entities
            seen, spans = set(), {}
            for j, entity in enumerate(entities):
                eloc = f'{loc}.ner[{j}]'
                if not isinstance(entity, list) or len(entity) != 3:
                    self.error('entity_schema', eloc, 'Expected [start_token, end_token, label]')
                    continue
                start, end, label = entity
                if isinstance(label, str):
                    label_counts['gliner'][label] += 1
                if not isinstance(label, str) or label not in self.labels:
                    self.error('unexpected_label', eloc, f'Unexpected GLiNER label: {label!r}')
                if (type(start) is not int or type(end) is not int or
                        not isinstance(tokens, list) or not 0 <= start <= end < len(tokens)):
                    self.error('token_span', eloc, 'Expected 0 <= start <= end < token count (inclusive end)')
                    continue
                if not isinstance(label, str):
                    continue
                key = (start, end, label)
                if key in seen:
                    self.error('duplicate_entity', eloc, f'Duplicate entity {key}')
                seen.add(key)
                if (start, end) in spans and spans[start, end] != label:
                    self.error('conflicting_entity', eloc, 'Same span has different labels')
                spans[start, end] = label
                info['entities'].append(key)

        if original is not None and converted is not None:
            if len(original) != len(converted):
                self.error('sample_count', split, 'Original and converted sample counts differ')
            for i, (source, target) in enumerate(zip(source_info, gliner_info)):
                loc = f'{split}[{i}]'
                if source['count'] != target['count']:
                    self.error('entity_count', loc, 'Original and converted entity counts differ')
                text, tokens = source['text'], target['tokens']
                if text is None or tokens is None:
                    continue
                # Align the supplied tokens directly; do not assume a tokenizer regex.
                offsets, cursor = [], 0
                for token in tokens:
                    while cursor < len(text) and text[cursor].isspace():
                        cursor += 1
                    if not text.startswith(token, cursor):
                        self.error('token_alignment', loc, f'Token {len(offsets)} does not match source at character {cursor}')
                        break
                    offsets.append((cursor, cursor + len(token)))
                    cursor += len(token)
                if len(offsets) != len(tokens):
                    continue
                if text[cursor:].strip():
                    self.error('token_alignment', loc, 'Tokens omit trailing source text')
                    continue
                starts = {a: j for j, (a, b) in enumerate(offsets)}
                ends = {b: j for j, (a, b) in enumerate(offsets)}
                expected = Counter()
                for start, end, label in source['entities']:
                    if start not in starts or end not in ends:
                        self.error('entity_boundary', loc, f'Character span {(start, end, label)} does not align exactly to token boundaries')
                        continue
                    expected[starts[start], ends[end], label] += 1
                actual = Counter(target['entities'])
                counts['exactly_mapped_entities'] += sum((expected & actual).values())
                if expected != actual:
                    self.error('entity_mapping', loc, f'Missing spans: {list((expected-actual).elements())}; extra spans: {list((actual-expected).elements())}')
            if counts['original_entities'] != counts['gliner_entities']:
                self.error('entity_count', split, 'Total entity counts differ')
            if label_counts['original'] != label_counts['gliner']:
                self.error('label_counts', split, 'Per-label mention counts differ')
            if counts['original_empty_ner_samples'] != counts['gliner_empty_ner_samples']:
                self.error('empty_ner_count', split, 'Empty-NER sample counts differ')
        counts['per_label_mentions'] = {
            kind: {label: values[label] for label in sorted(set(self.labels) | set(values))}
            for kind, values in label_counts.items()
        }
        counts['duplicate_sample_groups'] = {}
        for key in ('id', 'text', 'tokens'):
            groups = [rows for rows in indices[key].values() if len(rows) > 1]
            counts['duplicate_sample_groups'][key] = groups
            for rows in groups:
                self.error('duplicate_sample', split, f'Duplicate {key} at sample indices {rows}')
        self.summaries[split] = counts
        self.indices[split] = indices

    def overlap(self):
        result = {}
        for key in ('id', 'text', 'tokens', 'source'):
            train, val = self.indices['train'][key], self.indices['val'][key]
            groups = [{'train_indices': train[value], 'val_indices': val[value]}
                      for value in sorted(train.keys() & val.keys())]
            result[key] = {'shared_values': len(groups), 'groups': groups}
            if groups:
                if key == 'source':
                    self.warnings.append(dict(code='shared_source', location='train/val',
                        message=f'{len(groups)} source identifiers occur in both splits; review document-level leakage'))
                else:
                    self.error('split_overlap', 'train/val', f'{len(groups)} shared {key} values; see overlap groups')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-schema', choices=SCHEMAS, default='original')
    parser.add_argument('--data-dir', type=Path, default=FINETUNE / 'data')
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    suffix = '_mapping' if args.label_schema == 'recommended' else ''
    data_dir = args.data_dir.resolve()
    report_path = (args.report or FINETUNE / f'outputs/gliner_data_validation_report{suffix}.json').resolve()
    # An explicit output override must never overwrite input data (including symlinks).
    inputs = [data_dir / name for split in ('train', 'val')
              for name in (f'{split}.json', f'{split}_gliner{suffix}.json')]
    if (report_path.is_relative_to(data_dir) or any(
            report_path == p.resolve() or (report_path.exists() and p.exists() and report_path.samefile(p))
            for p in inputs)):
        parser.error('Report must be outside the input data directory and must not alias an input file')
    validator = Validator(args.label_schema)
    for split in ('train', 'val'):
        original = validator.read(data_dir / f'{split}.json')
        converted = validator.read(data_dir / f'{split}_gliner{suffix}.json')
        validator.validate_split(split, original, converted)
    overlap = validator.overlap()
    report = {
        'status': 'FAIL' if validator.errors else 'PASS',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'expected_labels': list(validator.labels), 'source_to_gliner_labels': validator.label_map,
        **schema_metadata(args.label_schema),
        'conventions': {'source_end': 'exclusive', 'gliner_end': 'inclusive',
                        'sample_pairing': 'array position, verified by exact token-to-text alignment',
                        'text_overlap': 'case-sensitive, whitespace-normalized',
                        'shared_source': 'warning; distinct samples may share a source document'},
        'inputs': validator.files, 'splits': validator.summaries, 'overlap': overlap,
        'critical_error_count': len(validator.errors), 'warning_count': len(validator.warnings),
        'critical_errors': validator.errors, 'warnings': validator.warnings,
    }
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    except OSError as exc:
        print(f'FAIL: cannot save report: {exc}', file=sys.stderr)
        return 1
    for split, counts in validator.summaries.items():
        print(f"{split}: samples {counts['original_samples']} -> {counts['gliner_samples']}; "
              f"entities {counts['original_entities']} -> {counts['gliner_entities']}; "
              f"empty NER {counts['original_empty_ner_samples']} -> {counts['gliner_empty_ner_samples']}")
        print('  label: original -> GLiNER')
        for label in validator.labels:
            print(f"  {label}: {counts['per_label_mentions']['original'][label]} -> {counts['per_label_mentions']['gliner'][label]}")
    print(f"Critical errors: {len(validator.errors)}; warnings: {len(validator.warnings)}")
    print(f'Report: {report_path}')
    print(report['status'])
    return 1 if validator.errors else 0


if __name__ == '__main__':
    sys.exit(main())
