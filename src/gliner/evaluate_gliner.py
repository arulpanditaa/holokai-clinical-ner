#!/usr/bin/env python3
"""Evaluate GLiNER with exact inclusive token-span and label matching."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

from validate_gliner_data import reject_constant, unique_keys
from label_schemas import ORIGINAL_LABELS as LABELS, SCHEMAS, get_labels, schema_metadata, check_model_schema


def threshold_value(value):
    threshold = float(value)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError('threshold must be finite and between 0 and 1')
    return threshold


def validate_span(span, token_count, label_schema='original'):
    labels = get_labels(label_schema)
    if not isinstance(span, (list, tuple)) or len(span) != 3:
        raise ValueError(f'Expected [start, end, label], got {span!r}')
    start, end, label = span
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < token_count:
        raise ValueError(f'Invalid inclusive token span: {span!r}')
    if not isinstance(label, str) or label not in labels:
        raise ValueError(f'Unexpected label: {label!r}')
    return start, end, label


def load_data(path, label_schema='original'):
    get_labels(label_schema)
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_keys,
                      parse_constant=reject_constant)
    if not isinstance(data, list) or not data:
        raise ValueError('Data must be a nonempty JSON array')
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f'Sample {i}: expected an object')
        tokens, ner = row.get('tokenized_text'), row.get('ner')
        if (not isinstance(tokens, list) or not tokens or
                any(not isinstance(t, str) or not t or any(c.isspace() for c in t) for t in tokens)):
            raise ValueError(f'Sample {i}: invalid tokenized_text')
        if not isinstance(ner, list):
            raise ValueError(f'Sample {i}: ner must be an array')
        spans = [validate_span(span, len(tokens), label_schema) for span in ner]
        if len(set(spans)) != len(spans):
            raise ValueError(f'Sample {i}: duplicate gold entities')
    return data, hashlib.sha256(raw).hexdigest()


def scores(tp, fp, fn):
    return {'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'tp': tp, 'fp': fp, 'fn': fn, 'support': tp + fn}


def compute_metrics(gold_samples, predicted_samples, label_schema='original'):
    """Match within each sample only; extra duplicate predictions count as FP."""
    labels = get_labels(label_schema)
    if len(gold_samples) != len(predicted_samples):
        raise ValueError('Gold and prediction sample counts differ')
    totals = {label: Counter(tp=0, fp=0, fn=0) for label in labels}
    for gold, predicted in zip(gold_samples, predicted_samples):
        for span in list(gold) + list(predicted):
            # Bounds are checked at input/inference boundaries; check shape here too.
            if not isinstance(span, (tuple, list)) or len(span) != 3:
                raise ValueError('Metric entities must be triples')
            validate_span(span, span[1] + 1 if type(span[1]) is int else 0, label_schema)
        gold_count, pred_count = Counter(map(tuple, gold)), Counter(map(tuple, predicted))
        if any(count > 1 for count in gold_count.values()):
            raise ValueError('Duplicate gold entities')
        for name, entities in [('tp', gold_count & pred_count),
                               ('fp', pred_count - gold_count), ('fn', gold_count - pred_count)]:
            for (_, _, label), count in entities.items():
                totals[label][name] += count
    per_label = {label: scores(**dict(counts)) for label, counts in totals.items()}
    overall = scores(**{name: sum(c[name] for c in totals.values()) for name in ('tp', 'fp', 'fn')})
    return {'overall': overall, 'per_label': per_label}


def predict(model, data, threshold, batch_size, flat_ner=False, label_schema='original'):
    """Use GLiNER 0.2.29's native token API, without exposing gold annotations."""
    labels = get_labels(label_schema)
    check_model_schema(model, label_schema)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    threshold_value(threshold)
    collator = model.create_collator()
    predictions = []
    for offset in range(0, len(data), batch_size):
        rows = data[offset:offset + batch_size]
        inputs = [{'tokenized_text': row['tokenized_text'][:], 'ner': []} for row in rows]
        batch = model.collate_batch(inputs, list(labels), collator)
        if batch.get('tokens') != [row['tokenized_text'] for row in rows]:
            raise ValueError(f'Tokens changed or were truncated in batch starting at sample {offset}')
        output = model.run_batch(batch, threshold=threshold, move_to_device=True)
        decoded = model.decode_batch(output, batch, threshold=threshold,
                                     flat_ner=flat_ner, multi_label=False)
        if len(decoded) != len(rows):
            raise ValueError('Model returned the wrong number of prediction samples')
        for index, (row, entities) in enumerate(zip(rows, decoded), start=offset):
            saved = []
            for entity in entities:
                # GLiNER 0.2.29 returns Span objects; older checkpoints may use
                # the legacy four-element decoder representation.
                if all(hasattr(entity, name) for name in ('start', 'end', 'entity_type', 'score')):
                    span = (entity.start, entity.end, entity.entity_type)
                    score = entity.score
                elif isinstance(entity, (tuple, list)) and len(entity) == 4:
                    span, score = entity[:3], entity[3]
                else:
                    raise ValueError(f'Sample {index}: unsupported decoder entity format')
                start, end, label = validate_span(span, len(row['tokenized_text']), label_schema)
                confidence = float(score)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError(f'Sample {index}: invalid confidence')
                # The native decoder applies its own threshold; retain no lower scores.
                if confidence < threshold:
                    continue
                saved.append({'start': start, 'end': end, 'label': label, 'score': confidence})
            predictions.append({'sample_index': index, 'tokenized_text': row['tokenized_text'],
                                'gold_ner': row['ner'], 'predicted_entities': saved})
        print(f'Evaluated {len(predictions)}/{len(data)} samples', file=sys.stderr, flush=True)
    return predictions


def save_results(output_dir, metrics, predictions):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'per_label_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['label', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'support'])
        writer.writeheader()
        for label, values in metrics['per_label'].items():
            writer.writerow({'label': label, **values})
    (output_dir / 'predictions.json').write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-schema', choices=SCHEMAS, default='original')
    parser.add_argument('--model', required=True, help='Hugging Face model ID or local GLiNER checkpoint directory')
    parser.add_argument('--data', type=Path, required=True, help='Path to val_gliner.json')
    parser.add_argument('--threshold', type=threshold_value, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--flat-ner', action='store_true', help='Suppress overlapping spans (default allows nested spans)')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    try:
        data_path, output_dir = args.data.resolve(), args.output_dir.resolve()
        targets = [output_dir / name for name in ('metrics.json', 'per_label_metrics.csv', 'predictions.json')]
        for target in targets:
            if target.exists():
                raise ValueError(f'Output already exists; choose a fresh output directory: {target}')
            if target.resolve() == data_path:
                raise ValueError('Output must not overwrite input data')
        data, digest = load_data(data_path, args.label_schema)
        import torch
        from gliner import GLiNER

        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        model = GLiNER.from_pretrained(args.model)
        model.to(device)
        model.eval()
        with torch.inference_mode():
            predictions = predict(model, data, args.threshold, args.batch_size, args.flat_ner, args.label_schema)
        metrics = compute_metrics([row['ner'] for row in data], [
            [(e['start'], e['end'], e['label']) for e in row['predicted_entities']] for row in predictions], label_schema=args.label_schema)
        metrics['metadata'] = {
            'model': args.model, 'data': str(data_path), 'data_sha256': digest,
            'threshold': args.threshold, **schema_metadata(args.label_schema), 'sample_count': len(data),
            'device': device, 'batch_size': args.batch_size, 'flat_ner': args.flat_ner,
            'multi_label': False, 'matching': 'exact inclusive token start/end + exact label within sample',
            'averaging': 'micro', 'zero_division': 0, 'duplicate_predictions': 'extra occurrences count as FP',
            'gliner_version': version('gliner'), 'torch_version': torch.__version__,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        save_results(output_dir, metrics, predictions)
        print(json.dumps(metrics['overall'], indent=2))
        print(f'Results: {output_dir}')
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
