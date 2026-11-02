#!/usr/bin/env python3
"""Tune post-decoding label thresholds on validation data; never train a model.

Example (run only when ready):
  python finetune/scripts/tune_per_label_thresholds.py \
    --model finetune/outputs/full_finetune_run1/final_model \
    --output-dir finetune/outputs/per_label_thresholds

One scored prediction pool is decoded using the evaluator at threshold 0.10.
All sweeps and the joint calibration filter this same pool; suppressed alternatives
are not re-decoded after filtering. Metrics are validation tuning scores, not an
independent test-set estimate. PROVIDER selection fails if recall >= 0.70 is
unattainable; the sweep CSV remains available for inspection.
"""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import traceback

from evaluate_gliner import LABELS, compute_metrics, load_data, predict, validate_span
from label_schemas import SCHEMAS, get_labels, provider_label, schema_metadata

THRESHOLDS = tuple(value / 100 for value in range(10, 91, 5))
PROVIDER_MIN_RECALL = 0.70
EXPECTED_SAMPLES = 171
OUTPUT_NAMES = ('threshold_sweep.csv', 'best_thresholds.json', 'calibrated_metrics.json',
                'calibrated_per_label_metrics.csv')
METRIC_FIELDS = ['tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'support']


def filter_predictions(predictions, thresholds, label_schema='original'):
    """Apply score >= label threshold; preserve order, empty samples and input data."""
    if set(thresholds) != set(get_labels(label_schema)):
        raise ValueError('Thresholds must contain exactly the fixed 11 labels')
    if any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1
           for t in thresholds.values()):
        raise ValueError('Each threshold must be a finite number in [0, 1]')
    filtered = []
    for row in predictions:
        kept = []
        for entity in row['predicted_entities']:
            validate_span((entity['start'], entity['end'], entity['label']), len(row['tokenized_text']), label_schema)
            score = entity['score']
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid prediction score')
            if score >= thresholds[entity['label']]:
                kept.append(dict(entity))
        filtered.append({**row, 'predicted_entities': kept})
    return filtered


def prediction_metrics(predictions, label_schema='original'):
    return compute_metrics([row['gold_ner'] for row in predictions], [
        [(entity['start'], entity['end'], entity['label']) for entity in row['predicted_entities']]
        for row in predictions], label_schema=label_schema)


def sweep_thresholds(predictions, label_schema='original'):
    labels = get_labels(label_schema)
    rows = []
    for threshold in THRESHOLDS:
        metrics = prediction_metrics(filter_predictions(predictions, dict.fromkeys(labels, threshold), label_schema), label_schema)
        for label in labels:
            rows.append({'label': label, 'threshold': threshold, **metrics['per_label'][label]})
    return rows


def select_threshold(label, rows, label_schema='original'):
    if label not in get_labels(label_schema):
        raise ValueError(f'Unexpected label {label!r} for {label_schema}')
    candidates = [row for row in rows if row['label'] == label]
    if label == provider_label(label_schema):
        candidates = [row for row in candidates if row['recall'] >= PROVIDER_MIN_RECALL]
    if not candidates:
        constraint = ' with recall >= 0.70' if label == provider_label(label_schema) else ''
        raise ValueError(f'No eligible threshold for {label}{constraint}; calibrated outputs were not produced')
    # Rounded distance gives 0.35 and 0.45 an equal distance from 0.40.
    # A final lower-threshold preference resolves any otherwise exact tie.
    return max(candidates, key=lambda row: (row['f1'], row['precision'],
               -round(abs(row['threshold'] - 0.4), 12), -row['threshold']))


def calibrate(predictions, sweep, label_schema='original'):
    selected = {label: select_threshold(label, sweep, label_schema) for label in get_labels(label_schema)}
    thresholds = {label: row['threshold'] for label, row in selected.items()}
    calibrated = filter_predictions(predictions, thresholds, label_schema)
    metrics = prediction_metrics(calibrated, label_schema)
    if metrics['per_label'][provider_label(label_schema)]['recall'] < PROVIDER_MIN_RECALL:
        raise ValueError('Calibrated PROVIDER recall does not meet the selection constraint')
    return thresholds, selected, metrics, calibrated


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def save_outputs(output, predictions, metadata, label_schema='original'):
    output = Path(output)
    if any((output / name).exists() for name in OUTPUT_NAMES):
        raise ValueError('Result files already exist; choose a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    sweep = sweep_thresholds(predictions, label_schema)
    write_csv(output / 'threshold_sweep.csv', ['label', 'threshold', *METRIC_FIELDS], sweep)
    thresholds, selected, metrics, calibrated = calibrate(predictions, sweep, label_schema)
    metadata = {**metadata, **schema_metadata(label_schema), 'threshold_grid': list(THRESHOLDS),
                'candidate_threshold': 0.10, 'filter_comparison': 'score >= threshold',
                'calibration_method': 'per-label filtering of a shared post-decoding prediction pool',
                'selection': 'highest F1, then precision, then closest to 0.4; final tie: lower threshold',
                'provider_min_recall': PROVIDER_MIN_RECALL, 'provider_label': provider_label(label_schema),
                'matching': 'exact inclusive token start/end + exact label within sample',
                'averaging': 'micro', 'zero_division': 0,
                'evaluation_scope': 'validation tuning; not independent held-out performance'}
    write_json(output / 'best_thresholds.json', {'thresholds': thresholds,
               'selected_validation_metrics': selected, 'metadata': metadata})
    write_json(output / 'calibrated_metrics.json', {**metrics, 'thresholds': thresholds, 'metadata': metadata})
    write_csv(output / 'calibrated_per_label_metrics.csv', ['label', 'threshold', *METRIC_FIELDS],
              [{'label': label, 'threshold': thresholds[label], **metrics['per_label'][label]} for label in get_labels(label_schema)])
    return metrics


def validate_output_location(output, data_path, model_name):
    if output.is_relative_to(data_path.parent):
        raise ValueError('Output must be outside the validation data directory')
    model_path = Path(model_name).expanduser()
    if model_path.exists() and output.is_relative_to(model_path.resolve()):
        raise ValueError('Output must be outside the selected model checkpoint')
    if any((output / name).exists() for name in OUTPUT_NAMES):
        raise ValueError('Result files already exist; choose a fresh output directory')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label-schema', choices=SCHEMAS, default='original')
    parser.add_argument('--model', required=True, help='Selected fine-tuned model directory or Hub ID (no base-model default)')
    parser.add_argument('--data', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--flat-ner', action='store_true', help='Match the evaluator overlap option; default allows nested spans')
    args = parser.parse_args()
    try:
        if args.batch_size < 1:
            raise ValueError('batch-size must be positive')
        suffix = '_mapping' if args.label_schema == 'recommended' else ''
        data_path = (args.data or Path(__file__).resolve().parents[1] / f'data/val_gliner{suffix}.json').resolve()
        output = args.output_dir.resolve()
        validate_output_location(output, data_path, args.model)
        data, digest = load_data(data_path, args.label_schema)
        if len(data) != EXPECTED_SAMPLES:
            raise ValueError(f'Expected the full {EXPECTED_SAMPLES}-sample validation set, got {len(data)}')
        observed = {entity[2] for row in data for entity in row['ner']}
        if observed != set(get_labels(args.label_schema)):
            raise ValueError(f'Validation data must cover all 11 labels; missing {sorted(set(get_labels(args.label_schema)) - observed)}')
        import torch
        from gliner import GLiNER

        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        model = GLiNER.from_pretrained(args.model)
        model.to(device)
        model.eval()
        with torch.inference_mode():
            predictions = predict(model, data, THRESHOLDS[0], args.batch_size, args.flat_ner, args.label_schema)
        metadata = {'model': args.model, 'data': str(data_path), 'data_sha256': digest,
                    'sample_count': len(data), 'device': device, 'batch_size': args.batch_size,
                    'flat_ner': args.flat_ner, 'multi_label': False,
                    'gliner_version': version('gliner'), 'torch_version': torch.__version__,
                    'created_at_utc': datetime.now(timezone.utc).isoformat()}
        metrics = save_outputs(output, predictions, metadata, args.label_schema)
        print(json.dumps(metrics['overall'], indent=2))
        print(f'Results: {output}')
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
