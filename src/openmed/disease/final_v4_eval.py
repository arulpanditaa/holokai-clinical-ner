"""One-time V4 evaluation of the validation-selected model. Run only after all training is finished.

Selection is decided on validation alone (highest validation micro F1 across completed groups,
earlier group in GROUPS order wins ties); V4 is then scored once and never used to choose anything.
Refuses to run again if outputs/final_v4_results.json already exists.
"""
import argparse, hashlib, json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from evaluate import PROJECT, TAGS, check, better, validate_rows, evaluate_model
from train import GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=PROJECT)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    out = project / 'outputs' / 'final_v4_results.json'
    check(not out.exists(), f'V4 was already evaluated once: {out}')

    # 1) Select on validation only.
    winner = None
    for group in GROUPS:
        path = project / 'outputs' / group / 'validation_results.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result.get('completed') and (winner is None or better(result['metrics']['overall']['f1'], winner['metrics']['overall']['f1'])):
            winner = result
    check(winner is not None, 'No completed training group found')
    model_dir = project / 'outputs' / winner['group'] / 'best_model'

    # 2) Load prepared V4 and verify it against the manifest.
    data = project / 'data'
    manifest = json.loads((data / 'preparation_manifest.json').read_text())
    raw = (data / 'prepared' / 'v4_test.jsonl').read_bytes()
    entry = manifest['splits']['v4_test']
    check(hashlib.sha256(raw).hexdigest() == entry['output_sha256'], 'v4_test: manifest hash mismatch')
    rows = [json.loads(l) for l in raw.decode('utf-8').splitlines() if l.strip()]
    flat, gold, summary = validate_rows(rows, 'v4_test', TAGS)
    check(summary['records'] == entry['records'] and summary['entities'] == entry['entities'], 'v4_test: manifest counts mismatch')

    # 3) Score once.
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    check(model.config.id2label == dict(enumerate(TAGS)), 'Saved model label schema mismatch')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    collator = DataCollatorForTokenClassification(tokenizer, padding=True, label_pad_token_id=-100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    metrics = evaluate_model(model, flat, gold, collator, device)
    result = dict(selected_group=winner['group'], selected_epoch=winner['best_epoch'],
                  validation_micro_f1=winner['metrics']['overall']['f1'],
                  v4_metrics=metrics, v4_sha256=entry['output_sha256'], model_dir=str(model_dir),
                  policy='selected on validation only; V4 scored once')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'selected_group': winner['group'], 'validation_f1': round(winner['metrics']['overall']['f1'], 4),
                      'v4_f1': round(metrics['overall']['f1'], 4)}, indent=2))


if __name__ == '__main__':
    main()
