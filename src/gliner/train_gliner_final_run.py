#!/usr/bin/env python3
"""Config-driven GLiNER training with exact-span validation and model exports."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback

from evaluate_gliner import compute_metrics, load_data, predict
from label_schemas import get_labels, schema_metadata
from validate_gliner_data import reject_constant, unique_keys

BASE_MODEL = 'knowledgator/gliner-pii-base-v1.0'
REQUIRED = {
    'train_data', 'val_data', 'output_dir', 'learning_rate', 'batch_size',
    'eval_batch_size', 'epochs', 'warmup_ratio', 'weight_decay', 'seed',
    'eval_steps', 'logging_steps', 'threshold', 'flat_ner',
    'gradient_accumulation_steps', 'max_grad_norm', 'adam_beta1', 'adam_beta2',
    'adam_epsilon', 'lr_scheduler_type', 'loss_reduction', 'device',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=unique_keys, parse_constant=reject_constant)


def write_json(path, value):
    """Replace summaries atomically so an interruption cannot leave partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError('Config must be a JSON object')
    missing, extra = REQUIRED - config.keys(), config.keys() - REQUIRED - {'base_model', 'label_schema', 'save_best_model'}
    if missing or extra:
        raise ValueError(f'Missing config fields: {sorted(missing)}; unknown fields: {sorted(extra)}')
    config.setdefault('base_model', BASE_MODEL)
    config.setdefault('label_schema', 'original')
    if type(config.get('save_best_model', False)) is not bool:
        raise ValueError('save_best_model must be boolean')
    get_labels(config['label_schema'])
    for key in ('batch_size', 'eval_batch_size', 'epochs', 'eval_steps',
                'logging_steps', 'gradient_accumulation_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    for key in ('learning_rate', 'warmup_ratio', 'weight_decay', 'threshold', 'max_grad_norm',
                'adam_beta1', 'adam_beta2', 'adam_epsilon'):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        if key in ('learning_rate', 'adam_epsilon', 'max_grad_norm') and value <= 0:
            raise ValueError(f'{key} must be positive')
        if key in ('warmup_ratio', 'threshold') and not 0 <= value <= 1:
            raise ValueError(f'{key} must be in [0, 1]')
        if key in ('adam_beta1', 'adam_beta2') and not 0 <= value < 1:
            raise ValueError(f'{key} must be in [0, 1)')
        if key == 'weight_decay' and value < 0:
            raise ValueError('weight_decay must be nonnegative')
    if type(config['flat_ner']) is not bool:
        raise ValueError('flat_ner must be boolean')
    for key, choices in {'device': ('auto', 'cpu', 'cuda'),
                         'lr_scheduler_type': ('linear', 'cosine', 'constant_with_warmup'),
                         'loss_reduction': ('mean', 'sum')}.items():
        if config[key] not in choices:
            raise ValueError(f'{key} must be one of {choices}')
    for key in ('train_data', 'val_data', 'output_dir', 'base_model'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    for key in ('train_data', 'val_data', 'output_dir'):
        config[key] = str((path.parent / config[key]).resolve())
    if config['base_model'].startswith(('.', '/', '~')):
        config['base_model'] = str((path.parent / Path(config['base_model']).expanduser()).resolve())
    output = Path(config['output_dir'])
    for key in ('train_data', 'val_data'):
        source = Path(config[key])
        if output == source or output.is_relative_to(source.parent):
            raise ValueError('output_dir must be outside the input data directories')
    if config['train_data'] == config['val_data']:
        raise ValueError('Training and validation must be different files')
    return config


def output_has_run_artifacts(config):
    """Allow a prepared input data/ folder, but never silently reuse run artifacts."""
    output = Path(config['output_dir'])
    if not output.exists():
        return False
    if not output.is_dir():
        raise ValueError('output_dir must be a directory')
    input_dirs = {Path(config[key]).resolve().parent for key in ('train_data', 'val_data')}
    for entry in output.iterdir():
        if entry.name == 'data' and entry.is_dir() and entry.resolve() in input_dirs:
            continue
        return True
    return False


def prepare_data(config):
    train, train_hash = load_data(Path(config['train_data']), config.get('label_schema', 'original'))
    val, val_hash = load_data(Path(config['val_data']), config.get('label_schema', 'original'))
    if {tuple(r['tokenized_text']) for r in train} & {tuple(r['tokenized_text']) for r in val}:
        raise ValueError('Training/validation token sequences overlap')
    manifest = {'train': {'sha256': train_hash, 'samples': len(train)},
                'val': {'sha256': val_hash, 'samples': len(val)},
                **schema_metadata(config.get('label_schema', 'original'))}
    return train, val, manifest


def save_history(output, state):
    history = state.log_history
    write_json(Path(output) / 'training_history.json', history)
    write_json(Path(output) / 'train_loss.json', [row for row in history if 'loss' in row or 'train_loss' in row])
    write_json(Path(output) / 'validation_metrics.json', [row for row in history if 'eval_exact_metrics' in row])


def validate_checkpoint(path, require_complete=True):
    """Only accept full single-process Trainer checkpoints, never epoch exports."""
    path = Path(path)
    required = ['gliner_config.json', 'model.safetensors', 'tokenizer_config.json',
                'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']
    if require_complete:
        required.append('checkpoint_complete.json')
    missing = [name for name in required if not (path / name).is_file() or not (path / name).stat().st_size]
    if missing:
        raise ValueError(f'Incomplete checkpoint {path}: {missing}')
    state = read_json(path / 'trainer_state.json')
    if path.name != f"checkpoint-{state['global_step']}":
        raise ValueError(f'Checkpoint step mismatch: {path}')
    return state


def resolve_resume(config, manifest, requested):
    output = Path(config['output_dir'])
    if not config.get('save_best_model', False):
        raise ValueError('Resume requires save_best_model=true')
    if read_json(output / 'training_config.json') != config or read_json(output / 'data_manifest.json') != manifest:
        raise ValueError('Resume config or dataset hashes differ from the original run')
    if (output / 'best_model').exists():
        raise ValueError('Best model already exported; reload it independently instead of resuming')
    checkpoints = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    complete = []
    for checkpoint in checkpoints:
        try:
            validate_checkpoint(checkpoint)
        except ValueError:
            continue
        complete.append(checkpoint.resolve())
    if not complete:
        raise ValueError('No complete resumable checkpoint found')
    chosen = complete[-1] if requested == 'latest' else Path(requested).resolve()
    if chosen != complete[-1]:
        raise ValueError('Resume from the latest complete checkpoint to preserve run history')
    state = validate_checkpoint(chosen)
    if state.get('best_model_checkpoint'):
        validate_checkpoint(state['best_model_checkpoint'])
    return str(chosen)


def export_best_model(trainer, output, config):
    """Trainer has restored the best weights; publish a standalone GLiNER directory."""
    output = Path(output)
    checkpoint = trainer.state.best_model_checkpoint
    if not checkpoint:
        raise ValueError('No best validation checkpoint was selected')
    validate_checkpoint(checkpoint)
    destination = output / 'best_model'
    if destination.exists():
        raise ValueError(f'Refusing to overwrite model export: {destination}')
    staging = Path(tempfile.mkdtemp(prefix='.best-model-', dir=output))
    trainer.model.save_pretrained(staging, safe_serialization=True)
    # Keep inference settings and provenance beside the native weights/tokenizer.
    selection = {'checkpoint': checkpoint, 'validation_f1': trainer.state.best_metric,
                 'selection': 'highest validation micro F1; earliest checkpoint wins ties',
                 'threshold': config['threshold'], 'flat_ner': config['flat_ner'],
                 **schema_metadata(config['label_schema'])}
    write_json(staging / 'selection.json', selection)
    staging.rename(destination)
    write_json(output / 'best_model.json', {**selection, 'path': str(destination)})
    return destination


def build_training_collator(model):
    collator = model.data_collator_class(model.config, data_processor=model.data_processor,
                                        prepare_labels=True, return_tokens=True)

    def collate(rows):
        # Copy lists because processors may modify their input during preparation.
        inputs = [{'tokenized_text': r['tokenized_text'][:], 'ner': [e[:] for e in r['ner']]} for r in rows]
        # Training requires per-sample class mappings inferred from these examples.
        batch = collator(inputs)
        if batch.get('tokens') != [r['tokenized_text'] for r in rows]:
            raise ValueError('Training tokens were changed or truncated')
        return batch

    return collate


def save_epoch_snapshot(model, output_dir, epoch):
    """Export only completed epochs, using GLiNER's reloadable model format."""
    if epoch is None or epoch < 1 or not math.isclose(epoch, round(epoch), abs_tol=1e-6):
        return None
    destination = Path(output_dir) / f'epoch-{round(epoch)}'
    model.save_pretrained(destination, safe_serialization=True)
    return destination


def build_trainer(model, config, train, val):
    # Deferred imports keep config tests and --check-config lightweight.
    import torch
    from transformers import Trainer as HFTrainer, TrainerCallback
    from gliner.training import Trainer, TrainingArguments

    class HistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            save_history(args.output_dir, state)

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if args.should_save and not config.get('save_best_model', False):
                save_epoch_snapshot(model, args.output_dir, state.epoch)

        def on_step_end(self, args, state, control, **kwargs):
            # Include the last partial interval in selection before best-weight reload.
            if config.get('save_best_model', False) and state.global_step >= state.max_steps:
                control.should_evaluate = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            validate_checkpoint(checkpoint, require_complete=False)
            write_json(checkpoint / 'checkpoint_complete.json', {'global_step': state.global_step})

    class ExactTrainer(Trainer):
        # The GLiNER override skips OOM batches. Standard Trainer fails loudly.
        training_step = HFTrainer.training_step

        def _restore_native_weights(self, checkpoint, model):
            # GLiNER serializes the inner module without the wrapper's `model.` prefix.
            # Do not rely on the installed GLiNER Trainer's version-specific loader.
            from gliner import GLiNER
            restored = GLiNER.from_pretrained(str(checkpoint), local_files_only=True)
            self.accelerator.unwrap_model(model).load_state_dict(restored.state_dict(), strict=True)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            self._restore_native_weights(resume_from_checkpoint, self.model if model is None else model)

        def _load_best_model(self):
            self._restore_native_weights(self.state.best_model_checkpoint, self.model)

        def _save_rng_state(self, output_dir):
            # Single-process CPU/CUDA state in a weights_only-compatible format.
            # NumPy arrays in HF's default RNG pickle cannot reload on torch 2.2.
            import random
            import numpy as np
            numpy_state = list(np.random.get_state())
            numpy_state[1] = numpy_state[1].tolist()
            states = {'python': random.getstate(), 'numpy': tuple(numpy_state),
                      'cpu': torch.random.get_rng_state()}
            if torch.cuda.is_available():
                states['cuda'] = torch.cuda.random.get_rng_state_all()
            torch.save(states, Path(output_dir) / 'rng_state.pth')

        def compute_loss(self, *args, **kwargs):
            loss = super().compute_loss(*args, **kwargs)
            checked = loss[0] if isinstance(loss, tuple) else loss
            if checked is None or not torch.isfinite(checked).all():
                raise ValueError('Training loss is missing or non-finite')
            return loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
            dataset = val if eval_dataset is None else eval_dataset
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.inference_mode():
                    predictions = predict(self.model, dataset, config['threshold'],
                                          config['eval_batch_size'], config['flat_ner'], config.get('label_schema', 'original'))
                exact = compute_metrics([r['ner'] for r in dataset], [
                    [(e['start'], e['end'], e['label']) for e in r['predicted_entities']]
                    for r in predictions], label_schema=config.get('label_schema', 'original'))
            finally:
                self.model.train(was_training)
            metrics = {f'{metric_key_prefix}_{key}': value for key, value in exact['overall'].items()}
            # Full per-label metrics remain in the run-level training history.
            metrics[f'{metric_key_prefix}_exact_metrics'] = exact
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    training_args = TrainingArguments(
        output_dir=config['output_dir'], learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'], per_device_eval_batch_size=config['eval_batch_size'],
        num_train_epochs=config['epochs'], warmup_ratio=config['warmup_ratio'],
        weight_decay=config['weight_decay'], seed=config['seed'], data_seed=config['seed'],
        eval_strategy='steps', eval_steps=config['eval_steps'],
        save_strategy='steps' if config.get('save_best_model', False) else 'no',
        save_steps=config['eval_steps'], save_total_limit=2,
        load_best_model_at_end=config.get('save_best_model', False),
        metric_for_best_model='eval_f1' if config.get('save_best_model', False) else None,
        greater_is_better=True, save_only_model=False,
        logging_strategy='steps', logging_steps=config['logging_steps'], logging_first_step=True,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        max_grad_norm=config['max_grad_norm'], adam_beta1=config['adam_beta1'],
        adam_beta2=config['adam_beta2'], adam_epsilon=config['adam_epsilon'],
        lr_scheduler_type=config['lr_scheduler_type'], loss_reduction=config['loss_reduction'],
        optim='adamw_torch', report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_pin_memory=False, save_safetensors=True,
        use_cpu=config['device'] == 'cpu' or (config['device'] == 'auto' and not torch.cuda.is_available()),
        fp16=False, bf16=False,
        logging_nan_inf_filter=False, disable_tqdm=True,
    )
    trainer = ExactTrainer(model=model, args=training_args, train_dataset=train,
                           eval_dataset=val, data_collator=build_training_collator(model), callbacks=[HistoryCallback()])
    # GLiNER loss does not consume HF's num_items_in_batch normalization.
    trainer.model_accepts_loss_kwargs = False
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--check-config', action='store_true', help='Validate config/data without loading a model')
    parser.add_argument('--resume-from-checkpoint', help='latest or the latest complete checkpoint path')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        train, val, manifest = prepare_data(config)
        output = Path(config['output_dir'])
        resume = resolve_resume(config, manifest, args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        if not resume and output_has_run_artifacts(config):
            raise ValueError('Use an output_dir without run artifacts (an input data/ folder is allowed)')
        if args.check_config:
            print(json.dumps({'config': config, 'data': manifest}, indent=2))
            print('PASS: configuration and data checks; no model loaded or training run')
            return 0
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            raise ValueError('This pipeline supports a single training process only')
        if sys.version_info.releaselevel != 'final':
            raise ValueError('Training requires a stable Python release; use Python 3.10 or 3.11, not a beta')
        import torch
        from transformers import set_seed
        from gliner import GLiNER

        if torch.cuda.device_count() > 1:
            raise ValueError('Select one GPU using CUDA_VISIBLE_DEVICES before starting')
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        set_seed(config['seed'])
        model = GLiNER.from_pretrained(resume or config['base_model'], local_files_only=bool(resume))
        for key, value in schema_metadata(config['label_schema']).items():
            setattr(model.config, key, value)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'training_config.json', config)
        write_json(output / 'data_manifest.json', manifest)
        trainer = build_trainer(model, config, train, val)
        trainer.train(resume_from_checkpoint=resume)
        if config.get('save_best_model', False):
            export_best_model(trainer, output, config)
            trainer.save_state()
        elif not any(row.get('step') == trainer.state.global_step and 'eval_exact_metrics' in row
                   for row in trainer.state.log_history):
            trainer.evaluate()
        if not config.get('save_best_model', False):
            trainer.save_model(str(output / 'final_model'))
        save_history(output, trainer.state)
        print(f'Training complete. Artifacts: {output}')
        return 0
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}', file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
