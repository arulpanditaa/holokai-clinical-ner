"""Offline lifecycle tests using a tiny random GLiNER and synthetic records only.

Run: python -m unittest discover -s finetune/scripts -p test_final_run.py -v
No project training data, pretrained downloads, or V4 evaluation are used.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from gliner import GLiNER, GLiNERConfig
from gliner.model import UniEncoderSpanGLiNER
from transformers import BertConfig, BertTokenizerFast, TrainerCallback, set_seed

import final_model.scripts.train_gliner_final_run as training
from label_schemas import schema_metadata


class InterruptedRun(Exception):
    pass


class FinalRunLifecycleTest(unittest.TestCase):
    def test_checkpoint_export_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                          '<<ENT>>', '<<SEP>>', 'name', 'alice', 'bob', 'visits']
            (root / 'vocab.txt').write_text('\n'.join(vocabulary) + '\n')
            tokenizer = BertTokenizerFast(vocab_file=str(root / 'vocab.txt'), do_lower_case=False)
            tokenizer.add_special_tokens({'additional_special_tokens': ['<<ENT>>', '<<SEP>>']})
            encoder = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32,
                                 max_position_embeddings=128)
            model_config = GLiNERConfig(model_name=str(root), encoder_config=encoder.to_dict(),
                                        hidden_size=16, max_width=2, max_len=32, dropout=0.1,
                                        num_rnn_layers=0, vocab_size=len(tokenizer),
                                        class_token_index=tokenizer.convert_tokens_to_ids('<<ENT>>'))
            for key, value in schema_metadata('recommended').items():
                setattr(model_config, key, value)
            set_seed(42)
            initial = UniEncoderSpanGLiNER(model_config, tokenizer=tokenizer, backbone_from_pretrained=False)
            initial.save_pretrained(root / 'initial', safe_serialization=True)
            train = [{'tokenized_text': ['alice', 'visits', 'bob'], 'ner': [[0, 0, 'name']]} for _ in range(5)]
            val = [{'tokenized_text': ['bob'], 'ner': [[0, 0, 'name']]}]
            base = training.load_config(Path(__file__).parents[2] / 'final_model/configs/finetune_final_run.json')
            base.update(epochs=1, batch_size=1, eval_batch_size=1, eval_steps=2,
                        logging_steps=1, device='cpu', base_model=str(root / 'initial'))
            manifest = {'synthetic': True}

            def make_trainer(directory):
                config = {**base, 'output_dir': str(directory)}
                directory.mkdir(exist_ok=True)
                training.write_json(directory / 'training_config.json', config)
                training.write_json(directory / 'data_manifest.json', manifest)
                set_seed(42)
                model = GLiNER.from_pretrained(str(root / 'initial'), local_files_only=True)
                return training.build_trainer(model, config, train, val), config

            class InterruptAfterSave(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    if state.global_step == 2:
                        raise InterruptedRun()

            # Controlled scores prove selection is based on validation F1, not last step.
            # Native inference still executes on synthetic validation records each time.
            def metric_at_step(trainer, final_score=0.2):
                def metrics(*args, **kwargs):
                    exact = original_metrics(*args, **kwargs)
                    exact['overall']['f1'] = {2: 0.8, 4: 0.8, 5: final_score}[trainer.state.global_step]
                    return exact
                return metrics

            original_metrics = training.compute_metrics
            full, _ = make_trainer(root / 'full')
            with patch.object(training, 'compute_metrics', side_effect=metric_at_step(full, final_score=0.9)):
                full.train()
            self.assertEqual(Path(full.state.best_model_checkpoint).name, 'checkpoint-5')
            interrupted, config = make_trainer(root / 'resumed')
            interrupted.add_callback(InterruptAfterSave())
            with patch.object(training, 'compute_metrics', side_effect=metric_at_step(interrupted)):
                with self.assertRaises(InterruptedRun):
                    interrupted.train()
            latest = training.resolve_resume(config, manifest, 'latest')
            self.assertEqual(Path(latest).name, 'checkpoint-2')
            # A partial newer write is never a valid resume target.
            (root / 'resumed/checkpoint-3').mkdir()
            self.assertEqual(training.resolve_resume(config, manifest, 'latest'), latest)
            with self.assertRaises(ValueError):
                training.resolve_resume({**config, 'learning_rate': 0.01}, manifest, 'latest')
            with self.assertRaises(ValueError):
                training.resolve_resume(config, {'synthetic': 'changed'}, 'latest')
            self.assertTrue(training.output_has_run_artifacts(config))

            resumed, config = make_trainer(root / 'resumed')
            with patch.object(training, 'compute_metrics', side_effect=metric_at_step(resumed)):
                resumed.train(resume_from_checkpoint=latest)
            self.assertEqual(resumed.state.global_step, 5)
            self.assertEqual(resumed.state.best_metric, 0.8)
            self.assertEqual(Path(resumed.state.best_model_checkpoint).name, 'checkpoint-2')
            self.assertEqual([r['step'] for r in resumed.state.log_history if 'eval_f1' in r], [2, 4, 5])
            self.assertEqual([r['loss'] for r in full.state.log_history if 'loss' in r],
                             [r['loss'] for r in resumed.state.log_history if 'loss' in r])
            for name in ('checkpoint-2', 'checkpoint-5'):
                training.validate_checkpoint(root / 'resumed' / name)
            self.assertFalse((root / 'resumed/checkpoint-4').exists())

            # Resume must reproduce the uninterrupted final weights, including RNG effects.
            a = GLiNER.from_pretrained(str(root / 'full/checkpoint-5'), local_files_only=True)
            b = GLiNER.from_pretrained(str(root / 'resumed/checkpoint-5'), local_files_only=True)
            for key, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[key], rtol=0, atol=0)
            for filename in ('optimizer.pt', 'scheduler.pt'):
                def compare(a, b):
                    if isinstance(a, torch.Tensor):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    elif isinstance(a, dict):
                        self.assertEqual(a.keys(), b.keys())
                        for key in a:
                            compare(a[key], b[key])
                    elif isinstance(a, (tuple, list)):
                        self.assertEqual(len(a), len(b))
                        for x, y in zip(a, b):
                            compare(x, y)
                    else:
                        self.assertEqual(a, b)
                compare(torch.load(root / 'full/checkpoint-5' / filename, weights_only=True),
                        torch.load(root / 'resumed/checkpoint-5' / filename, weights_only=True))
            best = GLiNER.from_pretrained(str(root / 'resumed/checkpoint-2'), local_files_only=True)
            for key, tensor in best.state_dict().items():
                torch.testing.assert_close(tensor, resumed.model.state_dict()[key], rtol=0, atol=0)
            exported = training.export_best_model(resumed, root / 'resumed', config)
            training.save_history(root / 'resumed', resumed.state)
            self.assertEqual(len(training.read_json(root / 'resumed/validation_metrics.json')), 3)
            self.assertTrue(training.read_json(root / 'resumed/train_loss.json'))
            with self.assertRaises(ValueError):
                training.export_best_model(resumed, root / 'resumed', config)
            with self.assertRaises(ValueError):
                training.resolve_resume(config, manifest, 'latest')
            # A new process reloads and performs inference without a Trainer or remote cache.
            subprocess.run([sys.executable, '-c',
                            'from gliner import GLiNER; import sys; '
                            'm=GLiNER.from_pretrained(sys.argv[1],local_files_only=True); '
                            'm.eval(); print(m.predict_entities("alice visits bob",["name"],threshold=0.4))',
                            str(exported)], check=True,
                           env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})


if __name__ == '__main__':
    unittest.main()
