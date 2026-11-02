"""Prepared-data checks and exact original-token entity evaluation. Never reads V4."""
import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
MODEL_ID = 'OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-184M'
REVISION = '1d72b9a9bf3e755fef613337a7d6b17e1b7ef48f'
MAPPING = {
 'condition':'DIAGNOSIS', 'organization medical facility':'FACILITY',
 'gender identity':'GENDER_IDENTITY', 'lab result or test finding':'LAB_RESULT',
 'location address':'LOCATION_ADDRESS', 'medication, drug, or prescription':'MEDICATION',
 'name':'PERSON', 'medical procedure or test performed':'PROCEDURE',
 'name medical professional':'PROVIDER', 'sex assigned at birth':'SEX',
 'personal identifier number':'UNIQUE_ID'}
TYPES = list(MAPPING.values())
TAGS = ['O'] + [f'{p}-{t}' for t in TYPES for p in ('B','I')]

def check(condition, message):
    if not condition:
        raise ValueError(message)

def better(score, previous):
    return previous is None or score > previous

def decode(tags):
    """Inclusive spans. Unmatched I-X starts a new entity (fixed policy)."""
    spans, current = [], None
    for i, tag in enumerate(list(tags) + ['O']):
        check(tag in TAGS, f'Unknown BIO tag: {tag}')
        prefix, label = ('O', None) if tag == 'O' else tag.split('-', 1)
        if current is not None and (prefix != 'I' or label != current[2]):
            spans.append((current[0], i - 1, current[2]))
            current = None
        if prefix != 'O' and current is None:
            current = (i, i, label)
    return spans

def scores(gold, predicted):
    gold, predicted = set(gold), set(predicted)
    def metric(g, p):
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, support=len(g))
    return {'overall': metric(gold, predicted), 'per_label': {
        label: metric({e for e in gold if e[-1] == label},
                      {e for e in predicted if e[-1] == label}) for label in TYPES}}

def validate_rows(rows, split, tags):
    check(bool(rows), f'{split}: empty dataset')
    flat, gold, ids = [], set(), set()
    counts = dict.fromkeys(TYPES, 0)
    for row in rows:
        rid = row['record_id']
        check(isinstance(rid, str) and rid not in ids, f'{split}: invalid/duplicate record ID')
        ids.add(rid)
        check(row['split'] == split, f'{rid}: wrong split')
        original = row['original']
        tokens, entities = original['tokenized_text'], original['ner']
        check(isinstance(tokens, list) and tokens and all(isinstance(t,str) and t.strip() for t in tokens), f'{rid}: invalid tokens')
        expected = ['O'] * len(tokens)
        seen = set()
        for start, end, mapped in entities:
            check(type(start) is int and type(end) is int and 0 <= start <= end < len(tokens), f'{rid}: invalid span')
            check(mapped in MAPPING, f'{rid}: unknown label')
            check((start,end,mapped) not in seen, f'{rid}: duplicate annotation')
            seen.add((start,end,mapped))
            check(all(t == 'O' for t in expected[start:end+1]), f'{rid}: overlapping annotations')
            label = MAPPING[mapped]
            expected[start] = f'B-{label}'
            expected[start+1:end+1] = [f'I-{label}'] * (end-start)
            gold.add((rid, start, end, label))
            counts[label] += 1
        check(row['bio_tags'] == expected, f'{rid}: BIO differs from original annotations')
        offset, recovered, chunk_ids = 0, set(), set()
        for chunk in row['chunks']:
            a, b = chunk['token_start'], chunk['token_end_exclusive']
            check(type(a) is int and type(b) is int and a == offset and a < b <= len(tokens), f'{rid}: chunk gap/overlap')
            check(chunk['chunk_id'] not in chunk_ids, f'{rid}: duplicate chunk ID')
            chunk_ids.add(chunk['chunk_id'])
            check(chunk['tokens'] == tokens[a:b] and chunk['bio_tags'] == expected[a:b], f'{rid}: chunk content mismatch')
            x, mask, y, words = (chunk[k] for k in ('input_ids','attention_mask','labels','word_ids'))
            check(0 < len(x) == len(mask) == len(y) == len(words) <= 512, f'{rid}: invalid tensor lengths')
            check(all(type(t) is int and 0 <= t < 128100 for t in x), f'{rid}: invalid token IDs')
            check(all(t == 1 for t in mask), f'{rid}: prepared inputs must be unpadded')
            check(all(type(t) is int and (t == -100 or 0 <= t < 23) for t in y), f'{rid}: invalid label IDs')
            word_sequence = [w for w in words if w is not None]
            check(all(type(w) is int and a <= w < b for w in word_sequence), f'{rid}: invalid word offsets')
            check(word_sequence == sorted(word_sequence) and set(word_sequence) == set(range(a,b)), f'{rid}: missing/reordered word alignment')
            word_seen, supervised = set(), []
            for tag_id, word in zip(y, words):
                if word is None or word in word_seen:
                    check(tag_id == -100, f'{rid}: invalid ignore mask')
                else:
                    check(tag_id >= 0 and tags[tag_id] == expected[word], f'{rid}: first subtoken label mismatch')
                    supervised.append(tags[tag_id])
                    word_seen.add(word)
            for s,e,label in decode(supervised):
                recovered.add((rid,s+a,e+a,label))
            flat.append(dict(record_id=rid, token_start=a, token_end_exclusive=b,
                             **{k:chunk[k] for k in ('input_ids','attention_mask','labels')}))
            offset = b
        check(offset == len(tokens), f'{rid}: incomplete token reconstruction')
        check(recovered == {e for e in gold if e[0] == rid}, f'{rid}: chunk entity reconstruction failed')
    return flat, gold, dict(records=len(rows), chunks=len(flat), entities=len(gold), label_counts=counts)

def load_prepared(project=PROJECT):
    project = Path(project)
    data = project / 'data'
    manifest = json.loads((data / 'preparation_manifest.json').read_text())
    mapping = json.loads((data / 'label_map.json').read_text())
    check(mapping['mapped_to_original'] == MAPPING, 'Label mapping differs from the fixed 11-label schema')
    check(mapping['original_to_mapped'] == {v:k for k,v in MAPPING.items()}, 'Inverse label mapping mismatch')
    tags = [mapping['id2label'][str(i)] for i in range(23)]
    check(tags == TAGS and mapping['label2id'] == {t:i for i,t in enumerate(TAGS)}, 'BIO vocabulary mismatch')
    check(manifest['tokenizer']['model_id'] == MODEL_ID and manifest['tokenizer']['revision'] == REVISION, 'Tokenizer/model revision mismatch')
    settings = manifest['settings']
    check(settings['max_length'] == 512 and settings['subtoken_supervision'] == 'first_only' and settings['ignore_index'] == -100, 'Unsupported preparation settings')
    datasets, summaries, hashes = {}, {}, {}
    # Explicit allowlist: no V4 path is opened, hashed or evaluated.
    for split in ('train', 'validation'):
        path = data / 'prepared' / f'{split}.jsonl'
        raw = path.read_bytes()
        hashes[split] = hashlib.sha256(raw).hexdigest()
        entry = manifest['splits'][split]
        check(entry['integrity_checks_passed'] and hashes[split] == entry['output_sha256'], f'{split}: manifest hash/integrity mismatch')
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        flat, gold, summary = validate_rows(rows, split, tags)
        check(summary['records'] == entry['records'] and summary['chunks'] == entry['chunks'] and summary['entities'] == entry['entities'] and summary['label_counts'] == entry['label_counts'], f'{split}: manifest counts mismatch')
        check(all(summary['label_counts'].values()), f'{split}: missing one of the 11 labels')
        datasets[split] = dict(rows=rows, flat=flat, gold=gold)
        summaries[split] = summary
    return datasets, mapping, manifest, summaries, hashes

def evaluate_model(model, flat, gold, collator, device, tags=TAGS):
    import torch
    model.eval()
    predicted = set()
    with torch.inference_mode():
        for item in flat:
            batch = collator([{k:item[k] for k in ('input_ids','attention_mask','labels')}])
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
            ids = logits[0].argmax(-1).tolist()
            word_tags = [tags[p] for p,g in zip(ids,item['labels']) if g != -100]
            check(len(word_tags) == item['token_end_exclusive'] - item['token_start'], 'Prediction alignment mismatch')
            for a,b,label in decode(word_tags):
                predicted.add((item['record_id'],a+item['token_start'],b+item['token_start'],label))
    return scores(gold, predicted)

def self_test():
    import copy
    import unittest
    class Tests(unittest.TestCase):
        def test_metrics(self):
            gold = {('r',0,1,'PERSON'),('r',2,2,'PERSON')}
            self.assertEqual(scores(gold,gold)['overall']['f1'],1)
            bad = {('r',0,0,'PERSON'),('r',2,2,'PROVIDER')}
            self.assertEqual(scores(gold,bad)['overall']['tp'],0)
            self.assertEqual(len(scores(set(),set())['per_label']),11)
            self.assertEqual(scores(set(),set())['overall']['f1'],0)
        def test_bio(self):
            self.assertEqual(decode(['B-PERSON','B-PERSON']),[(0,0,'PERSON'),(1,1,'PERSON')])
            self.assertEqual(decode(['I-PERSON','I-PERSON','I-PROVIDER']),[(0,1,'PERSON'),(2,2,'PROVIDER')])
        def test_chunks(self):
            row = dict(record_id='train:0',split='train',original=dict(tokenized_text=['A','B'],ner=[[0,0,'name']]),bio_tags=['B-PERSON','O'],chunks=[])
            for i,t in enumerate(row['bio_tags']):
                row['chunks'].append(dict(chunk_id=str(i),token_start=i,token_end_exclusive=i+1,tokens=[row['original']['tokenized_text'][i]],bio_tags=[t],input_ids=[1,10,11,2],attention_mask=[1]*4,labels=[-100,TAGS.index(t),-100,-100],word_ids=[None,i,i,None]))
            flat,gold,summary = validate_rows([row],'train',TAGS)
            self.assertEqual(summary['chunks'],2)
            self.assertEqual(gold,{('train:0',0,0,'PERSON')})
            broken=copy.deepcopy(row);broken['chunks'][1]['token_start']=0
            with self.assertRaises(ValueError):validate_rows([broken],'train',TAGS)
            broken=copy.deepcopy(row);broken['chunks'][0]['labels'][2]=TAGS.index('B-PERSON')
            with self.assertRaises(ValueError):validate_rows([broken],'train',TAGS)
        def test_selection(self):
            self.assertTrue(better(0,None));self.assertFalse(better(.5,.5));self.assertTrue(better(.6,.5))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    check(result.wasSuccessful(),'Synthetic tests failed')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path,default=PROJECT)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true')
    mode.add_argument('--check-data',action='store_true')
    mode.add_argument('--model-dir',type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    data,mapping,manifest,summary,hashes=load_prepared(args.project_dir)
    if args.check_data:
        print(json.dumps({'integrity':'PASS','splits':summary,'sha256':hashes},indent=2));return
    import torch
    from transformers import AutoTokenizer,AutoModelForTokenClassification,DataCollatorForTokenClassification
    model=AutoModelForTokenClassification.from_pretrained(args.model_dir,local_files_only=True)
    check(model.config.id2label == dict(enumerate(TAGS)),'Saved model label schema mismatch')
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    collator=DataCollatorForTokenClassification(tokenizer,padding=True,label_pad_token_id=-100)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(json.dumps(evaluate_model(model,data['validation']['flat'],data['validation']['gold'],collator,device),indent=2))

if __name__=='__main__':
    main()
