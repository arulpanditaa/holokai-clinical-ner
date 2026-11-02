"""Full-model OpenMed training. Explicit execution only; Train/Validation only."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from evaluate import PROJECT, MODEL_ID, REVISION, TAGS, check, better, load_prepared, evaluate_model

GROUPS = ('learning_rate_3e-5','learning_rate_5e-5','learning_rate_8e-5')

def read_config(path):
    config=json.loads(path.read_text())
    required={'name','learning_rate','epochs','seed','batch_size','gradient_accumulation_steps','weight_decay','warmup_ratio','max_grad_norm','gradient_checkpointing'}
    check(set(config)==required, f'{path}: unexpected/missing configuration keys')
    check(config['name'] in GROUPS and path.stem==config['name'], 'Configuration/group name mismatch')
    for field in ('epochs','batch_size','gradient_accumulation_steps'):
        check(type(config[field]) is int and config[field]>0, f'Invalid {field}')
    check(type(config['seed']) is int and 0<=config['seed']<2**32,'Invalid seed')
    for field in ('learning_rate','max_grad_norm'):
        check(type(config[field]) in (int,float) and math.isfinite(config[field]) and config[field]>0, f'Invalid {field}')
    check(type(config['weight_decay']) in (int,float) and math.isfinite(config['weight_decay']) and config['weight_decay']>=0,'Invalid weight decay')
    check(type(config['warmup_ratio']) in (int,float) and 0<=config['warmup_ratio']<=1,'Invalid warmup ratio')
    check(type(config['gradient_checkpointing']) is bool,'Invalid gradient checkpointing setting')
    return config

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def run_group(config, project, data, mapping, manifest, hashes):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, get_linear_schedule_with_warmup, set_seed
    check(torch.cuda.is_available(),'A CUDA Colab runtime is required')
    set_seed(config['seed'])
    device=torch.device('cuda')
    bf16=torch.cuda.is_bf16_supported()
    precision='bf16' if bf16 else 'fp16'
    dtype=torch.bfloat16 if bf16 else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,use_fast=True,trust_remote_code=False)
    model=AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,revision=REVISION,num_labels=23,id2label=dict(enumerate(TAGS)),
        label2id={t:i for i,t in enumerate(TAGS)},ignore_mismatched_sizes=True,
        trust_remote_code=False)
    for parameter in model.parameters():parameter.requires_grad_(True)
    check(all(p.requires_grad for p in model.parameters()),'Full-model training requires all parameters trainable')
    if config['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.to(device)
    collator=DataCollatorForTokenClassification(tokenizer,padding=True,label_pad_token_id=-100)
    examples=[{k:item[k] for k in ('input_ids','attention_mask','labels')} for item in data['train']['flat']]
    loader=DataLoader(examples,batch_size=config['batch_size'],shuffle=True,collate_fn=collator,num_workers=0)
    accumulation=config['gradient_accumulation_steps']
    updates_per_epoch=math.ceil(len(loader)/accumulation)
    total_updates=updates_per_epoch*config['epochs']
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=get_linear_schedule_with_warmup(optimizer,math.ceil(total_updates*config['warmup_ratio']),total_updates)
    scaler=torch.amp.GradScaler('cuda',enabled=not bf16)
    group_dir=project/'outputs'/config['name']
    group_dir.parent.mkdir(parents=True,exist_ok=True)
    group_dir.mkdir(exist_ok=False)
    resolved=dict(config, model_id=MODEL_ID, model_revision=REVISION,
                  precision=precision,scheduler='linear',optimizer='AdamW',
                  train_data=str(project/'data/prepared/train.jsonl'),
                  validation_data=str(project/'data/prepared/validation.jsonl'),
                  effective_batch_size=config['batch_size']*accumulation,
                  torch_version=torch.__version__)
    import transformers
    resolved['transformers_version']=transformers.__version__
    best_score, best_result=None,None
    for epoch in range(1,config['epochs']+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index,batch in enumerate(loader):
            batch={k:v.to(device) for k,v in batch.items()}
            group_start=(index//accumulation)*accumulation
            window_size=min(accumulation,len(loader)-group_start)
            with torch.autocast('cuda',dtype=dtype):
                loss=model(**batch).loss/window_size
            check(bool(torch.isfinite(loss).item()),'Non-finite training loss; stopping without selecting this step')
            scaler.scale(loss).backward()
            if (index+1)%accumulation==0 or index+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
                old_scale=scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale()>=old_scale:scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics=evaluate_model(model,data['validation']['flat'],data['validation']['gold'],collator,device)
        score=metrics['overall']['f1']
        print(f"{config['name']} epoch {epoch}: Validation micro F1={score:.6f}",flush=True)
        if better(score,best_score):
            best_score=score
            best_result=dict(group=config['name'],best_epoch=epoch,metrics=metrics,
                             configuration=resolved,data_sha256=hashes,
                             preparation_settings=manifest['settings'],
                             selection_metric='validation_entity_exact_match_micro_f1',
                             tie_policy='earlier_epoch_then_configuration_order',
                             completed=False)
            destination=group_dir/'best_model'
            model.save_pretrained(destination,safe_serialization=True)
            tokenizer.save_pretrained(destination)
            write_json(destination/'label_map.json',mapping)
            write_json(destination/'selection.json',dict(group=config['name'],epoch=epoch,validation_micro_f1=score,configuration=resolved,data_sha256=hashes))
            write_json(group_dir/'validation_results.json',best_result)
    best_result['completed']=True
    write_json(group_dir/'validation_results.json',best_result)
    del loader,optimizer,scheduler,scaler,model,tokenizer
    gc.collect();torch.cuda.empty_cache()
    return best_result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path,default=PROJECT)
    parser.add_argument('--configs',nargs='+',type=Path)
    parser.add_argument('--check-data',action='store_true',help='Read-only checks; does not initialize models or create outputs')
    args=parser.parse_args()
    project=args.project_dir.resolve()
    paths=args.configs or [project/'configs'/f'{g}.json' for g in GROUPS]
    configs=[read_config(p) for p in paths]
    check(len({c['name'] for c in configs})==len(configs),'Repeated configuration group')
    data,mapping,manifest,summaries,hashes=load_prepared(project)
    if args.check_data:
        print(json.dumps(dict(status='PASS',configurations=configs,data=summaries),indent=2));return
    for config in configs:
        path=project/'outputs'/config['name']
        check(not path.exists() and not path.is_symlink(),f'Refusing to overwrite existing experiment: {path}')
    winner=None
    for config in configs:
        result=run_group(config,project,data,mapping,manifest,hashes)
        if winner is None or better(result['metrics']['overall']['f1'],winner['metrics']['overall']['f1']):
            winner=result
    print(json.dumps({'best_group':winner['group'],'validation_micro_f1':winner['metrics']['overall']['f1'],
                      'model_dir':str(project/'outputs'/winner['group']/'best_model')},indent=2))

if __name__=='__main__':
    main()
