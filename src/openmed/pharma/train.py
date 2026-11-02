"""Full-model OpenMed training. Explicit execution only; Train/Validation only."""
import argparse
import gc
import json
import math
import time
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from evaluate import PROJECT, MODEL_ID, REVISION, TAGS, check, better, load_prepared, evaluate_model

GROUPS = ('learning_rate_1e-5','learning_rate_2e-5','learning_rate_3e-5')

def read_config(path):
    config=json.loads(path.read_text())
    required={'name','learning_rate','epochs','seed','batch_size','gradient_accumulation_steps','weight_decay','warmup_ratio','max_grad_norm','gradient_checkpointing'}
    check(set(config)==required, f'{path}: unexpected/missing configuration keys')
    check(isinstance(config['name'],str) and path.stem==config['name'] and config['name'].replace('_','').replace('-','').replace('.','').isalnum(), 'Configuration/group name mismatch')
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


def completed_results(project, hashes=None):
    entries=[]
    for path in (project/'outputs').glob('*/validation_results.json'):
        result=json.loads(path.read_text())
        if not result.get('completed'): continue
        if hashes is not None: check(result['data_sha256']==hashes, f'Dataset mismatch: {path}')
        entries.append(result)
    def order(result):
        name=result['group']
        return (int(name[3:5]) if name.startswith('run') and name[3:5].isdigit() else 0,
                result['configuration']['learning_rate'],name)
    return sorted(entries,key=order)

def global_selection(project):
    path=project/'outputs'/'best_model'/'selection.json'
    return json.loads(path.read_text()) if path.is_file() else None

def consolidate_best(project, hashes):
    results=completed_results(project,hashes)
    check(bool(results),'Existing completed results are required before adaptive tuning')
    destination=project/'outputs'/'best_model'
    if not destination.exists():
        winner=max(results,key=lambda r:r['metrics']['overall']['f1'])
        source=project/'outputs'/winner['group']/'best_model'
        if (source/'model.safetensors').is_file() and (source/'config.json').is_file():
            source.rename(destination)
        else:
            destination.mkdir()
            write_json(destination/'selection.json',dict(group=winner['group'],epoch=winner['best_epoch'],validation_micro_f1=winner['metrics']['overall']['f1'],configuration=winner['configuration'],data_sha256=hashes,checkpoint_available=False))
            print('Historical checkpoint weights are unavailable; preserving metrics and establishing a new best available checkpoint.',flush=True)
    selected=global_selection(project)
    check(selected is not None and selected['data_sha256']==hashes,'Overall checkpoint provenance mismatch')
    print('Incumbent/reference:',selected['group'],selected['validation_micro_f1'],flush=True)

def export_overall(model, tokenizer, mapping, selection, project):
    import shutil
    from safetensors import safe_open
    root=project/'outputs'
    destination=root/'best_model'
    pending=root/'.best_model_pending'
    previous=root/'.best_model_previous'
    check(not pending.exists() and not previous.exists(),'Incomplete checkpoint promotion needs inspection; existing best is preserved')
    selection['checkpoint_available']=True
    selection['historical_best_validation_micro_f1']=max(r['metrics']['overall']['f1'] for r in completed_results(project))
    model.save_pretrained(pending,safe_serialization=True)
    tokenizer.save_pretrained(pending)
    write_json(pending/'label_map.json',mapping)
    write_json(pending/'selection.json',selection)
    with safe_open(pending/'model.safetensors',framework='pt',device='cpu') as saved:
        check(tuple(saved.get_slice('classifier.weight').get_shape())==(23,1024),'Exported classifier shape mismatch')
    check(json.loads((pending/'config.json').read_text())['id2label']=={str(i):t for i,t in enumerate(TAGS)},'Exported label schema mismatch')
    if destination.exists(): destination.rename(previous)
    try: pending.rename(destination)
    except Exception:
        if previous.exists(): previous.rename(destination)
        raise
    if previous.exists(): shutil.rmtree(previous)
    print('Saved best available checkpoint:',selection['group'],'epoch',selection['epoch'],selection['validation_micro_f1'],flush=True)

def propose_next(project, number, results):
    best=max(results,key=lambda r:r['metrics']['overall']['f1'])
    cfg=best['configuration']
    keys={'name','learning_rate','epochs','seed','batch_size','gradient_accumulation_steps','weight_decay','warmup_ratio','max_grad_norm','gradient_checkpointing'}
    proposal={k:cfg[k] for k in keys}
    tried={(r['configuration']['learning_rate'],r['configuration']['epochs']) for r in results}
    if best['best_epoch']>=cfg['epochs']-1 and cfg['epochs']<12 and (cfg['learning_rate'],12) not in tried:
        proposal['epochs']=12
        reason=f"Best run {best['group']} peaked late ({best['best_epoch']}/{cfg['epochs']}); test a longer 12-epoch schedule at the same learning rate."
    else:
        rate=cfg['learning_rate']
        rates=sorted({r['configuration']['learning_rate'] for r in results})
        neighbors=sorted([x for x in rates if x<rate],reverse=True)[:1]+[x for x in rates if x>rate][:1]
        neighbors.sort(key=lambda x:max(r['metrics']['overall']['f1'] for r in results if r['configuration']['learning_rate']==x),reverse=True)
        candidates=[(rate+x)/2 for x in neighbors]+[rate*.8,rate*1.2]
        proposal['epochs']=cfg['epochs'] if cfg['epochs']>5 else 8
        for candidate in candidates:
            candidate=float(f'{candidate:.8g}')
            if 1e-5<=candidate<=6e-5 and (candidate,proposal['epochs']) not in tried:
                proposal['learning_rate']=candidate
                break
        else: return None,'No distinct local learning-rate candidate remains.'
        reason=f"Refine learning rate around {best['group']} (Validation F1 {best['metrics']['overall']['f1']:.6f}); keep its epoch budget."
    rate_name=f"{proposal['learning_rate']/1e-5:.4g}".replace('.','p')+'e-5'
    proposal['name']=f"run{number:02d}_lr{rate_name}_ep{proposal['epochs']}"
    return proposal,reason

def adaptive_runs(paths, project, data, mapping, manifest, hashes):
    for path in paths:
        config=read_config(path)
        output=project/'outputs'/config['name']/'validation_results.json'
        if output.is_file() and json.loads(output.read_text()).get('completed'):
            print('Preserving completed run:',config['name'],flush=True)
            continue
        check(not output.parent.exists(),f'Incomplete output exists: {output.parent}; will not silently rerun')
        run_group(config,project,data,mapping,manifest,hashes,reason='Initial extension: eight epochs; other settings match the previous runs.')
    for number in range(6,9):
        results=completed_results(project,hashes)
        adaptive=[r for r in results if r['group'].startswith(('run06','run07','run08'))]
        if any(r['group'].startswith(f'run{number:02d}_') for r in adaptive): continue
        if len(adaptive)>=2 and all(r.get('global_improvement',0)<0.001 for r in adaptive[-2:]):
            reason='Stopped: two successive adaptive runs improved overall Validation micro F1 by less than 0.001.'
            last=adaptive[-1];last['adaptive_stop_reason']=reason
            write_json(project/'outputs'/last['group']/'validation_results.json',last)
            print(reason,flush=True);break
        config,reason=propose_next(project,number,results)
        if config is None: print(reason,flush=True);break
        path=project/'configs'/f"{config['name']}.json"
        check(not path.exists(),f'Configuration already exists without completed results: {path}')
        write_json(path,config)
        print('ADAPTIVE DECISION:',reason, json.dumps(config),flush=True)
        run_group(config,project,data,mapping,manifest,hashes,reason=reason)
    print('FINAL COMPARISON',flush=True)
    for result in completed_results(project,hashes):
        print(json.dumps({k:result[k] for k in ('group','best_epoch')}|{'learning_rate':result['configuration']['learning_rate'],'max_epochs':result['configuration']['epochs'],**result['metrics']['overall']}),flush=True)
    print('FINAL BEST',json.dumps(global_selection(project)),flush=True)

def run_group(config, project, data, mapping, manifest, hashes, reason=None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, get_linear_schedule_with_warmup, set_seed
    check(torch.cuda.is_available(),'A CUDA Colab runtime is required')
    started=time.monotonic()
    incumbent=global_selection(project)
    starting_global_score=incumbent['validation_micro_f1'] if incumbent else -1.0
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
    resolved['tuning_reason']=reason or 'Explicit configuration run'
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
            if (index+1)%300==0: print(f"{config['name']} epoch {epoch}: batch {index+1}/{len(loader)}",flush=True)
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
            best_result['global_improvement']=max(0.0,score-starting_global_score)
            selected=global_selection(project)
            if selected is None or not selected.get('checkpoint_available',True) or better(score,selected['validation_micro_f1']):
                export_overall(model,tokenizer,mapping,dict(group=config['name'],epoch=epoch,validation_micro_f1=score,configuration=resolved,data_sha256=hashes),project)
            write_json(group_dir/'validation_results.json',best_result)
    best_result['completed']=True
    best_result['duration_seconds']=round(time.monotonic()-started,2)
    write_json(group_dir/'validation_results.json',best_result)
    del loader,optimizer,scheduler,scaler,model,tokenizer
    gc.collect();torch.cuda.empty_cache()
    return best_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',type=Path,default=PROJECT)
    parser.add_argument('--configs',nargs='+',type=Path)
    parser.add_argument('--adaptive',action='store_true',help='Run/skip completed Runs 4-5, then select at most Runs 6-8 using Validation only')
    parser.add_argument('--check-data',action='store_true',help='Read-only checks, without model initialization')
    args=parser.parse_args()
    project=args.project_dir.resolve()
    paths=args.configs or [project/'configs'/name for name in ('run04_lr3e-5_ep8.json','run05_lr4e-5_ep8.json')]
    configs=[read_config(path) for path in paths]
    check(len({c['name'] for c in configs})==len(configs),'Repeated configuration group')
    data,mapping,manifest,summaries,hashes=load_prepared(project)
    if args.check_data:
        print(json.dumps(dict(status='PASS',configurations=configs,data=summaries),indent=2));return
    if args.adaptive:
        check([c['name'] for c in configs]==['run04_lr3e-5_ep8','run05_lr4e-5_ep8'],'Adaptive mode starts only from Runs 4-5')
    else:
        for c in configs: check(not (project/'outputs'/c['name']).exists(),f"Refusing to rerun {c['name']}")
    consolidate_best(project,hashes)
    if args.adaptive:
        adaptive_runs(paths,project,data,mapping,manifest,hashes)
    else:
        for c in configs:run_group(c,project,data,mapping,manifest,hashes)
        print('FINAL BEST',json.dumps(global_selection(project)),flush=True)

if __name__=='__main__':
    main()
