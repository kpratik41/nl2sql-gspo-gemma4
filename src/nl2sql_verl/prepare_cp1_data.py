"""Prepare all 6,601 filtered training records and a nested fixed 640-row dev set."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from src.nl2sql_verl.prepare_data import normalize_record,TOOLS
root=Path('outputs/cp1_recipe').resolve()
(root/'data').mkdir(parents=True,exist_ok=True)
source=Path('outputs/b200_two_hour/data')
model='/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475'
tok=AutoTokenizer.from_pretrained(model)
manifest={'training_source':'data/raw/train-6601.jsonl','raw_9428_used':False,'dev_seed':20261026,'dev_selection':'First 640 rows of existing seeded eligible-dev order; includes original 32','length_filter':False,'splits':{}}
sets={}
for split,rows in [('train',pq.read_table(source/'train_source.parquet').to_pylist()),('validation',pq.read_table(source/'dev_eligible.parquet').slice(0,640).to_pylist())]:
 accepted=[];lengths=[]
 cached_path=root/'data'/f'{split}.parquet'; cached_lengths_path=root/'data'/f'{split}_prompt_lengths.json'
 cached_rows=pq.read_table(cached_path).to_pylist() if cached_path.exists() and cached_lengths_path.exists() else []
 cached_lengths=json.loads(cached_lengths_path.read_text()) if cached_rows else []
 for i,row in enumerate(rows):
  record=normalize_record(row,i)
  if i<len(cached_rows) and record==cached_rows[i]:
   length=cached_lengths[i]
  else:
   length=len(tok.apply_chat_template(record['prompt'],tools=TOOLS,tokenize=True,add_generation_prompt=True,return_dict=False))
  accepted.append(record);lengths.append(length)
  if (i+1)%1000==0:print(split,i+1,flush=True)
 table=pa.Table.from_pylist(accepted);path=root/'data'/f'{split}.parquet';pq.write_table(table,path)
 sets[split]={x['extra_info']['db_id'] for x in accepted}
 a=np.array(lengths)
 manifest['splits'][split]={'count':len(rows),'database_count':len(sets[split]),'database_ids':sorted(sets[split]),'prompt_tokens':{'min':int(a.min()),'p50':float(np.percentile(a,50)),'p90':float(np.percentile(a,90)),'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99)),'max':int(a.max()),'above_8192':int((a>8192).sum())},'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 (root/'data'/f'{split}_prompt_lengths.json').write_text(json.dumps(lengths)+'\n')
 manifest['splits'][split]['duplicate_prompt_rows_retained']=len(rows)-len({json.dumps(r['prompt'],sort_keys=True) for r in accepted})
assert manifest['splits']['train']['count']==6601
assert manifest['splits']['validation']['count']==640
assert not sets['train']&sets['validation']
old=pq.read_table(source/'validation.parquet').to_pylist();new=pq.read_table(root/'data/validation.parquet').slice(0,16).to_pylist()
assert all(x['prompt']==y['prompt'] and x['reward_model']==y['reward_model'] for x,y in zip(old,new,strict=True))
max_prompt=max(s['prompt_tokens']['max'] for s in manifest['splits'].values())
manifest['covering_prompt_limit']=math.ceil(max_prompt/1024)*1024
(root/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2),flush=True)
