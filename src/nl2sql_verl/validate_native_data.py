"""Audit the versioned data against its parent and actual SQLite primary keys."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3

from .prepare_reference_data import file_hash
from .task import database_path
from .tool_profiles import load_tool_profile


def validate(directory, database_dir):
    import pyarrow.parquet as pq
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert file_hash(Path(__file__).parent/name)==digest, f'Preparation source changed: {name}'
    profile = load_tool_profile(manifest['tool_profile'])
    changes = json.loads((root/'foreign_key_corrections.json').read_text())
    # Check corrected targets against the actual declared primary keys, rather
    # than merely accepting the transformation's saved replacement strings.
    verified_edges = 0
    for db, edges in changes.items():
        with sqlite3.connect(f'file:{database_path(db, database_dir)}?mode=ro', uri=True) as c:
            for edge in edges:
                left, target = edge['new'].split('=')
                child, column = left.split('.',1)
                parent, target_column = target.split('.',1)
                q = lambda name: '"'+name.replace('"','""')+'"'
                fk = [r for r in c.execute('PRAGMA foreign_key_list('+q(child)+')')
                      if r[2]==parent and r[3]==column and r[4] is None]
                pk = sorted((r[5],r[1]) for r in c.execute('PRAGMA table_info('+q(parent)+')') if r[5])
                assert fk and all(pk[r[1]][1]==target_column for r in fk), (db, edge)
                verified_edges += 1
    rows = {}
    for split in ['train','validation','dev_full']:
        path = root/f'{split}.parquet'
        source = Path(manifest['parent'])/path.name
        assert file_hash(path)==manifest['splits'][split]['sha256']
        assert file_hash(source)==manifest['parent_sha256'][str(source)]
        before = pq.read_table(source).to_pylist()
        after = pq.read_table(path).to_pylist()
        assert len(before)==len(after)==manifest['splits'][split]['count']
        for old, new in zip(before,after,strict=True):
            for key in old:
                if key not in ['prompt','agent_name','extra_info']:
                    assert old[key]==new[key], (split,key)
            assert new['agent_name']==profile.agent_name
            assert new['prompt'][0]==profile.messages('')[0]
            old_user, new_user = old['prompt'][1]['content'],new['prompt'][1]['content']
            old_parts = re.split(r'(<database_schema>.*?</database_schema>)',old_user,flags=re.S)
            new_parts = re.split(r'(<database_schema>.*?</database_schema>)',new_user,flags=re.S)
            assert len(old_parts)==len(new_parts)==3
            assert old_parts[0]==new_parts[0] and old_parts[2]==new_parts[2]
            expected = old_parts[1]
            for edge in changes.get(old['extra_info']['db_id'],[]):
                expected = re.sub(r'(?<![\w.])'+re.escape(edge['old'])+r'(?!\w)', lambda _:edge['new'],expected)
            assert expected==new_parts[1]
            assert not re.search(r'\b\w+\.None\b',new_parts[1])
            for key,value in old['extra_info'].items():
                if key!='tool_profile':
                    assert new['extra_info'][key]==value,(split,key)
        lengths=json.loads((root/f'{split}_prompt_lengths.json').read_text())
        assert len(lengths)==len(after) and max(lengths)<=30720
        rows[split]=after
    assert [len(rows[s]) for s in ['train','validation','dev_full']]==[6601,640,1534]
    by_reference={r['extra_info']['reference_index']:r for r in rows['dev_full']}
    assert rows['validation']==[by_reference[r['extra_info']['reference_index']] for r in rows['validation']]
    report={'status':'pass','rows':{s:len(r) for s,r in rows.items()},'gold_and_source_order_preserved':True,
            'all_non_schema_user_content_preserved':True,'evaluation_membership_preserved':True,
            'implicit_fk_edges_independently_verified':verified_edges,'remaining_none_targets':0,
            'dropped_or_truncated_rows':0,'profile':profile.name,'response_budget':4096,'context_budget':34816}
    (root/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return rows, report


def fixed_sample(rows, root, count=128, seed=20261029):
    """Include all DBs and the longest prompts, then seeded random fill."""
    selected=[]
    for split in ['train','validation']:
        items=rows[split]
        lengths=json.loads((root/f'{split}_prompt_lengths.json').read_text())
        by_db={}
        for index,row in enumerate(items):
            by_db.setdefault(row['extra_info']['db_id'],[]).append(index)
        rng=random.Random(seed)
        indices={rng.choice(indices) for _,indices in sorted(by_db.items())}
        indices.update(sorted(range(len(items)),key=lambda i:lengths[i],reverse=True)[:8])
        remaining=[i for i in range(len(items)) if i not in indices]
        rng.shuffle(remaining)
        indices.update(remaining[:count-len(indices)])
        assert len(indices)==count
        for index in sorted(indices):
            row=items[index]
            selected.append({'split':split,'index':index,'db_id':row['extra_info']['db_id'],
                             'reference_index':row['extra_info']['reference_index'],'prompt_tokens':lengths[index],
                             'prompt_sha256':hashlib.sha256(json.dumps(row['prompt'],ensure_ascii=False).encode()).hexdigest()})
    result={'seed':seed,'selection':'all databases, eight longest each split, seeded fill',
            'count':len(selected),'data_sha256':{s:file_hash(root/f'{s}.parquet') for s in ['train','validation']},'samples':selected}
    (root/'fixed_validation_sample.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='outputs/gemma4_native_v1/data')
    p.add_argument('--database-dir',default='/opt/dlami/nvme/nl2sql/databases')
    args=p.parse_args()
    rows, report=validate(args.data_dir,args.database_dir)
    sample=fixed_sample(rows,Path(args.data_dir))
    print(json.dumps(report|{'fixed_sample_count':sample['count']},indent=2))
