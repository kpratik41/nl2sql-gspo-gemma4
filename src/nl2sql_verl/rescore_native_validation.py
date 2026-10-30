"""Re-score immutable saved generations after tightening terminal-answer gating."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from .prepare_reference_data import file_hash
from .reward_entry_native import compute_score


def main():
    import pyarrow.parquet as pq
    os.environ['NL2SQL_DATABASE_DIR']='/opt/dlami/nvme/nl2sql/databases'
    os.environ['NL2SQL_LENGTH_MAX']='4096'
    os.environ['NL2SQL_LENGTH_BUFFER']='512'
    root=Path('outputs/h200_checks/native_v1')
    data=Path('outputs/gemma4_native_v1/data')
    tables={s:pq.read_table(data/f'{s}.parquet').to_pylist() for s in ['train','validation']}
    provenance={'method':'Terminal rewards recomputed from immutable saved model generations; no generation changes',
                'reward_source_sha256':file_hash('src/nl2sql_verl/reward_entry_native.py'),
                'cohorts':{}}
    for label in ['base','step5']:
        directory=root/label
        rows=[json.loads(line) for line in (directory/'results.jsonl').open()]
        original=json.loads((directory/'provenance.json').read_text())

        def rescore(row):
            truth=tables[row['split']][row['index']]['reward_model']['ground_truth']
            extra=row['metrics']|{'assistant_text':row['assistant_text'],'generated_token_count':row['assistant_tokens']}
            reward=compute_score('nl2sql_bird','',truth,extra)
            return dict(row,scores=reward,previous_scores=row['scores'])

        with ThreadPoolExecutor(max_workers=4) as pool:
            corrected=list(pool.map(rescore,rows))
        (directory/'strict_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in corrected))
        summary=json.loads((directory/'summary.json').read_text())
        for split,group in summary['groups'].items():
            selected=[r for r in corrected if split=='all' or r['split']==split]
            for component in ['format','execution','result']:
                group[component+'_pass']=sum(r['scores'][component] for r in selected)
            group['format_percent']=100*group['format_pass']/len(selected)
            group['execution_percent']=100*group['execution_pass']/len(selected)
        summary['format_99_percent_target_met']=summary['groups']['all']['format_percent']>=99
        summary['execution_98_percent_target_met']=summary['groups']['all']['execution_percent']>=98
        summary['terminal_reward_recomputed']=True
        (directory/'strict_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        changed=[r['number'] for r in corrected if r['scores']!=r['previous_scores']]
        provenance['cohorts'][label]={'original_reward_source_sha256':next(
            v for k,v in original['source_sha256'].items() if k.endswith('/reward_entry_native.py')),
            'input_sha256':file_hash(directory/'results.jsonl'),
            'output_sha256':file_hash(directory/'strict_results.jsonl'),'reward_rows_changed':changed,
            'format_pass':summary['groups']['all']['format_pass'],
            'execution_pass':summary['groups']['all']['execution_pass']}
    (root/'strict_rescoring.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    main()
