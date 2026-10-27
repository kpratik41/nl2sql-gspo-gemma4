"""Check, start, or inspect the validated CP1 recipe as a persistent user service."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
ARTIFACTS=REPO/'outputs/cp1_recipe'
RUN=REPO/'outputs/cp1_training'
UNIT='nl2sql-grpo-cp1'
MODEL=Path('/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475')
CHECKPOINT=Path('/opt/dlami/nvme/nl2sql/cp1-grpo')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    manifest=json.loads((ARTIFACTS/'data/manifest.json').read_text())
    for split,count in (('train',6601),('validation',640)):
        assert manifest['splits'][split]['count']==count
        assert digest(ARTIFACTS/f'data/{split}.parquet')==manifest['splits'][split]['sha256']
    report=json.loads((ARTIFACTS/'readiness.json').read_text())
    if not report.get('ready'):
        raise RuntimeError('CP1 numerical/memory/integration checks have not all passed; see readiness.json')
    for name,sha in report['source_sha256'].items():
        if digest(REPO/name)!=sha:raise RuntimeError(f'{name} changed since validation; rerun checks')
    return report


def command(initial_model):
    cmd=['env','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','OMP_NUM_THREADS=2','RECIPE_CONFIG=cp1',
        f'MODEL_PATH={MODEL}',f'TRAIN_PARQUET={ARTIFACTS}/data/train.parquet',
        f'VAL_PARQUET={ARTIFACTS}/data/validation.parquet','MAX_PROMPT_LENGTH=11264',
        'MAX_RESPONSE_LENGTH=2048','NL2SQL_EXEC_TIMEOUT_S=30',f'VERL_FILE_LOGGER_PATH={RUN}/metrics.jsonl']
    if initial_model:cmd.append(f'INIT_MODEL_CHECKPOINT={initial_model}')
    return cmd+['bash',str(REPO/'src/nl2sql_verl/launch.sh'),f'trainer.default_local_dir={CHECKPOINT}',
        f'trainer.rollout_data_dir={RUN}/rollouts',f'trainer.validation_data_dir={RUN}/validation',
        '+ray_kwargs.ray_init.include_dashboard=false']


def render_report():
    try:
        subprocess.run(['/usr/bin/python3',str(REPO/'src/nl2sql_verl/cp1_report.py'),'--run-dir',str(RUN)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30,check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'Optional progress rendering failed: {exc}',flush=True)


def supervise():
    session=json.loads((RUN/'session.json').read_text());start=time.time()
    with (RUN/'training.log').open('a',buffering=1) as out:
        child=subprocess.Popen(session['command'],cwd=REPO,stdout=out,stderr=subprocess.STDOUT)
        while child.poll() is None:
            (RUN/'heartbeat.json').write_text(json.dumps({'pid':child.pid,'elapsed_seconds':time.time()-start,'state':'running'})+'\n')
            render_report()
            time.sleep(30)
        result={'exit_code':child.returncode,'state':'complete' if child.returncode==0 else 'failed','elapsed_seconds':time.time()-start}
        render_report()
        (RUN/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(child.returncode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','start','status','supervise'])
    parser.add_argument('--initial-model',type=Path,help='Optional model-only checkpoint directory, e.g. global_step_6/actor/model. A new recipe uses fresh optimizer and dataloader state.')
    args=parser.parse_args()
    if args.action=='supervise':return supervise()
    if args.action=='status':
        subprocess.run(['systemctl','--user','status',UNIT,'--no-pager'],check=False)
        if (RUN/'status.json').exists():print((RUN/'status.json').read_text())
        return
    report=check()
    if args.action=='check':print(json.dumps(report,indent=2));return
    if args.initial_model and not args.initial_model.is_dir():raise FileNotFoundError(args.initial_model)
    RUN.mkdir(parents=True,exist_ok=True);CHECKPOINT.mkdir(parents=True,exist_ok=True)
    if subprocess.run(['systemctl','--user','is-active','--quiet',UNIT]).returncode==0:
        raise RuntimeError('CP1 service is already running')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('GPUs still have compute processes; finish the active diagnostic/job first')
    initial=args.initial_model.resolve() if args.initial_model else None
    if list(CHECKPOINT.glob('global_step_*/nl2sql_complete.json')):initial=None
    for name in ('status.json','heartbeat.json'):
        path=RUN/name
        if path.exists():path.rename(RUN/f'{path.stem}_previous_{int(time.time())}.json')
    cmd=command(initial)
    session={'started_epoch':time.time(),'command':cmd,'checkpoint_dir':str(CHECKPOINT),'train_rows':6601,'eval_rows':640,'tp':2,'dp':4,'cp':1,'rollout_tp':4,'rollout_dp':2,'compute_dtype':'bf16','initial_model_only':str(initial) if initial else None,'optimizer_updates':100}
    if (RUN/'session.json').exists():
        (RUN/f'session_previous_{int(time.time())}.json').write_text((RUN/'session.json').read_text())
    (RUN/'session.json').write_text(json.dumps(session,indent=2)+'\n')
    (RUN/'readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run(['systemd-run','--user',f'--unit={UNIT}','--collect',f'--property=WorkingDirectory={REPO}',
        '--property=KillMode=control-group','--property=TimeoutStopSec=30','/usr/bin/python3',str(Path(__file__).resolve()),'supervise'],check=True)
    print(f'Started {UNIT}; logs: {RUN}/training.log')


if __name__=='__main__':main()
