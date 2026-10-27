"""Supervise an isolated native FSDP2/CP2 validation or training service."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
MODEL=Path('/home/ubuntu/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475')
ROOT=REPO/'outputs/fsdp_cp_recipe'


def command(run,checkpoint,smoke):
    runtime=REPO/'src/nl2sql_verl/worker_python.sh'
    cmd=[str(runtime),'-m','src.nl2sql_verl.train','--config-name','fsdp_cp2',
         f'actor_rollout_ref.model.path={MODEL}',f'trainer.default_local_dir={checkpoint}',
         f'trainer.rollout_data_dir={run}/rollouts',f'trainer.validation_data_dir={run}/validation',
         f'ray_kwargs.ray_init.runtime_env.py_executable={runtime}',
         '+ray_kwargs.ray_init.include_dashboard=false']
    if smoke:
        cmd+=['data.train_batch_size=2','actor_rollout_ref.actor.ppo_mini_batch_size=2',
              'trainer.total_training_steps=2','trainer.save_freq=1','trainer.test_freq=2',
              'trainer.val_before_train=true','trainer.resume_mode=disable']
    return cmd


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train']);p.add_argument('--run',action='store_true');a=p.parse_args()
    run=ROOT/a.mode;unit=f'nl2sql-fsdp-cp2-{a.mode}'
    checkpoint=Path(f'/opt/dlami/nvme/nl2sql/fsdp-cp2-{a.mode}')
    if not a.run:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU job already active')
        # The validator writes this only after numerical and capacity gates pass.
        ready=json.loads((ROOT/'readiness.json').read_text())
        for name,digest in ready['source_sha256'].items():
            if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'{name} changed since validation')
        for name,digest in ready['dataset_sha256'].items():
            if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'{name} changed since validation')
        if not ready.get('numerical_and_memory_ready'):raise RuntimeError('Numerical/memory gates are incomplete')
        if a.mode=='train' and not ready.get('integration_ready'):raise RuntimeError('RL/checkpoint integration gate is incomplete')
        if run.exists():run.rename(run.with_name(f'{a.mode}_previous_{int(time.time())}'))
        run.mkdir(parents=True)
        subprocess.run(['systemd-run','--user',f'--unit={unit}','--collect',f'--property=WorkingDirectory={REPO}',
                        '--property=KillMode=control-group','--property=TimeoutStopSec=60',
                        str(REPO/'.venv-verl/bin/python'),'-m','src.nl2sql_verl.run_fsdp',a.mode,'--run'],check=True)
        return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='2',NL2SQL_ATTENTION_FP32='1',NL2SQL_FLEX_FP32_PRECISION='tf32',NL2SQL_FLEX_CANONICAL_LAYOUT='0',NL2SQL_DATABASE_DIR=str(REPO/'databases'),NL2SQL_EXEC_TIMEOUT_S='30',NL2SQL_LENGTH_MAX='2048',NL2SQL_LENGTH_BUFFER='512',VERL_FILE_LOGGER_PATH=str(run/'metrics.jsonl'))
    cmd=command(run,checkpoint,a.mode=='smoke');(run/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    start=time.time()
    with (run/'training.log').open('w',buffering=1) as log:
        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            (run/'heartbeat.json').write_text(json.dumps({'pid':proc.pid,'elapsed_seconds':time.time()-start,'state':'running'})+'\n')
            if a.mode=='smoke' and time.time()-start>5400:
                os.killpg(proc.pid,signal.SIGTERM);break
            time.sleep(20)
        code=proc.wait()
    (run/'status.json').write_text(json.dumps({'exit_code':code,'elapsed_seconds':time.time()-start,'checkpoint':str(checkpoint)},indent=2)+'\n')
    raise SystemExit(code)

if __name__=='__main__':main()
