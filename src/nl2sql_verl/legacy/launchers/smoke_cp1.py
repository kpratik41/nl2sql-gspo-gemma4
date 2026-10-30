"""One real CP1 GRPO/DAPO update (2 groups x 16), followed by all 640 dev rows."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from .manage_cp1 import REPO,ARTIFACTS,command

ROOT=ARTIFACTS/'smoke'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--initial-model',type=Path);a=p.parse_args()
    if not a.run:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Finish current GPU jobs before running the smoke check')
        cmd=['systemd-run','--user','--unit=nl2sql-cp1-smoke','--collect',f'--property=WorkingDirectory={REPO}',
             '--property=RuntimeMaxSec=5520','--property=TimeoutStopSec=30',str(REPO/'.venv-verl/bin/python'),'-m','src.nl2sql_verl.smoke_cp1','--run']
        if a.initial_model:cmd+=['--initial-model',str(a.initial_model.resolve())]
        subprocess.run(cmd,check=True);return
    if ROOT.exists():ROOT.rename(ROOT.with_name(f'smoke_previous_{int(time.time())}'))
    ROOT.mkdir(parents=True)
    cmd=command(a.initial_model)
    for i,arg in enumerate(cmd):
        if arg.startswith('VERL_FILE_LOGGER_PATH='):cmd[i]=f'VERL_FILE_LOGGER_PATH={ROOT}/metrics.jsonl'
        elif arg.startswith('trainer.default_local_dir='):cmd[i]='trainer.default_local_dir=/opt/dlami/nvme/nl2sql/cp1-smoke'
        elif arg.startswith('trainer.rollout_data_dir='):cmd[i]=f'trainer.rollout_data_dir={ROOT}/rollouts'
        elif arg.startswith('trainer.validation_data_dir='):cmd[i]=f'trainer.validation_data_dir={ROOT}/validation'
    cmd+=['data.train_batch_size=2','actor_rollout_ref.actor.ppo_mini_batch_size=2','trainer.total_training_steps=1',
          'trainer.val_before_train=false','trainer.save_freq=1','trainer.test_freq=1','trainer.resume_mode=disable']
    (ROOT/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    start=time.time()
    with (ROOT/'training.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=child.wait(timeout=5400)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM);code=124
    result={'exit_code':code,'status':'pass' if code==0 else 'fail','elapsed_seconds':time.time()-start}
    (ROOT/'training.status.json').write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(code)


if __name__=='__main__':main()
