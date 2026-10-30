"""One-update CP2/DP4 integration on the isolated spare-node Ray cluster."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--generations',type=int,default=16)
    p.add_argument('--prompt-groups',type=int,default=2)
    args=p.parse_args()
    trajectories=args.generations*args.prompt_groups
    if args.generations<2 or args.prompt_groups<1 or trajectories%4:
        raise ValueError('Smoke needs heterogeneous groups and a trajectory count divisible by DP4')
    import ray
    address='172.31.10.43:6381'
    ray.init(address=address,ignore_reinit_error=True)
    alive=[n for n in ray.nodes() if n['Alive']]
    assert len(alive)==1 and alive[0]['NodeManagerAddress']=='172.31.10.43'
    assert int(ray.cluster_resources()['GPU'])==8
    ray.shutdown()
    root=Path(args.output_dir).resolve()
    root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    cmd=['bash',str(repo/'src/nl2sql_verl/launch_dapo_16gpu.sh'),
         'trainer.nnodes=1','trainer.n_gpus_per_node=8','trainer.nl2sql_controller_node_ip=172.31.10.43',
         'ray_kwargs.ray_init.address='+address,f'data.train_batch_size={args.prompt_groups}',
         f'actor_rollout_ref.actor.ppo_mini_batch_size={args.prompt_groups}',
         f'actor_rollout_ref.rollout.n={args.generations}',
         'actor_rollout_ref.rollout.agent.num_workers=8',
         'trainer.total_training_steps=1','trainer.total_epochs=1',
         'trainer.val_before_train=false','trainer.save_freq=-1','trainer.test_freq=-1',
         'trainer.logger=[console,file]','trainer.experiment_name=gemma4-native-v1-spare-node-smoke',
         'trainer.default_local_dir='+str(root/'checkpoints'),
         'trainer.nl2sql_model_backup_dir='+str(root/'exports'),
         'trainer.rollout_data_dir='+str(root/'rollouts'),
         'trainer.validation_data_dir='+str(root/'validation'),
         'actor_rollout_ref.model.path='+str(Path(args.model).resolve())]
    env=os.environ.copy()
    env.update(RAY_ADDRESS=address,VERL_PYTHON='/opt/dlami/nvme/nl2sql/runtime/venv/bin/python',
               VERL_SOURCE='/opt/dlami/nvme/nl2sql/runtime/verl',WANDB_MODE='disabled',
               VERL_FILE_LOGGER_PATH=str(root/'metrics.jsonl'),
               TORCHINDUCTOR_CACHE_DIR='/opt/dlami/nvme/nl2sql/caches/inductor',
               TRITON_CACHE_DIR='/opt/dlami/nvme/nl2sql/caches/triton')
    (root/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    paths=['native_runtime.py','native_agent_loop.py','native_tools.py','reward_entry_native.py',
           'fsdp_engine.py','fsdp_attention.py','fsdp_projection.py','tool_profiles.py',
           'gemma_native_profile.py','agent_loop.py','tools.py','task.py','train.py',
           'launch_dapo_16gpu.sh','worker_python.sh','configs/dapo_fsdp_cp2_16gpu.yaml',
           'configs/agent_native.yaml','configs/tools_native.yaml']
    (root/'source_sha256.json').write_text(json.dumps({
        'src/nl2sql_verl/'+name:hashlib.sha256((repo/'src/nl2sql_verl'/name).read_bytes()).hexdigest()
        for name in paths},indent=2)+'\n')
    start=time.monotonic()
    with (root/'training.log').open('w',buffering=1) as stream:
        proc=subprocess.Popen(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            (root/'heartbeat.json').write_text(json.dumps({'pid':proc.pid,'elapsed_seconds':time.monotonic()-start})+'\n')
            time.sleep(10)
    result={'exit_code':proc.returncode,'elapsed_seconds':time.monotonic()-start,
            'expected_trajectories':trajectories,'generations_per_prompt':args.generations,
            'scope':'One native-tool DAPO update, CP2/DP4, 8 GPUs on spare node; no checkpoint/evaluation export',
            'ray_address':address}
    (root/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    raise SystemExit(proc.returncode)


if __name__=='__main__':
    main()
