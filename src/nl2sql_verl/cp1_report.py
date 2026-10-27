"""Render persisted CP1 metrics without altering the trainer or its checkpoint state."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=Path('outputs/cp1_training'));args=p.parse_args()
    rows={}
    for path in sorted(args.run_dir.glob('metrics*.jsonl')):
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            rows.setdefault(row['step'],{}).update(row['data'])
    names={'critic/score/mean':'train_reward','reward/result/mean':'train_execution_accuracy','val-core/nl2sql_bird/acc/mean@1':'dev_execution_accuracy','val-aux/nl2sql_bird/reward/mean@1':'dev_reward','actor/grad_norm':'grad_norm','timing_s/step':'step_seconds','timing_s/testing':'eval_seconds','timing_s/save_checkpoint':'checkpoint_seconds'}
    summary=[{'step':s,**{new:r[old] for old,new in names.items() if old in r}} for s,r in sorted(rows.items())]
    args.run_dir.mkdir(parents=True,exist_ok=True)
    (args.run_dir/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(json.dumps(summary[-1] if summary else {}));return
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for ax,key,label in [(axes[0],'train_reward','Training reward'),(axes[0],'dev_reward','Dev reward (640)'),(axes[1],'train_execution_accuracy','Training execution accuracy'),(axes[1],'dev_execution_accuracy','Dev execution accuracy (640)')]:
        values=[(r['step'],r[key]) for r in summary if key in r]
        if values:ax.plot(*zip(*values),marker='o',label=label)
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.lines:ax.legend()
    axes[1].set_ylim(0,1);axes[1].set_xlabel('Optimizer step');axes[0].set_ylabel('Reward');axes[1].set_ylabel('Accuracy')
    fig.suptitle('BIRD GRPO — TP4 / DP2 / CP1');fig.tight_layout();fig.savefig(args.run_dir/'reward_progress.png',dpi=150);plt.close(fig)
    print(json.dumps(summary[-1] if summary else {}))


if __name__=='__main__':main()
