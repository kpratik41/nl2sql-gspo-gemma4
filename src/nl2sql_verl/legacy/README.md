# Historical backend and launcher boundaries

Current recorded training uses **native verl FSDP2 + FlexAttention + Ulysses
CP=2**, via `run_screened_training.py` and
`configs/dapo_screened_8k_16gpu.yaml`. Setup uses `setup_fsdp_env.sh` and the
repository-root package snapshot. Historical code is preserved for experiments
and reproducibility; no backend or training-policy change is made here.

- `automodel/`: the optional NeMo engine, registration, legacy preflight,
  setup/launch scripts, and development requirements.
- `launchers/`: old CP1 management/report/smoke tools and the eight-B200
  `run_fsdp` supervisor. This directory name does not imply FSDP2 itself is legacy.
- Original top-level names remain thin compatibility entry points. Existing
  module commands and Hydra `_target_` names keep resolving. Historical YAML
  profiles stay in `../configs/` because current profiles share their defaults.
- Tests and numerical probes stay in `../tests/`; native task code still uses
  `agent_loop.py`, `tools.py`, `task.py`, and `consensus/`. Do not remove those
  shared modules or the projection/gradient/checkpoint validation probes.
- `run_h200.py` remains at its existing path: the screened launcher uses its W&B
  authentication helper. It also contains historical supervision behavior.

The refactor preserves arguments, paths, and recipe settings. Persisted source
hashes from an earlier run are historical evidence, not authorization to mark
modified source as validated. On a destination cluster, run the normal numerical,
capacity, and integration checks; never rewrite readiness hashes to bypass them.
