"""Guard handoff boundaries without importing optional NeMo or allocating GPUs."""
from pathlib import Path
import importlib
import pytest
from benchmarks.layer_replay_report import render_report


def row(backend='eager', fingerprint='same'):
    return {'length':128,'layer':0,'backend':backend,'input_sha256':fingerprint,
            'errors':{s:{'max_abs':0.,'relative_l2':0.} for s in ['attention','projected','decoder']}}


def test_renderer_handles_both_historical_schemas_and_never_hides_failure():
    old=[{**row(), 'causal_prefix_max_abs':0.}]
    newer={'replays':[row('eager_bf16')],
           'semantics':[{'kind':'causality','max_abs':.2,'pass':False}]}
    assert 'Semantic failures: 0' in render_report(old)
    report=render_report(newer)
    assert 'Semantic failures: 1' in report and '| eager |' in report
    assert 'No semantic checks were recorded' in render_report([row()])
    with pytest.raises(ValueError,match='identical'):
        render_report([row(),row('sdpa','different')])


def test_legacy_launchers_preserve_exports_and_repository_root():
    root=Path(__file__).resolve().parents[3]
    for name in ['run_fsdp','manage_cp1','smoke_cp1','cp1_report']:
        old=importlib.import_module('src.nl2sql_verl.'+name)
        new=importlib.import_module('src.nl2sql_verl.legacy.launchers.'+name)
        assert old.main is new.main
        if hasattr(new,'REPO'):assert new.REPO==root
    from src.nl2sql_verl import preflight
    assert preflight.load_config.__module__=='src.nl2sql_verl.legacy.automodel.preflight'


def test_native_runtime_imports_without_optional_automodel():
    for name in ['fsdp_engine','fsdp_runtime','native_agent_loop','native_tools',
                 'reward_entry_native','native_runtime']:
        importlib.import_module('src.nl2sql_verl.'+name)
