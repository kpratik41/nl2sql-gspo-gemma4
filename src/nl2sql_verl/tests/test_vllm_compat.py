import pytest
from transformers import Gemma4Config, PretrainedConfig

from src.nl2sql_verl.vllm_compat import legacy_gemma4_config


def config(**kwargs):
    return Gemma4Config(text_config=dict(num_hidden_layers=6, hidden_size=5376,
        num_attention_heads=32, num_key_value_heads=16, head_dim=256,
        global_head_dim=512, num_global_key_value_heads=4, attention_k_eq_v=True, **kwargs))


def test_legacy_fields_preserve_both_attention_families():
    cfg = config()
    before = [(x.head_dim, x.num_key_value_heads) for x in cfg.text_config.per_layer_config]
    actual = legacy_gemma4_config(cfg).text_config
    assert (actual.head_dim, actual.num_key_value_heads) == (256, 16)
    assert (actual.global_head_dim, actual.num_global_key_value_heads) == (512, 4)
    assert [(x.head_dim, x.num_key_value_heads) for x in actual.per_layer_config] == before
    assert legacy_gemma4_config(cfg) is cfg  # Idempotent on reload.


def test_unrepresentable_heterogeneity_is_rejected():
    cfg = config(per_layer_config={0: {'head_dim': 128},
                                   5: {'head_dim': 512, 'num_key_value_heads': 4}})
    with pytest.raises(ValueError, match='varying sliding_attention'):
        legacy_gemma4_config(cfg)


def test_non_gemma_and_dummy_configs_are_untouched():
    cfg = PretrainedConfig(model_type='dummy_gemma4')
    assert legacy_gemma4_config(cfg) is cfg
