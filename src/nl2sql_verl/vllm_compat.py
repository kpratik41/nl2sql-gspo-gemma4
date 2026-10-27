"""Local bridge from Transformers 5.18 Gemma4 configs to vLLM 0.19.1.

The model implementation in this vLLM pin understands sliding/global heads,
but reads the pre-5.18 field names. Never flatten arbitrary heterogeneous
configs: only the two Gemma4 attention families are representable here.
"""
from functools import partial


def legacy_gemma4_config(config, overrides=None):
    if callable(overrides):
        config = overrides(config)
    elif overrides:
        config.update(overrides)
    if getattr(config, 'model_type', None) != 'gemma4':
        return config  # Includes vLLM's dummy config used to probe overrides.
    text = config.get_text_config()
    families = {}
    for index, kind in enumerate(text.layer_types):
        layer = text.per_layer_config[index]
        shape = (layer.head_dim, layer.num_key_value_heads, layer.num_attention_heads)
        if kind not in {'sliding_attention', 'full_attention'}:
            raise ValueError(f'Unsupported Gemma4 attention family: {kind}')
        if families.setdefault(kind, shape) != shape:
            raise ValueError(f'vLLM 0.19.1 cannot represent varying {kind} head shapes')
    sliding = families.get('sliding_attention', families.get('full_attention'))
    full = families.get('full_attention', sliding)
    if sliding[2] != full[2]:
        raise ValueError('vLLM 0.19.1 requires a common query-head count')
    if sliding[1] != full[1] and not text.attention_k_eq_v:
        raise ValueError('Different global KV-head counts require attention_k_eq_v')
    # Validate the inherited defaults before exposing them to the legacy reader.
    raw = vars(text)
    if (raw['head_dim'], raw['num_key_value_heads'], raw['num_attention_heads']) != sliding:
        raise ValueError('Gemma4 global defaults do not match sliding attention')
    text.global_head_dim = full[0]
    text.num_global_key_value_heads = full[1]
    text.allow_global_per_layer_attribute_access = True
    return config


def _load_replica():
    import ray
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

    class Gemma4HttpServer(vLLMHttpServer):
        def _resolve_sleep_level(self):
            # vLLM 0.19.1 Gemma4 produces corrupted outputs after level-2
            # sleep even when every checkpoint tensor is reloaded. Level 1
            # preserves the model allocation state on CPU; the same refit
            # then reproduces the initial output and sampled logprobs.
            # Keep this scoped to the recipe until level 2 passes that gate.
            return 1

        async def run_server(self, args):
            args.hf_overrides = partial(legacy_gemma4_config, overrides=args.hf_overrides)
            return await super().run_server(args)

        async def run_headless(self, args):
            args.hf_overrides = partial(legacy_gemma4_config, overrides=args.hf_overrides)
            return await super().run_headless(args)

    class Gemma4Replica(vLLMReplica):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server_class = ray.remote(Gemma4HttpServer)

    return Gemma4Replica


def register():
    from verl.workers.rollout.replica import RolloutReplicaRegistry
    RolloutReplicaRegistry.register('vllm', _load_replica)
