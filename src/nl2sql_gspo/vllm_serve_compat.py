from trl.scripts import vllm_serve as trl_vllm_serve


_ORIGINAL_SAMPLING_PARAMS = trl_vllm_serve.SamplingParams


def _sampling_params_compat(**kwargs):
    try:
        return _ORIGINAL_SAMPLING_PARAMS(**kwargs)
    except TypeError as exc:
        if "truncate_prompt_tokens" not in kwargs or "truncate_prompt_tokens" not in str(exc):
            raise
        compat_kwargs = dict(kwargs)
        compat_kwargs.pop("truncate_prompt_tokens", None)
        return _ORIGINAL_SAMPLING_PARAMS(**compat_kwargs)


trl_vllm_serve.SamplingParams = _sampling_params_compat


def main() -> None:
    parser = trl_vllm_serve.make_parser()
    (script_args,) = parser.parse_args_and_config()
    trl_vllm_serve.main(script_args)


if __name__ == "__main__":
    main()