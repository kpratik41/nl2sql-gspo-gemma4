import json
from pathlib import Path


def _load_auto_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required for tokenizer loading. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return AutoTokenizer


def _load_model_classes():
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch and transformers are required for model loading. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        from transformers import AutoModelForImageTextToText
    except Exception:
        AutoModelForImageTextToText = None

    return torch, AutoModelForCausalLM, AutoModelForImageTextToText


def resolve_tokenizer_source(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path)
    if not model_path.is_dir():
        return model_name_or_path

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "processor_config.json",
        "preprocessor_config.json",
    ]

    if any((model_path / file_name).exists() for file_name in tokenizer_files):
        return model_name_or_path

    config_path = model_path / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        for field_name in ["_name_or_path", "name_or_path", "model_name_or_path"]:
            source_name = config.get(field_name)
            if isinstance(source_name, str) and source_name and source_name != model_name_or_path:
                print(
                    f"Tokenizer files not found in {model_name_or_path}; "
                    f"loading tokenizer from base model {source_name}."
                )
                return source_name

    raise ValueError(
        f"Local model path {model_name_or_path} does not contain tokenizer or processor files. "
        "For Gemma 4, point MODEL_PATH to a checkpoint or base model directory, "
        "or load the base tokenizer from google/gemma-4-31B."
    )


def load_tokenizer(model_name_or_path: str):
    AutoTokenizer = _load_auto_tokenizer()
    tokenizer_source = resolve_tokenizer_source(model_name_or_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            use_fast=True,
        )
    except ValueError as exc:
        print("Fast tokenizer load failed. Retrying with the slow tokenizer.")
        print(f"Original tokenizer error: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            use_fast=False,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not getattr(tokenizer, "chat_template", None):
        # Gemma-4 base ships without a chat template; copy it from the
        # instruct sibling (e.g. google/gemma-4-31B -> google/gemma-4-31B-it).
        instruct_source = None
        if isinstance(tokenizer_source, str) and not tokenizer_source.endswith("-it"):
            instruct_source = f"{tokenizer_source}-it"
        if instruct_source:
            try:
                instruct_tok = AutoTokenizer.from_pretrained(
                    instruct_source, trust_remote_code=True
                )
                if getattr(instruct_tok, "chat_template", None):
                    tokenizer.chat_template = instruct_tok.chat_template
                    print(
                        f"Loaded chat_template from instruct variant {instruct_source}."
                    )
            except Exception as exc:
                print(
                    f"Could not load chat_template from {instruct_source}: {exc}"
                )

    if not getattr(tokenizer, "chat_template", None):
        # Last-resort plain-text fallback.
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] | upper }}: {{ message['content'] }}\n\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}ASSISTANT:{% endif %}"
        )

    return tokenizer


def load_inference_model_and_tokenizer(model_name_or_path: str, device_map=None):
    torch, AutoModelForCausalLM, AutoModelForImageTextToText = _load_model_classes()
    tokenizer = load_tokenizer(model_name_or_path)

    causal_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if device_map is not None:
        causal_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="sdpa",
            **causal_kwargs,
        )
        print(
            "Loaded inference model with AutoModelForCausalLM using sdpa. "
            f"device_map={device_map!r}"
        )
    except Exception as exc:
        print("AutoModelForCausalLM with sdpa failed for inference.")
        print(f"Original error: {exc}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                attn_implementation="eager",
                **causal_kwargs,
            )
            print(
                "Loaded inference model with AutoModelForCausalLM using eager fallback. "
                f"device_map={device_map!r}"
            )
        except Exception as sdpa_exc:
            print("AutoModelForCausalLM with sdpa failed for inference.")
            print(f"SDPA fallback error: {sdpa_exc}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    **causal_kwargs,
                )
                print(
                    "Loaded inference model with AutoModelForCausalLM default fallback. "
                    f"device_map={device_map!r}"
                )
            except Exception as causal_exc:
                if AutoModelForImageTextToText is None:
                    raise RuntimeError(
                        "AutoModelForCausalLM inference load failed, and AutoModelForImageTextToText "
                        "is not available in this Transformers version."
                    ) from causal_exc

                print("AutoModelForCausalLM inference load failed. Falling back to AutoModelForImageTextToText.")
                print(f"Fallback trigger error: {causal_exc}")

                model = AutoModelForImageTextToText.from_pretrained(
                    model_name_or_path,
                    **causal_kwargs,
                )

    model.eval()
    model.config.use_cache = True
    return model, tokenizer
