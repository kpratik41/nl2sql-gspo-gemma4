import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None


def freeze_multimodal_modules(model):
    freeze_name_keywords = [
        "vision",
        "visual",
        "image",
        "vit",
        "clip",
        "siglip",
        "vision_tower",
        "image_tower",
        "audio",
        "speech",
        "whisper",
        "wav",
        "sound",
        "mm_projector",
        "multi_modal_projector",
        "multimodal_projector",
        "modality_projector",
        "connector",
        "resampler",
        "perceiver",
    ]

    llm_projection_exceptions = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        lower_name = name.lower()

        should_freeze = any(
            keyword in lower_name
            for keyword in freeze_name_keywords
        )

        if any(exception in lower_name for exception in llm_projection_exceptions):
            should_freeze = False

        if should_freeze:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()

    print(f"Frozen multimodal parameters: {frozen_count:,}")
    print(f"Trainable parameters after freezing: {trainable_count:,}")

    return model


def print_trainable_parameters(model):
    total = 0
    trainable = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    pct = 100 * trainable / max(1, total)
    print(f"Trainable params: {trainable:,} / {total:,} = {pct:.2f}%")


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model_and_tokenizer(model_name_or_path: str):
    tokenizer = load_tokenizer(model_name_or_path)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        print("Loaded model with AutoModelForCausalLM.")

    except Exception as exc:
        if AutoModelForImageTextToText is None:
            raise RuntimeError(
                "AutoModelForCausalLM failed, and AutoModelForImageTextToText "
                "is not available in this Transformers version."
            ) from exc

        print("AutoModelForCausalLM failed. Falling back to AutoModelForImageTextToText.")
        print(f"Original error: {exc}")

        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        model = freeze_multimodal_modules(model)

    model.config.use_cache = False
    print_trainable_parameters(model)

    return model, tokenizer


def load_inference_model_and_tokenizer(model_name_or_path: str):
    tokenizer = load_tokenizer(model_name_or_path)

    causal_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            **causal_kwargs,
        )
        print("Loaded inference model with AutoModelForCausalLM.")
    except Exception as exc:
        print("AutoModelForCausalLM with flash_attention_2 failed for inference.")
        print(f"Original error: {exc}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **causal_kwargs,
            )
            print("Loaded inference model with AutoModelForCausalLM fallback.")
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