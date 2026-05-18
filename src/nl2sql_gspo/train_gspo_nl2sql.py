import argparse
import os
from typing import Callable


# Order: format, execution, result, table_linking, column_linking, nonnull, length_penalty
DEFAULT_REWARD_WEIGHTS = [0.2, 0.5, 2.0, 0.5, 0.5, 0.1, 0.1]


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def parse_reward_weights(value: str | None) -> list[float]:
    if not value:
        return list(DEFAULT_REWARD_WEIGHTS)

    weights = [float(item.strip()) for item in value.split(",") if item.strip()]

    if len(weights) != len(DEFAULT_REWARD_WEIGHTS):
        raise argparse.ArgumentTypeError(
            f"Expected {len(DEFAULT_REWARD_WEIGHTS)} comma-separated reward weights, got {len(weights)}"
        )

    return weights


def weight_reward_functions(
    reward_functions: list[Callable],
    reward_weights: list[float],
) -> list[Callable]:
    """Return reward wrappers for trainers that do not support reward_weights.

    AsyncGRPOTrainer sums rewards directly, unlike GRPOTrainer which accepts a
    separate `reward_weights` argument. Wrapping preserves the same reward scale
    across trainer backends.
    """

    weighted = []
    for reward_func, weight in zip(reward_functions, reward_weights, strict=True):
        reward_name = getattr(reward_func, "__name__", reward_func.__class__.__name__)

        def _weighted_reward(*args, _reward_func=reward_func, _weight=weight, **kwargs):
            values = _reward_func(*args, **kwargs)
            return [None if value is None else float(value) * _weight for value in values]

        _weighted_reward.__name__ = f"{reward_name}_x{weight:g}"
        weighted.append(_weighted_reward)
    return weighted


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--eval_limit", type=int, default=-1)
    parser.add_argument("--database_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--trainer_backend",
        type=str,
        choices=["grpo", "async_grpo"],
        default="grpo",
        help="Use the existing DynamicSamplingGRPOTrainer or TRL experimental AsyncGRPOTrainer.",
    )
    parser.add_argument(
        "--distributed_backend",
        type=str,
        choices=["deepspeed", "fsdp", "none"],
        default="deepspeed",
        help="Trainer sharding backend. Use fsdp to avoid DeepSpeed.",
    )

    parser.add_argument("--vllm_server_base_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument(
        "--vllm_group_port",
        type=int,
        default=29600,
        help="Port used for TRL/vLLM weight-update communicator; keep it outside the OS ephemeral port range.",
    )

    parser.add_argument("--max_prompt_length", type=int, default=20000)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument(
        "--enable_tool_rollouts",
        action="store_true",
        help=(
            "Pass callable NL2SQL tools to TRL's GRPO tool loop. The JSONL "
            "tool definitions are still used for chat-template rendering."
        ),
    )
    parser.add_argument(
        "--tool_db_extra_roots",
        type=str,
        default=None,
        help="Optional os.pathsep-separated DB roots appended to BIRD_DB_ROOTS for tool execution.",
    )

    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--reward_weights",
        type=parse_reward_weights,
        default=list(DEFAULT_REWARD_WEIGHTS),
        help="Comma-separated weights for format, execution, result, table_linking, column_linking, nonnull, length_penalty rewards.",
    )

    default_report_to = "wandb" if os.environ.get("WANDB_PROJECT") else "none"
    parser.add_argument(
        "--report_to",
        type=str,
        default=default_report_to,
        help="Comma-separated reporting backends such as wandb,tensorboard or none.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default=None)

    parser.add_argument("--deepspeed", type=str, default="configs/ds_zero3_bf16.json")
    parser.add_argument("--fsdp", type=str, default="full_shard auto_wrap")
    parser.add_argument("--fsdp_config", type=str, default="configs/fsdp_gemma4_bf16.json")

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_only_model", action="store_true")
    parser.add_argument("--save_latest_full_checkpoint", action="store_true")
    parser.add_argument(
        "--latest_full_checkpoint_dir_name",
        type=str,
        default="latest-full-checkpoint",
        help=(
            "Stable directory name under output_dir for the most recent full "
            "optimizer-state checkpoint used for exact resume."
        ),
    )
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_on_start", action="store_true")
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--num_completions_to_print", type=int, default=2)

    # TRL experimental AsyncGRPO controls.
    parser.add_argument("--async_max_tool_calling_iterations", type=int, default=8)
    parser.add_argument("--async_request_timeout", type=int, default=600)
    parser.add_argument("--async_max_inflight_tasks", type=int, default=-1)
    parser.add_argument("--async_max_staleness", type=int, default=4)
    parser.add_argument("--async_queue_maxsize", type=int, default=1024)
    parser.add_argument("--async_weight_sync_steps", type=int, default=1)

    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.28)
    parser.add_argument("--loss_type", type=str, default="dapo")
    parser.add_argument("--scale_rewards", type=str, default="batch")

    # GSPO / DAPO controls.
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of optimizer passes over each generation buffer (PPO mu).",
    )
    parser.add_argument(
        "--steps_per_generation",
        type=int,
        default=None,
        help="If set, overrides gradient_accumulation_steps for generation cadence.",
    )
    parser.add_argument(
        "--mask_truncated_completions",
        action="store_true",
        help="Zero out completion_mask for rollouts that hit max_completion_length.",
    )

    # Dynamic sampling (DAPO).
    parser.add_argument(
        "--enable_dynamic_sampling",
        action="store_true",
        help="Filter out groups with near-zero reward std before the loss.",
    )
    parser.add_argument(
        "--dynamic_sampling_min_std",
        type=float,
        default=1e-6,
        help="Threshold below which a group is considered to have no learning signal.",
    )
    parser.add_argument(
        "--dapo_max_rounds",
        type=int,
        default=6,
        help=(
            "Maximum number of rollout rounds per optimizer step in DAPO "
            "oversample-and-replace mode (round 1 = the dataloader's own "
            "prompts; subsequent rounds draw from a per-rank backup pool). "
            "Set to 1 to effectively disable resampling (only zero-mask flat groups)."
        ),
    )
    parser.add_argument(
        "--dapo_oversample_factor",
        type=int,
        default=1,
        help=(
            "Single-shot oversample multiplier K. When >1, each rank "
            "generates K * target_local_groups groups in ONE round and "
            "keeps the first target_local_groups heterogeneous ones; "
            "remaining slots are filled with random non-het groups whose "
            "completion_mask is zeroed (no gradient contribution). "
            "Preferred over iterative resampling when collective-ops "
            "constraints make multi-round inefficient."
        ),
    )
    parser.add_argument(
        "--dynamic_sampling_reward_name",
        type=str,
        default=None,
        help=(
            "Optional reward function name (e.g. result_reward) to use for the "
            "heterogeneity check. Default uses the aggregated/normalized advantages."
        ),
    )

    # Reward shaping.
    parser.add_argument(
        "--exec_timeout_s",
        type=float,
        default=60.0,
        help="Per-query SQL execution timeout for predicted and gold SQL during reward computation.",
    )
    parser.add_argument(
        "--reward_workers",
        type=int,
        default=1,
        help="Number of thread workers per rank for DB-backed reward functions such as result_reward.",
    )
    parser.add_argument(
        "--length_penalty_max",
        type=int,
        default=4096,
        help="L_max in DAPO Soft Overlong Punishment. Should match max_completion_length.",
    )
    parser.add_argument(
        "--length_penalty_buffer",
        type=int,
        default=512,
        help="L_cache in DAPO Soft Overlong Punishment. Linear ramp begins at L_max - L_cache.",
    )

    return parser.parse_args(argv)


def validate_required_fields(dataset, dataset_name: str) -> None:
    missing_examples = []

    for idx, row in enumerate(dataset):
        missing_fields = []
        if not row.get("db_id"):
            missing_fields.append("db_id")
        if not row.get("gold_sql"):
            missing_fields.append("gold_sql")

        if missing_fields:
            missing_examples.append(f"idx={idx} missing={','.join(missing_fields)}")

        if len(missing_examples) >= 5:
            break

    if missing_examples:
        raise ValueError(
            f"{dataset_name} is missing required fields after normalization: "
            + "; ".join(missing_examples)
        )


def filter_by_prompt_length(dataset, tokenizer, max_prompt_length: int, split_name: str):
    """Drop rows whose chat-templated prompt exceeds max_prompt_length tokens.

    TRL >= 0.29 removed `max_prompt_length` from GRPOConfig, so we filter the
    dataset up front to keep prompts within the vLLM server context window.
    """
    if max_prompt_length is None or max_prompt_length <= 0:
        return dataset

    def _token_count(tokenized):
        if hasattr(tokenized, "input_ids"):
            return len(tokenized.input_ids)
        if isinstance(tokenized, dict) and "input_ids" in tokenized:
            return len(tokenized["input_ids"])
        if hasattr(tokenized, "ids"):
            return len(tokenized.ids)
        if hasattr(tokenized, "encodings") and tokenized.encodings:
            return len(tokenized.encodings[0].ids)
        return len(tokenized)

    def _is_short(example):
        prompt = example.get("prompt") or example.get("messages")
        if not prompt:
            return False
        tools = example.get("tools") or None
        try:
            ids = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, tools=tools
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
        return _token_count(ids) <= max_prompt_length

    before = len(dataset)
    filtered = dataset.filter(_is_short, desc=f"Filtering {split_name} by prompt length")
    after = len(filtered)
    dropped = before - after
    if dropped:
        print(
            f"[{split_name}] dropped {dropped}/{before} rows with prompt > "
            f"{max_prompt_length} tokens; kept {after}."
        )
    return filtered


def main():
    from datasets import load_dataset
    from nl2sql_gspo.data import normalize_record
    from nl2sql_gspo.model_utils import load_model_and_tokenizer, load_tokenizer
    from nl2sql_gspo.rewards import make_nl2sql_rewards
    from nl2sql_gspo.tool_calling import (
        configure_tool_db_roots,
        get_grpo_tool_functions,
        get_sync_grpo_tool_functions,
    )

    args = parse_args()

    report_to = parse_csv_list(args.report_to)
    if not report_to:
        report_to = ["none"]

    logging_dir = args.logging_dir or os.path.join(args.output_dir, "tb")

    if args.trainer_backend == "async_grpo":
        tokenizer = load_tokenizer(args.model_name_or_path)
        model = None
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    raw_train_dataset = load_dataset(
        "json",
        data_files=args.train_file,
        split="train",
    )

    train_dataset = raw_train_dataset.map(
        normalize_record,
        remove_columns=raw_train_dataset.column_names,
        desc="Normalizing NL2SQL chat records",
    )
    validate_required_fields(train_dataset, "train dataset")
    train_dataset = filter_by_prompt_length(
        train_dataset, tokenizer, args.max_prompt_length, "train"
    )

    trainer_tools = None
    if args.enable_tool_rollouts:
        roots = configure_tool_db_roots(
            database_dir=args.database_dir,
            extra_roots=args.tool_db_extra_roots,
        )
        if args.trainer_backend == "async_grpo":
            trainer_tools = get_sync_grpo_tool_functions()
        else:
            trainer_tools = get_grpo_tool_functions()
        print(f"[tools] enabled GRPO callable tools with BIRD_DB_ROOTS={roots}")
    elif args.trainer_backend == "async_grpo":
        trainer_tools = None

    if args.train_limit >= 0:
        train_dataset = train_dataset.select(range(min(args.train_limit, len(train_dataset))))

    eval_dataset = None
    if args.eval_file:
        raw_eval_dataset = load_dataset(
            "json",
            data_files=args.eval_file,
            split="train",
        )

        eval_dataset = raw_eval_dataset.map(
            normalize_record,
            remove_columns=raw_eval_dataset.column_names,
            desc="Normalizing NL2SQL eval records",
        )
        validate_required_fields(eval_dataset, "eval dataset")
        eval_dataset = filter_by_prompt_length(
            eval_dataset, tokenizer, args.max_prompt_length, "eval"
        )

        if args.eval_limit >= 0:
            eval_dataset = eval_dataset.select(range(min(args.eval_limit, len(eval_dataset))))

    reward_functions = make_nl2sql_rewards(
        database_dir=args.database_dir,
        exec_timeout_s=args.exec_timeout_s,
        gold_timeout_s=args.exec_timeout_s,
        reward_workers=args.reward_workers,
        tokenizer=tokenizer,
        length_penalty_max=args.length_penalty_max,
        length_penalty_buffer=args.length_penalty_buffer,
    )

    use_deepspeed = args.distributed_backend == "deepspeed"
    use_fsdp = args.distributed_backend == "fsdp"
    deepspeed_config = args.deepspeed if use_deepspeed else None
    fsdp = args.fsdp if use_fsdp else None
    fsdp_config = args.fsdp_config if use_fsdp else None
    gradient_checkpointing = not use_fsdp

    if args.trainer_backend == "async_grpo":
        from trl.experimental.async_grpo.async_grpo_config import AsyncGRPOConfig
        from trl.experimental.async_grpo.async_grpo_trainer import AsyncGRPOTrainer

        if use_deepspeed:
            raise ValueError(
                "AsyncGRPOTrainer is experimental and only supports FSDP for distributed training. "
                "Run with --distributed_backend fsdp (recommended) or none."
            )
        if eval_dataset is not None:
            print("[async_grpo] eval_dataset is loaded for filtering checks but AsyncGRPOTrainer does not accept eval_dataset; skipping online eval.")
        if args.num_iterations != 1 or args.enable_dynamic_sampling:
            print("[async_grpo] num_iterations/dynamic-sampling controls are not used by AsyncGRPOTrainer.")
        if args.top_p != 1.0:
            print("[async_grpo] AsyncGRPOConfig in TRL 1.4.0 has no top_p parameter; ignoring --top_p.")
        if args.save_latest_full_checkpoint:
            print("[async_grpo] save_latest_full_checkpoint is only implemented by DynamicSamplingGRPOTrainer; ignoring it.")

        training_args = AsyncGRPOConfig(
            output_dir=args.output_dir,
            epsilon=args.epsilon,
            epsilon_high=args.epsilon_high,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            max_tool_calling_iterations=(
                args.async_max_tool_calling_iterations if args.enable_tool_rollouts else None
            ),
            vllm_server_base_url=args.vllm_server_base_url,
            vllm_server_timeout=600,
            request_timeout=args.async_request_timeout,
            max_inflight_tasks=args.async_max_inflight_tasks,
            max_staleness=args.async_max_staleness,
            queue_maxsize=args.async_queue_maxsize,
            weight_sync_steps=args.async_weight_sync_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            bf16=True,
            gradient_checkpointing=gradient_checkpointing,
            deepspeed=deepspeed_config,
            fsdp=fsdp,
            fsdp_config=fsdp_config,
            remove_unused_columns=False,
            logging_steps=args.logging_steps,
            logging_dir=logging_dir,
            save_steps=args.save_steps,
            save_only_model=args.save_only_model,
            save_total_limit=args.save_total_limit,
            report_to=report_to,
            run_name=args.run_name,
            log_completions=args.log_completions,
            num_completions_to_print=args.num_completions_to_print,
            dataloader_num_workers=2,
        )

        trainer = AsyncGRPOTrainer(
            model=args.model_name_or_path,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=weight_reward_functions(reward_functions, args.reward_weights),
            tools=trainer_tools if args.enable_tool_rollouts else None,
        )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        return

    from nl2sql_gspo.dynamic_sampling_trainer import DynamicSamplingGRPOTrainer
    from trl import GRPOConfig

    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # GSPO-style behavior through sequence-level importance sampling
        importance_sampling_level="sequence",

        # GRPO/GSPO objective settings
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        reward_weights=args.reward_weights,

        # PPO-style multiple updates per generation buffer
        num_iterations=args.num_iterations,
        steps_per_generation=args.steps_per_generation,
        mask_truncated_completions=args.mask_truncated_completions,

        # Rollout sampling
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=(
            args.async_max_tool_calling_iterations if args.enable_tool_rollouts else None
        ),
        temperature=args.temperature,
        top_p=args.top_p,

        # vLLM server mode
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_group_port=args.vllm_group_port,
        vllm_server_timeout=600,

        # Training
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        # Memory
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        deepspeed=deepspeed_config,
        fsdp=fsdp,
        fsdp_config=fsdp_config,

        # Important because rewards need db_id, gold_sql, evidence, messages
        remove_unused_columns=False,

        # Logging/saving
        logging_steps=args.logging_steps,
        logging_dir=logging_dir,
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_on_start=args.eval_on_start if eval_dataset is not None else False,
        save_total_limit=args.save_total_limit,
        report_to=report_to,
        run_name=args.run_name,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,

        dataloader_num_workers=2,
    )

    trainer = DynamicSamplingGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        tools=trainer_tools,
        enable_dynamic_sampling=args.enable_dynamic_sampling,
        dynamic_sampling_min_std=args.dynamic_sampling_min_std,
        dapo_max_rounds=args.dapo_max_rounds,
        dapo_oversample_factor=args.dapo_oversample_factor,
        dynamic_sampling_reward_name=args.dynamic_sampling_reward_name,
        save_latest_full_checkpoint=args.save_latest_full_checkpoint,
        latest_full_checkpoint_dir_name=args.latest_full_checkpoint_dir_name,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
