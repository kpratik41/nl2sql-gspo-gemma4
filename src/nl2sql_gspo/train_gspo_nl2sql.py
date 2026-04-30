import argparse
import os


DEFAULT_REWARD_WEIGHTS = [0.25, 1.0, 2.5, 0.5, 0.5, 0.25]


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--database_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--vllm_server_base_url", type=str, default="http://127.0.0.1:8000")

    parser.add_argument("--max_prompt_length", type=int, default=16384)
    parser.add_argument("--max_completion_length", type=int, default=4096)

    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--reward_weights",
        type=parse_reward_weights,
        default=list(DEFAULT_REWARD_WEIGHTS),
        help="Comma-separated weights for format, execution, result, schema_linking, ngram, evidence rewards.",
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

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)

    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.28)
    parser.add_argument("--loss_type", type=str, default="dapo")
    parser.add_argument("--scale_rewards", type=str, default="batch")

    return parser.parse_args(argv)


def main():
    from datasets import load_dataset
    from nl2sql_gspo.data import normalize_record
    from nl2sql_gspo.model_utils import load_model_and_tokenizer
    from nl2sql_gspo.rewards import make_nl2sql_rewards
    from trl import GRPOConfig, GRPOTrainer

    args = parse_args()

    report_to = parse_csv_list(args.report_to)
    if not report_to:
        report_to = ["none"]

    logging_dir = args.logging_dir or os.path.join(args.output_dir, "tb")

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

    reward_functions = make_nl2sql_rewards(database_dir=args.database_dir)

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

        # Rollout sampling
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=0.8,
        top_p=0.95,

        # vLLM server mode
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=args.vllm_server_base_url,
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
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,

        # Important because rewards need db_id, gold_sql, evidence, messages
        remove_unused_columns=False,

        # Logging/saving
        logging_steps=args.logging_steps,
        logging_dir=logging_dir,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_total_limit=3,
        report_to=report_to,
        run_name=args.run_name,
        log_completions=True,
        num_completions_to_print=2,

        dataloader_num_workers=2,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()