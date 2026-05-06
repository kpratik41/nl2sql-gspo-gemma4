import argparse

from trl.scripts.vllm_serve import main as trl_vllm_serve_main
from trl.scripts.vllm_serve import make_parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser(prog="python -m nl2sql_gspo.vllm_serve_compat")
    (script_args,) = parser.parse_args_and_config(args=argv)
    trl_vllm_serve_main(script_args)


if __name__ == "__main__":
    main()