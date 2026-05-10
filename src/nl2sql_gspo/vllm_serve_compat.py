import argparse
import logging
import time

from trl.scripts.vllm_serve import WeightSyncWorkerExtension
from trl.scripts.vllm_serve import main as trl_vllm_serve_main
from trl.scripts.vllm_serve import make_parser


LOGGER = logging.getLogger(__name__)


def _install_weight_sync_diagnostics() -> None:
    if getattr(WeightSyncWorkerExtension.update_named_param, "_nl2sql_weight_sync_wrapped", False):
        return

    original_update_named_param = WeightSyncWorkerExtension.update_named_param

    def update_named_param_with_diagnostics(self, name: str, dtype: str, shape) -> None:
        numel = 1
        for dim in shape:
            numel *= int(dim)

        dtype_name = str(dtype).split(".")[-1]
        try:
            import torch

            element_size = torch.empty((), dtype=getattr(torch, dtype_name)).element_size()
        except Exception:
            element_size = 0

        approx_gib = (numel * element_size) / float(1024**3) if element_size else 0.0
        start = time.time()
        LOGGER.warning(
            "[weight-sync] begin name=%s dtype=%s shape=%s numel=%s approx_gib=%.3f",
            name,
            dtype,
            tuple(shape),
            numel,
            approx_gib,
        )
        try:
            original_update_named_param(self, name, dtype, shape)
        except Exception:
            LOGGER.exception(
                "[weight-sync] failed name=%s dtype=%s shape=%s numel=%s approx_gib=%.3f elapsed_s=%.3f",
                name,
                dtype,
                tuple(shape),
                numel,
                approx_gib,
                time.time() - start,
            )
            raise
        LOGGER.warning(
            "[weight-sync] done name=%s elapsed_s=%.3f",
            name,
            time.time() - start,
        )

    update_named_param_with_diagnostics._nl2sql_weight_sync_wrapped = True
    WeightSyncWorkerExtension.update_named_param = update_named_param_with_diagnostics


def main(argv: list[str] | None = None) -> None:
    _install_weight_sync_diagnostics()
    parser = make_parser(prog="python -m nl2sql_gspo.vllm_serve_compat")
    (script_args,) = parser.parse_args_and_config(args=argv)
    trl_vllm_serve_main(script_args)


if __name__ == "__main__":
    main()