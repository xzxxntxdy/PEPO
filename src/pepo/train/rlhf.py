"""PEPO RLHF entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional, Union

def _get_cli_argv(args: Optional[Union[List[str], "PepoRLHFArguments"]]) -> Optional[List[str]]:
    if args is None:
        return sys.argv[1:]
    if isinstance(args, list):
        return args
    return None


def _get_torchrun_args() -> Optional[List[str]]:
    nproc_per_node = os.getenv("NPROC_PER_NODE")
    nnodes = os.getenv("NNODES")
    if nproc_per_node is None and nnodes is None:
        return None
    if any(os.getenv(env_key) is not None for env_key in ("LOCAL_RANK", "RANK", "WORLD_SIZE")):
        return None

    torchrun_args: List[str] = []
    for env_key in ("NPROC_PER_NODE", "MASTER_PORT", "NNODES", "NODE_RANK", "MASTER_ADDR"):
        env_val = os.getenv(env_key)
        if env_val is not None:
            torchrun_args.extend([f"--{env_key.lower()}", env_val])
    return torchrun_args


def _maybe_launch_with_torchrun(args: Optional[Union[List[str], "PepoRLHFArguments"]]) -> None:
    argv = _get_cli_argv(args)
    torchrun_args = _get_torchrun_args()
    if argv is None or torchrun_args is None:
        return

    cmd = [sys.executable, "-m", "torch.distributed.run", *torchrun_args, "--module", "pepo.train.rlhf", *argv]
    print(f"run sh: `{' '.join(cmd)}`", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    raise SystemExit(0)


from swift.llm.train.rlhf import SwiftRLHF
from swift.trainers.trainer_factory import TrainerFactory

from .args import PepoRLHFArguments


def install_pepo_runtime() -> None:
    TrainerFactory.TRAINER_MAPPING["grpo"] = "pepo.train.grpo_trainer.PepoGRPOTrainer"
    TrainerFactory.TRAINING_ARGS_MAPPING["grpo"] = "pepo.train.args.PepoGRPOConfig"


class PepoSwiftRLHF(SwiftRLHF):
    args_class = PepoRLHFArguments


def main(args: Optional[Union[List[str], PepoRLHFArguments]] = None):
    _maybe_launch_with_torchrun(args)
    install_pepo_runtime()
    return PepoSwiftRLHF(args).main()


if __name__ == "__main__":
    main()
