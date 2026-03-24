"""PEPO-specific training argument extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from swift.llm.argument.rlhf_args import RLHFArguments as SwiftRLHFArguments
from swift.trainers.rlhf_arguments import GRPOConfig as SwiftGRPOConfig


@dataclass
class PepoRLHFArguments(SwiftRLHFArguments):
    gate_alpha: Optional[float] = None
    gate_temperature: Optional[float] = None
    img_token: Optional[str] = None
    use_vision_weights: bool = False
    vision_score_type: Literal["cosine", "l2", "l1", "dot"] = "cosine"

    def __post_init__(self):
        super().__post_init__()
        self._init_pepo_defaults()

    def _init_pepo_defaults(self) -> None:
        if self.rlhf_type != "grpo":
            return
        if self.gate_alpha is None:
            self.gate_alpha = 0.05
        if self.gate_temperature is None:
            self.gate_temperature = 1.0
        if self.use_vision_weights and not self.img_token:
            raise ValueError("img_token must be provided when use_vision_weights is enabled.")


@dataclass
class PepoGRPOConfig(SwiftGRPOConfig):
    gate_alpha: Optional[float] = None
    gate_temperature: Optional[float] = None
    img_token: Optional[str] = None
    use_vision_weights: bool = False
    vision_score_type: Literal["cosine", "l2", "l1", "dot"] = "cosine"

    def __post_init__(self):
        super().__post_init__()
        if self.gate_alpha is None:
            self.gate_alpha = 0.05
        if self.gate_temperature is None:
            self.gate_temperature = 1.0
        if self.use_vision_weights and not self.img_token:
            raise ValueError("img_token must be provided when use_vision_weights is enabled.")
