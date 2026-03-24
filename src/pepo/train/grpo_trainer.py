"""PEPO GRPO trainer built on top of ms-swift."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from swift.trainers.rlhf_trainer import grpo_trainer as swift_grpo


class PepoGRPOTrainer(swift_grpo.GRPOTrainer):
    _VISION_SCORE_TYPES = {"cosine", "l2", "l1", "dot"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = self.args
        self.img_token = getattr(args, "img_token", None)
        self.use_vision_weights = bool(getattr(args, "use_vision_weights", False))
        self.vision_score_type = getattr(args, "vision_score_type", "cosine") or "cosine"
        if self.vision_score_type not in self._VISION_SCORE_TYPES:
            raise ValueError(
                "vision_score_type must be one of 'cosine', 'l2', 'l1', 'dot', "
                f"got {self.vision_score_type!r}"
            )
        self.gate_alpha = getattr(args, "gate_alpha", 0.05)
        self.gate_temperature = getattr(args, "gate_temperature", 1.0)
        self.img_id = None
        if self.img_token is not None:
            self.img_id = self.processing_class.convert_tokens_to_ids(self.img_token)
        if self.use_vision_weights and self.img_id is None:
            raise ValueError("img_token must be provided when use_vision_weights is enabled.")

    @staticmethod
    def _masked_percentiles(values: Optional[torch.Tensor], mask: torch.Tensor, accelerator) -> Optional[Dict[int, float]]:
        if values is None:
            return None
        mask = mask.bool()
        if mask.numel() == 0:
            return None
        masked = values.masked_fill(~mask, float("nan")).float()
        flat = masked.flatten()
        if flat.numel() == 0 or torch.isnan(flat).all():
            return None
        quantiles = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], device=flat.device)
        pct_vals = torch.nanquantile(flat, quantiles)
        gathered = accelerator.gather_for_metrics(pct_vals)
        if gathered.numel() % quantiles.numel() == 0:
            gathered = gathered.view(-1, quantiles.numel())
            pct_vals = torch.nanmean(gathered, dim=0)
        pct_vals = pct_vals.detach()
        return {
            100: pct_vals[0].item(),
            80: pct_vals[1].item(),
            60: pct_vals[2].item(),
            40: pct_vals[3].item(),
            20: pct_vals[4].item(),
            0: pct_vals[5].item(),
        }

    def _get_image_token_ranges(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        ranges: List[torch.Tensor] = []
        for ids in input_ids:
            idxs = (ids == self.img_id).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                idxs = torch.arange(1, ids.numel(), device=ids.device)
            ranges.append(idxs)
        return ranges

    @swift_grpo.patch_profiling_decorator
    def _compute_vision_scores_cosine(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        token_ranges = self._get_image_token_ranges(input_ids)
        device = hidden_states[0].device
        dtype = hidden_states[0].dtype
        scores = [torch.zeros(logits_to_keep, device=device, dtype=dtype) for _ in range(batch_size)]

        for layer_hidden in hidden_states:
            for row_idx, idxs in enumerate(token_ranges):
                if idxs.numel() == 0:
                    continue
                vision_hidden = layer_hidden[row_idx].index_select(0, idxs)
                gen_hidden = layer_hidden[row_idx, -logits_to_keep:]
                gen_hidden = torch.nn.functional.normalize(gen_hidden, dim=-1)
                vision_hidden = torch.nn.functional.normalize(vision_hidden, dim=-1)
                sim_matrix = gen_hidden @ vision_hidden.T
                scores[row_idx] += sim_matrix.mean(dim=1)

        num_layers = max(1, len(hidden_states))
        for row_idx in range(batch_size):
            scores[row_idx] /= num_layers
        return torch.stack(scores, dim=0)

    @swift_grpo.patch_profiling_decorator
    def _compute_vision_scores_l2(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        token_ranges = self._get_image_token_ranges(input_ids)
        device = hidden_states[0].device
        dtype = hidden_states[0].dtype
        scores = [torch.zeros(logits_to_keep, device=device, dtype=dtype) for _ in range(batch_size)]

        for layer_hidden in hidden_states:
            for row_idx, idxs in enumerate(token_ranges):
                if idxs.numel() == 0:
                    continue
                vision_hidden = layer_hidden[row_idx].index_select(0, idxs)
                gen_hidden = layer_hidden[row_idx, -logits_to_keep:]
                gen_sq = (gen_hidden * gen_hidden).sum(dim=1, keepdim=True)
                vis_sq = (vision_hidden * vision_hidden).sum(dim=1, keepdim=True).T
                dist_sq = gen_sq + vis_sq - 2.0 * (gen_hidden @ vision_hidden.T)
                dist_sq = dist_sq.clamp_min(0.0)
                dist = torch.sqrt(dist_sq + 1e-6)
                scores[row_idx] += -dist.mean(dim=1)

        num_layers = max(1, len(hidden_states))
        for row_idx in range(batch_size):
            scores[row_idx] /= num_layers
        return torch.stack(scores, dim=0)

    @swift_grpo.patch_profiling_decorator
    def _compute_vision_scores_l1(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        token_ranges = self._get_image_token_ranges(input_ids)
        device = hidden_states[0].device
        dtype = hidden_states[0].dtype
        accum_dtype = torch.float32 if device.type == "cuda" and dtype == torch.bfloat16 else dtype
        scores = torch.zeros((batch_size, logits_to_keep), device=device, dtype=accum_dtype)

        for layer_hidden in hidden_states:
            for row_idx, idxs in enumerate(token_ranges):
                if idxs.numel() == 0:
                    continue
                vision_hidden = layer_hidden[row_idx].index_select(0, idxs)
                gen_hidden = layer_hidden[row_idx, -logits_to_keep:]
                if gen_hidden.is_cuda and gen_hidden.dtype == torch.bfloat16:
                    gen_hidden = gen_hidden.float()
                    vision_hidden = vision_hidden.float()
                dist = torch.cdist(gen_hidden, vision_hidden, p=1)
                scores[row_idx] += (-dist.mean(dim=1)).to(accum_dtype)

        scores /= max(1, len(hidden_states))
        return scores.to(dtype)

    @swift_grpo.patch_profiling_decorator
    def _compute_vision_scores_dot(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        token_ranges = self._get_image_token_ranges(input_ids)
        device = hidden_states[0].device
        dtype = hidden_states[0].dtype
        scores = [torch.zeros(logits_to_keep, device=device, dtype=dtype) for _ in range(batch_size)]

        for layer_hidden in hidden_states:
            for row_idx, idxs in enumerate(token_ranges):
                if idxs.numel() == 0:
                    continue
                vision_hidden = layer_hidden[row_idx].index_select(0, idxs)
                gen_hidden = layer_hidden[row_idx, -logits_to_keep:]
                sim_matrix = gen_hidden @ vision_hidden.T
                scores[row_idx] += sim_matrix.mean(dim=1)

        num_layers = max(1, len(hidden_states))
        for row_idx in range(batch_size):
            scores[row_idx] /= num_layers
        return torch.stack(scores, dim=0)

    def _compute_vision_scores(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        if self.vision_score_type == "l2":
            return self._compute_vision_scores_l2(hidden_states, input_ids, logits_to_keep)
        if self.vision_score_type == "l1":
            return self._compute_vision_scores_l1(hidden_states, input_ids, logits_to_keep)
        if self.vision_score_type == "dot":
            return self._compute_vision_scores_dot(hidden_states, input_ids, logits_to_keep)
        return self._compute_vision_scores_cosine(hidden_states, input_ids, logits_to_keep)

    @swift_grpo.patch_profiling_decorator
    def _get_pepo_per_token_outputs(
        self,
        model,
        inputs,
        compute_entropy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size = inputs["seq_lengths"].shape[0] if self.template.padding_free else inputs["input_ids"].shape[0]
        mode = "train" if self.model.training else "eval"
        expected_bs = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        should_chunk = self.dynamic_num_samples and any(swift_grpo.gather_object([batch_size > expected_bs]))
        if not should_chunk:
            return self._get_pepo_per_token_outputs_single(model, inputs, compute_entropy=compute_entropy)
        return self._get_pepo_per_token_outputs_chunked(model, inputs, compute_entropy=compute_entropy)

    def _get_pepo_per_token_outputs_single(
        self,
        model,
        inputs,
        compute_entropy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.template.sequence_parallel_size > 1:
            raise ValueError("PEPO vision weights do not support sequence_parallel_size > 1.")

        logits_to_keep = inputs["logits_to_keep"]
        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"].bool()

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {
                "logits_to_keep",
                "completion_mask",
                "ref_per_token_logps",
                "advantages",
                "old_per_token_logps",
                "truncated_mask",
                "seq_lengths",
            }
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        outputs = model(return_dict=True, output_hidden_states=True, **model_inputs)
        logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]
        hidden_states = outputs.hidden_states
        logits = logits / self.temperature
        input_ids = input_ids[:, -logits_to_keep:]
        logps = swift_grpo.selective_log_softmax(logits, input_ids)

        if compute_entropy:
            entropies = swift_grpo.entropy_from_logits(logits)
            entropy_for_gate = entropies
        else:
            entropies = None
            with torch.no_grad():
                entropy_for_gate = swift_grpo.entropy_from_logits(logits)

        eps = 1e-6
        vision_scores = self._compute_vision_scores(hidden_states, inputs["input_ids"], logits_to_keep)
        valid_counts = completion_mask.sum(dim=1, keepdim=True).clamp(min=1)

        v_min = vision_scores.masked_fill(~completion_mask, float("inf")).min(dim=1, keepdim=True).values
        v_max = vision_scores.masked_fill(~completion_mask, float("-inf")).max(dim=1, keepdim=True).values
        v_norm = (vision_scores - v_min) / (v_max - v_min).clamp(min=eps)
        v_norm = torch.where(completion_mask, v_norm, torch.zeros_like(v_norm))

        h_min = entropy_for_gate.masked_fill(~completion_mask, float("inf")).min(dim=1, keepdim=True).values
        h_max = entropy_for_gate.masked_fill(~completion_mask, float("-inf")).max(dim=1, keepdim=True).values
        h_norm = (entropy_for_gate - h_min) / (h_max - h_min).clamp(min=eps)
        h_norm = torch.where(completion_mask, h_norm, torch.zeros_like(h_norm))

        mixed = v_norm + h_norm
        mixed_mean = (mixed * completion_mask).sum(dim=1, keepdim=True) / valid_counts
        mixed_centered = mixed - mixed_mean
        gate = 1.0 + self.gate_alpha * torch.tanh(mixed_centered)

        gated_logits = vision_scores * gate / self.gate_temperature
        neg_inf = torch.full_like(gated_logits, -float("inf"))
        masked_logits = torch.where(completion_mask, gated_logits, neg_inf)
        weights = torch.softmax(masked_logits, dim=1)
        vision_weights = weights * valid_counts.to(weights.dtype)

        return logps, entropies, vision_weights

    def _get_pepo_per_token_outputs_chunked(
        self,
        model,
        inputs,
        compute_entropy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size = inputs["seq_lengths"].shape[0] if self.template.padding_free else inputs["input_ids"].shape[0]
        mode = "train" if self.model.training else "eval"
        chunk_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        batch_sizes = swift_grpo.gather_object([batch_size])
        chunks_per_device = [(bs + chunk_size - 1) // chunk_size for bs in batch_sizes]
        max_chunks = max(chunks_per_device)
        new_chunk_size = (batch_size + max_chunks - 1) // max_chunks

        all_logps: List[torch.Tensor] = []
        all_entropies: List[torch.Tensor] = []
        all_vision_weights: List[torch.Tensor] = []

        for chunk_idx in range(max_chunks):
            start_idx = chunk_idx * new_chunk_size
            end_idx = min(start_idx + new_chunk_size, batch_size)
            if start_idx >= end_idx:
                continue
            chunk_inputs = self.get_chunked_inputs(inputs, start_idx, end_idx)
            chunk_logps, chunk_entropies, chunk_vision_weights = self._get_pepo_per_token_outputs_single(
                model,
                chunk_inputs,
                compute_entropy=compute_entropy,
            )
            all_logps.append(chunk_logps)
            if compute_entropy and chunk_entropies is not None:
                all_entropies.append(chunk_entropies)
            all_vision_weights.append(chunk_vision_weights)

        device = self.accelerator.device
        final_logps = torch.cat(all_logps, dim=0) if all_logps else torch.empty(0, device=device)
        final_entropies = torch.cat(all_entropies, dim=0) if all_entropies else None
        final_vision_weights = torch.cat(all_vision_weights, dim=0) if all_vision_weights else torch.empty(0, device=device)
        return final_logps, final_entropies, final_vision_weights

    def _compute_loss_and_metrics(self, model, inputs):
        mode = "train" if self.model.training else "eval"

        completion_mask = inputs["completion_mask"]
        truncated_mask = inputs["truncated_mask"]
        if self.template.padding_free:
            lengths = inputs["seq_lengths"]
        if self.use_vision_weights:
            per_token_logps, entropies, vision_weights = self._get_pepo_per_token_outputs(
                model,
                inputs,
                compute_entropy=self.compute_entropy,
            )
        else:
            per_token_logps, entropies = super()._get_per_token_logps_and_entropies(
                model,
                inputs,
                compute_entropy=self.compute_entropy,
            )
            vision_weights = None

        entropy_mask = None
        entropy_metrics = {}

        if self.compute_entropy:
            entropies = entropies.masked_fill(completion_mask == 0, float("nan"))
            if self.args.log_entropy:
                if self.template.padding_free:
                    entropy_list = torch.split(entropies, lengths.tolist())
                    per_completion_entropies_mean = torch.stack([torch.nanmean(e) for e in entropy_list])
                else:
                    per_completion_entropies_mean = torch.nanmean(entropies, dim=1)
                global_entropies = swift_grpo.gather(per_completion_entropies_mean)
                entropy_metrics = {
                    "entropy_logs": global_entropies.tolist(),
                    "entropy_mean": global_entropies.nanmean().item(),
                    "entropy_max": swift_grpo.nanmax(global_entropies).item(),
                    "entropy_min": swift_grpo.nanmin(global_entropies).item(),
                }

            if self.args.top_entropy_quantile < 1.0:
                entropy_threshold = torch.nanquantile(entropies.flatten().float(), 1 - self.top_entropy_quantile)
                entropy_metrics["entropy_threshold"] = entropy_threshold.item()
                entropy_mask = entropies >= entropy_threshold

        if self.args.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                swift_grpo.logger.info(
                    "All completions are overlong and truncated, resulting in NaN some values for some metrics."
                )
            if self.template.padding_free:
                truncated_mask = torch.repeat_interleave(truncated_mask, lengths).unsqueeze(0)
                assert truncated_mask.shape == completion_mask.shape
            else:
                truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask)
            completion_mask = completion_mask & (~truncated_mask)

        entropy_percentiles = self._masked_percentiles(entropies, completion_mask, self.accelerator)
        vision_weights_percentiles = self._masked_percentiles(vision_weights, completion_mask, self.accelerator)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level in ["sequence", "sequence_token"]:
            if self.template.padding_free:
                log_ratio_list = torch.split(log_ratio.squeeze(0), lengths.tolist())
                mask_list = torch.split(completion_mask.squeeze(0), lengths.tolist())
                seq_weights = [(lr * mask).sum() / mask.sum().clamp(min=1.0) for lr, mask in zip(log_ratio_list, mask_list)]
                seq_level_log_weights = torch.stack(seq_weights).to(log_ratio.dtype).unsqueeze(-1)
                if self.importance_sampling_level == "sequence":
                    log_importance_weights = seq_level_log_weights
                else:
                    seq_level_log_weight = seq_level_log_weights.detach()
                    seq_level_log_weight = torch.repeat_interleave(seq_level_log_weight, lengths).unsqueeze(0)
                    log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
            else:
                seq_level_log_weights = ((log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
                if self.importance_sampling_level == "sequence":
                    log_importance_weights = seq_level_log_weights
                else:
                    seq_level_log_weight = seq_level_log_weights.detach()
                    log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}.")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        if self.template.padding_free:
            advantages = advantages[-coef_1.shape[1]:]
            per_token_loss1 = coef_1 * advantages.unsqueeze(0)
            per_token_loss2 = coef_2 * advantages.unsqueeze(0)
        else:
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vision_weights and vision_weights is not None:
            max_steps = max(1, getattr(self.state, "max_steps", 0) or 0)
            progress = min(1.0, getattr(self.state, "global_step", 0) / max_steps)
            per_token_loss = per_token_loss * (1 - progress + progress * vision_weights)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            if self.template.padding_free:
                loss_list = torch.split(per_token_loss.squeeze(0), lengths.tolist())
                mask_list = torch.split(completion_mask.squeeze(0), lengths.tolist())
                sample_loss = [(loss * mask).sum() / mask.sum().clamp(min=1.0) for loss, mask in zip(loss_list, mask_list)]
                loss = torch.stack(sample_loss).mean()
            else:
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            batch_size = lengths.shape[0] if self.template.padding_free else inputs["input_ids"].shape[0]
            loss = (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(values: torch.Tensor) -> torch.Tensor:
            if values.shape[1] == 1:
                return values.mean()
            return (values * completion_mask).sum() / completion_token_count

        metrics_data = {
            "mode": mode,
            "entropy": entropy_metrics,
            "entropy_percentiles": entropy_percentiles,
            "vision_weights_percentiles": vision_weights_percentiles,
            "completion_mask": completion_mask,
            "completion_token_count": completion_token_count,
        }

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            metrics_data["kl"] = self.accelerator.gather_for_metrics(mean_kl).nanmean().item()

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)

        metrics_data["clipping"] = {
            "low_clip_mean": gathered_low_clip.nanmean().item(),
            "low_clip_min": swift_grpo.nanmin(gathered_low_clip).item(),
            "high_clip_mean": gathered_high_clip.nanmean().item(),
            "high_clip_max": swift_grpo.nanmax(gathered_high_clip).item(),
            "region_clip_mean": gathered_clip_ratio.nanmean().item(),
        }
        if mode == "train" and self.chord_sft_iterator is not None:
            loss = swift_grpo.compute_chord_loss(self, grpo_loss=loss)

        return loss, metrics_data

    def _update_metrics(self, metrics_data):
        super()._update_metrics(metrics_data)
        mode = metrics_data["mode"]

        entropy_percentiles = metrics_data.get("entropy_percentiles")
        if entropy_percentiles:
            for percentile, value in entropy_percentiles.items():
                self._metrics[mode][f"entropy/p{percentile}"].append(value)

        vision_weights_percentiles = metrics_data.get("vision_weights_percentiles")
        if vision_weights_percentiles:
            for percentile, value in vision_weights_percentiles.items():
                self._metrics[mode][f"vision_weights/p{percentile}"].append(value)
