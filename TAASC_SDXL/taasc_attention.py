"""TA-ASC cross-attention control for Diffusers SDXL.

The controller implements the paper's training-free attention branch:

1. aggregate unmodified phrase attention maps over U-Net layers and heads;
2. estimate temporal stability from consecutive denoising timesteps;
3. build a high-response mask from the preceding stable map; and
4. add ``log(1 + gain)`` to conditional-branch phrase logits in that mask.

Only inference-time tensors are changed. Model parameters are never updated.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionFeedbackTarget:
    """One low-confidence semantic phrase controlled during inference."""

    key: str
    phrase: str
    kind: str
    token_positions: Tuple[int, ...]
    confidence: float
    gain: float


class TrainingFreeAttentionController:
    """Aggregate, stabilize, and locally strengthen phrase attention."""

    def __init__(self) -> None:
        self.enabled = False
        self.targets: Dict[str, AttentionFeedbackTarget] = {}
        self.batch_size = 1
        self.map_size = 16
        self.stability_steps = 3
        self.stability_threshold = 0.85
        self.alpha = 0.4
        self.beta = 0.3
        self.gamma = 0.3
        self.centroid_sigma = 0.15
        self.high_response_ratio = 0.2
        self.latent_height = 128
        self.latent_width = 128
        self.mask_mode = "paper"

        self.current_timestep: Optional[float] = None
        self._step_map_sum: Dict[str, torch.Tensor] = {}
        self._step_map_count: Dict[str, int] = defaultdict(int)
        self._last_maps: Dict[str, torch.Tensor] = {}
        self._stability_history: Dict[str, deque] = {}
        self._stable: Dict[str, bool] = {}
        self._region_cache: Dict[Tuple[str, int, str], torch.Tensor] = {}
        self.reset_runtime()

    def configure(
        self,
        targets: Iterable[dict],
        *,
        batch_size: int = 1,
        map_size: int = 16,
        stability_steps: int = 3,
        stability_threshold: float = 0.85,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        centroid_sigma: float = 0.15,
        high_response_ratio: float = 0.2,
        latent_height: int = 128,
        latent_width: int = 128,
        mask_mode: str = "paper",
    ) -> None:
        if stability_steps < 1:
            raise ValueError("stability_steps must be positive")
        if not 0.0 <= stability_threshold <= 1.0:
            raise ValueError("stability_threshold must be in [0, 1]")
        if centroid_sigma <= 0.0:
            raise ValueError("centroid_sigma must be positive")
        if not 0.0 < high_response_ratio <= 1.0:
            raise ValueError("high_response_ratio must be in (0, 1]")
        if map_size < 2:
            raise ValueError("map_size must be at least 2")
        if latent_height < 1 or latent_width < 1:
            raise ValueError("latent dimensions must be positive")

        coefficient_sum = float(alpha + beta + gamma)
        if not all(math.isfinite(v) and v >= 0 for v in (alpha, beta, gamma)) or coefficient_sum <= 0.0:
            raise ValueError("alpha + beta + gamma must be positive")
        if mask_mode not in ("paper", "sd14"):
            raise ValueError("mask_mode must be paper or sd14")

        # Duplicate semantic entries can point to the same prompt span. Retain
        # the lowest confidence, which also carries the largest required gain.
        merged: Dict[Tuple[str, Tuple[int, ...]], AttentionFeedbackTarget] = {}
        for index, raw in enumerate(targets):
            positions = tuple(
                sorted(
                    {
                        int(position)
                        for position in raw.get("token_positions", [])
                        if int(position) >= 0
                    }
                )
            )
            if not positions:
                continue

            confidence = max(0.0, min(1.0, float(raw.get("confidence", 1.0))))
            gain = max(0.0, float(raw.get("gain", 0.0)))
            phrase = str(raw.get("phrase", "")).strip()
            kind = str(raw.get("kind", "unknown"))
            target = AttentionFeedbackTarget(
                key=str(raw.get("key", f"target:{index}:{phrase}")),
                phrase=phrase,
                kind=kind,
                token_positions=positions,
                confidence=confidence,
                gain=gain,
            )
            merge_key = (phrase.casefold(), positions)
            previous = merged.get(merge_key)
            if previous is None or target.confidence < previous.confidence:
                merged[merge_key] = target

        self.targets = {
            target.key: target
            for target in merged.values()
            if target.gain > 0.0
        }
        self.enabled = bool(self.targets)
        self.batch_size = max(1, int(batch_size))
        self.map_size = int(map_size)
        self.stability_steps = int(stability_steps)
        self.stability_threshold = float(stability_threshold)
        self.alpha = float(alpha) / coefficient_sum
        self.beta = float(beta) / coefficient_sum
        self.gamma = float(gamma) / coefficient_sum
        self.centroid_sigma = float(centroid_sigma)
        self.high_response_ratio = float(high_response_ratio)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.mask_mode = mask_mode
        self.reset_runtime()

        if self.enabled:
            summary = ", ".join(
                f"{target.kind}:{target.phrase} "
                f"score={target.confidence:.3f} gain={target.gain:.3f}"
                for target in self.targets.values()
            )
            print(f"[TA-ASC attention] active targets: {summary}")

    def reset_runtime(self) -> None:
        self.current_timestep = None
        self._step_map_sum = {}
        self._step_map_count = defaultdict(int)
        self._last_maps = {}
        self._stability_history = {
            key: deque(maxlen=max(5, self.stability_steps))
            for key in self.targets
        }
        self._stable = {key: False for key in self.targets}
        self._region_cache = {}
        self.events = []
        self.observed_layers = 0
        self.enhanced_chunks = 0
        self.step_index = -1

    def clear_tensor_state(self) -> None:
        """Drop every tensor at the end of an image; preserve scalar diagnostics."""
        self.current_timestep = None
        self._step_map_sum.clear()
        self._step_map_count.clear()
        self._last_maps.clear()
        self._region_cache.clear()
        self._stability_history.clear()
        self._stable.clear()

    @staticmethod
    def _as_timestep(timestep) -> Optional[float]:
        if timestep is None:
            return None
        if torch.is_tensor(timestep):
            if timestep.numel() == 0:
                return None
            return float(timestep.detach().reshape(-1)[0].item())
        return float(timestep)

    def begin_timestep(self, timestep) -> None:
        value = self._as_timestep(timestep)
        if value is None:
            return
        # Count U-Net evaluations, retaining fractional/repeated scheduler times.
        if self.current_timestep is not None:
            self._finalize_current_timestep()
        self.current_timestep = value
        self.step_index += 1

    def finish(self) -> None:
        if self.current_timestep is not None:
            self._finalize_current_timestep()
        self.current_timestep = None

    def _spatial_shape(self, query_count: int) -> Optional[Tuple[int, int]]:
        # U-Net feature maps preserve the latent aspect ratio while being
        # downsampled by powers of two. Check those exact shapes first.
        for factor in (1, 2, 4, 8, 16, 32):
            if self.latent_height % factor or self.latent_width % factor:
                continue
            height = self.latent_height // factor
            width = self.latent_width // factor
            if height * width == query_count:
                return height, width

        # For uncommon sizes whose downsampling rounds an odd dimension,
        # choose the factor pair closest to the original latent aspect ratio.
        target_ratio = self.latent_width / self.latent_height
        best_shape = None
        best_error = float("inf")
        for height in range(1, int(math.sqrt(query_count)) + 1):
            if query_count % height:
                continue
            width = query_count // height
            for candidate_height, candidate_width in (
                (height, width),
                (width, height),
            ):
                ratio = candidate_width / candidate_height
                error = abs(math.log(max(ratio, 1e-8) / target_ratio))
                if error < best_error:
                    best_error = error
                    best_shape = (candidate_height, candidate_width)
        return best_shape

    def _resize_and_normalize(
        self,
        token_map: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        shape = self._spatial_shape(int(token_map.numel()))
        if shape is None:
            return None
        height, width = shape
        token_map = token_map.reshape(1, 1, height, width).float()
        if height != self.map_size or width != self.map_size:
            token_map = F.interpolate(
                token_map,
                size=(self.map_size, self.map_size),
                mode="bilinear",
                align_corners=False,
            )
        token_map = token_map[0, 0]
        token_map = token_map - token_map.min()
        return token_map / (token_map.sum() + 1e-8)

    def observe(self, attention: torch.Tensor, heads: int) -> None:
        """Observe raw, unmodified cross-attention probabilities."""

        if not self.enabled or self.current_timestep is None:
            return
        if attention.ndim != 3 or attention.shape[0] % heads != 0:
            return

        batch = attention.shape[0] // heads
        query_count = attention.shape[1]
        context_count = attention.shape[2]
        view = attention.detach().reshape(batch, heads, query_count, context_count)

        # SDXL Diffusers concatenates CFG as [unconditional, conditional].
        conditional_count = min(self.batch_size, batch)
        view = view[batch - conditional_count :]
        mean_attention = view.float().mean(dim=(0, 1))

        phrase_maps = {}
        for key, target in self.targets.items():
            positions = [
                position
                for position in target.token_positions
                if position < context_count
            ]
            if not positions:
                continue
            phrase_maps[key] = mean_attention[:, positions].sum(dim=-1)
        self.observe_phrase_maps(phrase_maps)

    def observe_phrase_maps(self, phrase_maps: dict) -> None:
        """One whole spatial vector per target, counted once per layer (Eq. 12)."""
        if not self.enabled or self.current_timestep is None:
            return
        self.observed_layers += 1
        for key, token_map in phrase_maps.items():
            token_map = self._resize_and_normalize(token_map)
            if token_map is None:
                continue
            if key not in self._step_map_sum:
                self._step_map_sum[key] = token_map
            else:
                self._step_map_sum[key] = self._step_map_sum[key] + token_map
            self._step_map_count[key] += 1

    def _high_response_mask(self, attention_map: torch.Tensor) -> torch.Tensor:
        flat = attention_map.flatten()
        count = max(1, int(math.ceil(flat.numel() * self.high_response_ratio)))
        # Exact top-k prevents ties from accidentally selecting the whole map.
        if self.mask_mode == "sd14":
            return attention_map >= torch.topk(flat, k=count).values[-1]
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[torch.argsort(flat, descending=True, stable=True)[:count]] = True
        return mask.reshape_as(attention_map)

    @staticmethod
    def _centroid(attention_map: torch.Tensor) -> torch.Tensor:
        height, width = attention_map.shape
        mass = attention_map / attention_map.sum().clamp_min(1e-8)
        ys = torch.linspace(0.0, 1.0, height, dtype=mass.dtype, device=mass.device)
        xs = torch.linspace(0.0, 1.0, width, dtype=mass.dtype, device=mass.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack(((mass * grid_x).sum(), (mass * grid_y).sum()))

    def _pair_stability(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        previous = previous.float()
        current = current.float()
        if previous.sum() <= 1e-8 or current.sum() <= 1e-8:
            # Uniform maps become zero after min subtraction: no localization.
            return 0.0, 0.0, 0.0, float("inf")
        cosine = F.cosine_similarity(
            previous.flatten().unsqueeze(0),
            current.flatten().unsqueeze(0),
        ).clamp(0.0, 1.0).item()

        previous_region = self._high_response_mask(previous)
        current_region = self._high_response_mask(current)
        intersection = (previous_region & current_region).sum().item()
        union = (previous_region | current_region).sum().item()
        iou = float(intersection / union) if union else 1.0

        distance = torch.linalg.vector_norm(
            self._centroid(previous) - self._centroid(current)
        ).item()
        centroid_term = math.exp(-distance / self.centroid_sigma)
        combined = (
            self.alpha * cosine
            + self.beta * iou
            + self.gamma * centroid_term
        )
        return combined, cosine, iou, distance

    def _finalize_current_timestep(self) -> None:
        for key, target in self.targets.items():
            count = self._step_map_count.get(key, 0)
            if count <= 0 or key not in self._step_map_sum:
                self._stable[key] = False
                self._stability_history[key].clear()
                self._last_maps.pop(key, None)
                continue

            current = (self._step_map_sum[key] / count).detach().cpu()
            previous = self._last_maps.get(key)
            if previous is not None:
                combined, cosine, iou, distance = self._pair_stability(
                    previous,
                    current,
                )
                history = self._stability_history[key]
                history.append(combined)
                was_stable = self._stable[key]
                recent = list(history)[-self.stability_steps :]
                self._stable[key] = (
                    len(recent) == self.stability_steps
                    and previous.sum().item() > 1e-8
                    and current.sum().item() > 1e-8
                    and all(
                        score >= self.stability_threshold
                        for score in recent
                    )
                )
                self.events.append({"step": self.step_index, "timestep": self.current_timestep,
                                    "phrase": target.phrase, "C": combined, "cosine": cosine,
                                    "iou": iou, "centroid_distance": distance if math.isfinite(distance) else None,
                                    "stable_for_next_step": self._stable[key], "layers": count})
                if self._stable[key] and not was_stable:
                    print(
                        f"[TA-ASC attention] stable phrase={target.phrase!r} "
                        f"t={self.current_timestep:g} C={combined:.3f} "
                        f"cos={cosine:.3f} IoU={iou:.3f} D={distance:.3f}; "
                        f"enable gain={target.gain:.3f}"
                    )

            self._last_maps[key] = current

        self._step_map_sum = {}
        self._step_map_count = defaultdict(int)
        self._region_cache = {}

    def _region_for_query(
        self,
        key: str,
        query_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        last_map = self._last_maps.get(key)
        shape = self._spatial_shape(query_count)
        if last_map is None or shape is None:
            return None
        height, width = shape

        cache_key = (key, query_count, str(device))
        cached = self._region_cache.get(cache_key)
        if cached is None:
            if self.mask_mode == "paper":
                # Eq. (16): TopMask on the previous aggregate, then resize mask.
                cached = F.interpolate(
                    self._high_response_mask(last_map).float()[None, None],
                    size=(height, width), mode="nearest",
                )[0, 0].flatten().to(device=device)
                self._region_cache[cache_key] = cached
                return cached.to(dtype=dtype)
            resized = F.interpolate(
                last_map.reshape(1, 1, self.map_size, self.map_size),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            cached = (
                self._high_response_mask(resized)
                .flatten()
                .float()
                .to(device=device)
            )
            self._region_cache[cache_key] = cached
        return cached.to(dtype=dtype)

    def enhance_logits(self, similarity: torch.Tensor, heads: int,
                       query_offset: int = 0, total_queries: int | None = None) -> torch.Tensor:
        """Apply the semantic, temporal, and spatial gates to logits."""

        if not self.enabled or not any(self._stable.values()):
            return similarity
        if similarity.ndim != 3 or similarity.shape[0] % heads != 0:
            return similarity

        batch_heads, query_count, context_count = similarity.shape
        batch = batch_heads // heads
        conditional_count = min(self.batch_size, batch)
        conditional_start = batch - conditional_count
        result = similarity
        view = None

        for key, target in self.targets.items():
            if not self._stable.get(key, False) or target.gain <= 0.0:
                continue
            positions = [
                position
                for position in target.token_positions
                if position < context_count
            ]
            if not positions:
                continue
            region = self._region_for_query(
                key,
                total_queries or query_count,
                similarity.device,
                similarity.dtype,
            )
            if region is None:
                continue
            region = region[query_offset:query_offset + query_count]

            if view is None:
                result = similarity.clone()
                view = result.reshape(batch, heads, query_count, context_count)
            spatial_boost = (
                region.reshape(1, 1, query_count)
                * math.log1p(target.gain)
            )
            for position in positions:
                view[conditional_start:, :, :, position] += spatial_boost

        if result is not similarity:
            self.enhanced_chunks += 1
        return result


class TAASCCrossAttnProcessor:
    """Diffusers attention processor exposing probabilities and logits.

    SDPA/xFormers kernels do not expose attention probabilities. This processor
    intentionally uses explicit batched matrix multiplications so TA-ASC can
    measure the unmodified maps and then alter selected logits.
    """

    def __init__(
        self,
        controller: TrainingFreeAttentionController,
        layer_name: str,
        original_processor=None,
        query_chunk_size: int = 1024,
    ) -> None:
        self.controller = controller
        self.layer_name = layer_name
        self.original_processor = original_processor
        if query_chunk_size < 0:
            raise ValueError("query_chunk_size must be nonnegative")
        self.query_chunk_size = query_chunk_size

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if not self.controller.enabled and self.original_processor is not None:
            return self.original_processor(attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, temb=temb)
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size,
                channel,
                height * width,
            ).transpose(1, 2)

        is_cross_attention = encoder_hidden_states is not None
        context = encoder_hidden_states
        if context is None:
            context = hidden_states

        batch_size, sequence_length, _ = context.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask,
            sequence_length,
            batch_size,
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if getattr(attn, "norm_q", None) is not None or getattr(attn, "norm_k", None) is not None:
            raise ValueError("This processor targets original SDXL base attention (without Q/K normalization)")
        if is_cross_attention and attn.norm_cross:
            context = attn.norm_encoder_hidden_states(context)
        key = attn.to_k(context)
        value = attn.to_v(context)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # FP32 logits and query chunks avoid fp16 overflow and keep memory bounded.
        # Each chunk contains ALL heads; aggregation still counts each layer once.
        total_queries = query.shape[1]
        chunk_size = self.query_chunk_size or total_queries
        # Preallocate the result: no per-chunk GPU output list or torch.cat copy.
        output_buffer = torch.empty_like(query)
        phrase_maps = {name: torch.zeros(total_queries, dtype=torch.float32, device='cpu')
                       for name in self.controller.targets}
        calculation_key = key.float().transpose(-1, -2)
        for start in range(0, total_queries, chunk_size):
            stop = min(start + chunk_size, total_queries)
            similarity = torch.bmm(query[:, start:stop].float() * attn.scale, calculation_key)
            if attention_mask is not None:
                mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[:, start:stop]
                if mask.dtype == torch.bool:
                    similarity = similarity.masked_fill(~mask, -torch.finfo(similarity.dtype).max)
                else:
                    similarity = similarity + mask.float()
            raw_attention = similarity.softmax(dim=-1)
            if is_cross_attention and self.controller.enabled:
                view = raw_attention.reshape(batch_size, attn.heads, stop - start, -1)
                mean = view[-self.controller.batch_size:].mean(dim=(0, 1))
                for target_key, target in self.controller.targets.items():
                    positions = [p for p in target.token_positions if p < mean.shape[-1]]
                    if positions:
                        phrase_maps[target_key][start:stop].copy_(mean[:, positions].sum(-1).detach().cpu())
                enhanced = self.controller.enhance_logits(similarity, attn.heads, start, total_queries)
                probs = enhanced.softmax(-1) if enhanced is not similarity else raw_attention
            else:
                probs = raw_attention
            output_buffer[:, start:stop] = torch.bmm(probs.to(value.dtype), value)
            if is_cross_attention and self.controller.enabled:
                del view, mean, enhanced
            del similarity, raw_attention, probs
        if is_cross_attention:
            self.controller.observe_phrase_maps(phrase_maps)
        hidden_states = output_buffer
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size,
                channel,
                height,
                width,
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


@dataclass
class AttentionInstallation:
    """Installed processor state, including a reversible restore method."""

    unet: object
    original_processors: dict
    pre_hook_handle: object
    cross_attention_layers: int

    def restore(self) -> None:
        self.pre_hook_handle.remove()
        self.unet.set_attn_processor(dict(self.original_processors))


def install_taasc_attention_processors(
    unet,
    controller: TrainingFreeAttentionController,
    query_chunk_size: int = 1024,
) -> AttentionInstallation:
    """Install TA-ASC only on SDXL U-Net cross-attention (attn2) layers."""

    original_processors = dict(unet.attn_processors)
    processors = {}
    cross_attention_layers = 0
    for name, processor in original_processors.items():
        if name.endswith("attn2.processor"):
            processors[name] = TAASCCrossAttnProcessor(controller, name, processor, query_chunk_size)
            cross_attention_layers += 1
        else:
            processors[name] = processor

    if cross_attention_layers == 0:
        raise RuntimeError(
            "No SDXL U-Net attn2 processors were found; check the checkpoint architecture."
        )
    unet.set_attn_processor(dict(processors))

    def begin_step_hook(module, args, kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) >= 2:
            timestep = args[1]
        controller.begin_timestep(timestep)

    pre_hook_handle = unet.register_forward_pre_hook(
        begin_step_hook,
        with_kwargs=True,
    )
    return AttentionInstallation(
        unet=unet,
        original_processors=original_processors,
        pre_hook_handle=pre_hook_handle,
        cross_attention_layers=cross_attention_layers,
    )
