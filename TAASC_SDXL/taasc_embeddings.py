"""SDXL phrase compensation (paper Eq. 5), with frozen encoding cached per prompt."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch


def token_spans(tokenizer, prompt: str, phrase: str):
    """Match every complete BPE occurrence, excluding BOS/EOS/padding."""
    full = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(phrase, add_special_tokens=False)
    # CLIP reserves one BOS and one EOS position.
    usable = full[: tokenizer.model_max_length - 2]
    spans = []
    if ids:
        for start in range(len(usable) - len(ids) + 1):
            if usable[start:start + len(ids)] == ids:
                spans.append((start + 1, len(ids)))
    return spans, full, ids


@dataclass
class ConditioningCache:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor | None
    pooled_prompt_embeds: torch.Tensor
    negative_pooled_prompt_embeds: torch.Tensor | None
    phrase_embeds: dict
    position_info: dict
    missing_phrases: list
    overlap_mode: str = 'mean'

    def compensate(self, weights: dict):
        """Always interpolate against the original C; never accumulate across rounds.

        Overlapping active phrases average their displacements by default;
        overlap_mode='last' retains the old last-phrase-wins behavior.
        At lambda=0 the entire conditioning remains bitwise identical to base.
        """
        updated = self.prompt_embeds.clone()
        deltas = torch.zeros_like(updated) if self.overlap_mode == 'mean' else None
        counts = torch.zeros(updated.shape[1], dtype=torch.int32) if deltas is not None else None
        for phrase, info in self.position_info.items():
            weight = float(weights[phrase])
            if not 0 <= weight <= 1:
                raise ValueError(f"Invalid lambda for {phrase!r}: {weight}")
            for start, width in info["spans"]:
                original = self.prompt_embeds[:, start:start + width]
                if weight == 0.0:
                    updated[:, start:start + width] = original
                    continue
                independent = self.phrase_embeds[phrase]
                if deltas is None:
                    updated[:, start:start + width] = (1 - weight) * original + weight * independent
                else:
                    deltas[:, start:start + width] += weight * (independent - original)
                    counts[start:start + width] += 1
        if deltas is not None:
            updated += deltas / counts.clamp_min(1).to(updated.dtype)[None,:,None]
        return dict(
            prompt_embeds=updated,
            negative_prompt_embeds=self.negative_prompt_embeds,
            pooled_prompt_embeds=self.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds,
        )


@torch.no_grad()
def build_conditioning_cache(pipe, prompt, phrases, negative_prompt=None, clip_skip=None,
                             encode_batch_size=1, strict_alignment=False, overlap_mode='mean'):
    """Use the official dual-encoder path, including SDXL's zero negative condition.

    Encoding each phrase through both encoders then interpolating the concatenated
    vector equals interpolating separately in CLIP-L and bigG and concatenating.
    Pooled positive/negative embeddings are those of the full original prompt.
    """
    for name in ("tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2"):
        if getattr(pipe, name, None) is None:
            raise ValueError(f"A complete SDXL base pipeline is required: missing {name}")
    if encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive")
    if overlap_mode not in ('mean','last'):
        raise ValueError('overlap_mode must be mean or last')
    phrases = list(dict.fromkeys(phrases))
    info, missing, claimed = {}, [], {}
    for phrase in phrases:
        spans1, full1, ids1 = token_spans(pipe.tokenizer, prompt, phrase)
        spans2, full2, ids2 = token_spans(pipe.tokenizer_2, prompt, phrase)
        if not spans1 or not spans2:
            missing.append(phrase)
            continue
        if spans1 != spans2:
            raise ValueError(f"SDXL tokenizers disagree for {phrase!r}: {spans1} vs {spans2}")
        positions = sorted({p for start, width in spans1 for p in range(start, start + width)})
        overlap = {claimed[p] for p in positions if p in claimed}
        if overlap:
            warnings.warn(f"Overlapping phrase {phrase!r} with {sorted(overlap)}; policy={overlap_mode}.")
        claimed.update({p: phrase for p in positions})
        info[phrase] = {"positions": positions, "spans": spans1,
                        "phrase_token_ids_encoder_1": ids1, "phrase_token_ids_encoder_2": ids2}
    if missing:
        message = (f"Phrases absent from the first 75 prompt tokens: {missing}. "
                   "Use exact caption substrings; unmatched phrases receive no control.")
        if strict_alignment:
            raise ValueError(message)
        warnings.warn(message)
    for tokenizer in (pipe.tokenizer, pipe.tokenizer_2):
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > tokenizer.model_max_length - 2:
            warnings.warn("Prompt exceeds CLIP context: the tail will be truncated, as in native SDXL.")
            break

    base = pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt,
                              device=pipe._execution_device, num_images_per_prompt=1,
                              do_classifier_free_guidance=True, clip_skip=clip_skip)
    base = tuple(value.detach().cpu().clone() if value is not None else None for value in base)
    independent = {}
    aligned = list(info)
    for start in range(0, len(aligned), encode_batch_size):
        batch = aligned[start:start + encode_batch_size]
        embeddings = pipe.encode_prompt(prompt=batch, device=pipe._execution_device,
                                        num_images_per_prompt=1,
                                        do_classifier_free_guidance=False,
                                        clip_skip=clip_skip)[0]
        for i, phrase in enumerate(batch):
            width = len(info[phrase]["phrase_token_ids_encoder_1"])
            # Position 0 is BOS, not the first semantic token.
            independent[phrase] = embeddings[i:i + 1, 1:1 + width].detach().cpu().clone()
        del embeddings
    return ConditioningCache(base[0], base[1], base[2], base[3], independent, info, missing, overlap_mode)
