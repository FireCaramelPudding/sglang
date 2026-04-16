from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.managers.io_struct import KVTransformSpec
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.hf_transformers_utils import get_hf_text_config, get_rope_config

logger = logging.getLogger(__name__)
_RESCALE_SHAPE_MISMATCH_FAIL_FAST = get_bool_env_var(
    "SGLANG_KV_GRAFT_RESCALE_SHAPE_MISMATCH_FAIL_FAST"
)


@dataclass
class GraftMaterializeResult:
    device_indices: torch.Tensor
    token_ids: List[int]
    composite: bool
    transform: Optional[KVTransformSpec]


class BaseKVGraftMaterializer:
    def __init__(self, kv_pool: KVCache, rope_theta: float = 10000.0):
        self.kv_pool = kv_pool
        self.rope_theta = float(rope_theta) if rope_theta and rope_theta > 0 else 10000.0

    def transform_segment(
        self,
        *,
        source_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        transform: KVTransformSpec,
        origin_start: int,
        current_prefix_len_before_append: int,
        layer_ids: Sequence[int],
        reference_indices: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError()

    @staticmethod
    def _summarize_norms_for_rescale(norms: torch.Tensor) -> torch.Tensor:
        # Local HF reference path uses [batch, head, seq, dim].
        # SGLang KV pool slices are token-major, e.g. [token, head, dim] for MHA
        # and [token, 1, dim] for MLA rope/nope parts. We therefore need to
        # average over the sequence/token axis differently depending on layout.
        if norms.dim() == 4:
            return norms.mean(dim=-2, keepdim=True)
        if norms.dim() in (2, 3):
            return norms.mean(dim=0, keepdim=True)
        if norms.dim() <= 1:
            return norms.mean()

        # Conservative fallback: reduce all leading token-like axes and keep the
        # tail shape for broadcasting. This is less specific, but avoids server
        # crashes if a backend returns an unexpected KV slice shape.
        reduce_dims = tuple(range(max(0, norms.dim() - 2)))
        if not reduce_dims:
            return norms
        return norms.mean(dim=reduce_dims, keepdim=True)

    @staticmethod
    def _rescale_tensor(
        src: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        src_f = src.float()
        tgt_f = tgt.float()

        src_norm = torch.norm(src_f, p=2, dim=-1, keepdim=True)
        tgt_norm = torch.norm(tgt_f, p=2, dim=-1, keepdim=True)
        if src_norm.numel() == 0 or tgt_norm.numel() == 0:
            return src

        src_norm = BaseKVGraftMaterializer._summarize_norms_for_rescale(src_norm)
        tgt_norm = BaseKVGraftMaterializer._summarize_norms_for_rescale(tgt_norm)
        scale = tgt_norm / (src_norm + eps)

        # Be defensive against any future backend/layout mismatch. If the
        # preserved axes still do not broadcast, log the mismatch explicitly.
        # In debug mode we fail fast; otherwise we fall back to a global scalar
        # scale so production runs can continue.
        try:
            torch.broadcast_shapes(src_f.shape, scale.shape)
        except RuntimeError as exc:
            message = (
                "[kv_graft rescale shape mismatch] "
                f"src_shape={tuple(src_f.shape)} tgt_shape={tuple(tgt_f.shape)} "
                f"src_norm_shape={tuple(src_norm.shape)} tgt_norm_shape={tuple(tgt_norm.shape)} "
                f"scale_shape={tuple(scale.shape)} fail_fast={_RESCALE_SHAPE_MISMATCH_FAIL_FAST} "
                f"error={exc}"
            )
            logger.warning(message)
            if _RESCALE_SHAPE_MISMATCH_FAIL_FAST:
                raise RuntimeError(message) from exc
            scale = tgt_norm.mean() / (src_norm.mean() + eps)

        return (src_f * scale).to(src.dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _rope_shift_tensor(
        self, k: torch.Tensor, delta: int, origin_start: int = 0
    ) -> torch.Tensor:
        if delta == 0:
            return k
        dim = int(k.shape[-1])
        if dim % 2 != 0 or k.dim() < 2:
            return k
        device = k.device
        dtype = k.dtype
        theta = self.rope_theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )

        # Match local HF path exactly: undo source-position rotation, then apply
        # target-position rotation. This uses absolute source/target positions
        # instead of relying on a pure relative-delta composition shortcut.
        #
        # For grafted KV slices, token axis is leading axis in SGLang
        # ([token, head, dim] for MHA, [token, 1, dim] for MLA rope part).
        seq_len = int(k.shape[0])
        old_positions = torch.arange(
            int(origin_start),
            int(origin_start) + seq_len,
            device=device,
            dtype=torch.float32,
        )
        new_positions = old_positions + float(delta)

        old_angles = old_positions[:, None] * inv_freq[None, :]
        new_angles = new_positions[:, None] * inv_freq[None, :]

        cos_old = torch.cos(old_angles).to(dtype=dtype).unsqueeze(1)
        sin_old = torch.sin(old_angles).to(dtype=dtype).unsqueeze(1)
        cos_new = torch.cos(new_angles).to(dtype=dtype).unsqueeze(1)
        sin_new = torch.sin(new_angles).to(dtype=dtype).unsqueeze(1)

        cos_old = torch.cat([cos_old, cos_old], dim=-1)
        sin_old = torch.cat([sin_old, sin_old], dim=-1)
        cos_new = torch.cat([cos_new, cos_new], dim=-1)
        sin_new = torch.cat([sin_new, sin_new], dim=-1)

        k_base = k * cos_old - self._rotate_half(k) * sin_old
        return k_base * cos_new + self._rotate_half(k_base) * sin_new


class MHAGraftMaterializer(BaseKVGraftMaterializer):
    def transform_segment(
        self,
        *,
        source_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        transform: KVTransformSpec,
        origin_start: int,
        current_prefix_len_before_append: int,
        layer_ids: Sequence[int],
        reference_indices: Optional[torch.Tensor] = None,
    ):
        delta = current_prefix_len_before_append - origin_start
        copy_only = bool(getattr(transform, "rescale_params", None) and transform.rescale_params.get("copy_only"))
        alias_bypass = bool(getattr(transform, "rescale_params", None) and transform.rescale_params.get("alias_bypass"))
        logger.info(
            "[kv_graft materialize MHA] src_len=%s dst_len=%s ref_len=%s origin_start=%s current_prefix_len=%s delta=%s rope=%s rescale=%s copy_only=%s alias_bypass=%s",
            int(source_indices.numel()),
            int(dst_indices.numel()),
            int(reference_indices.numel()) if reference_indices is not None else 0,
            origin_start,
            current_prefix_len_before_append,
            delta,
            getattr(transform, "rope_shift", None),
            getattr(transform, "rescale_profile", None),
            copy_only,
            alias_bypass,
        )
        sample_src = source_indices[: min(8, int(source_indices.numel()))].tolist()
        sample_dst = dst_indices[: min(8, int(dst_indices.numel()))].tolist()
        for layer_id in layer_ids:
            k = self.kv_pool.get_key_buffer(layer_id)[source_indices]
            v = self.kv_pool.get_value_buffer(layer_id)[source_indices]
            orig_k = k if copy_only else None
            orig_v = v if copy_only else None
            if copy_only and layer_id == layer_ids[0]:
                logger.info(
                    "[kv_graft copy_check MHA before] layer=%s sample_src=%s sample_dst=%s k_shape=%s v_shape=%s",
                    layer_id,
                    sample_src,
                    sample_dst,
                    tuple(k.shape),
                    tuple(v.shape),
                )
            if not copy_only and (
                transform.rescale_profile == "match_stats"
                and reference_indices is not None
                and reference_indices.numel() > 0
            ):
                ref_k = self.kv_pool.get_key_buffer(layer_id)[reference_indices]
                ref_v = self.kv_pool.get_value_buffer(layer_id)[reference_indices]
                k = self._rescale_tensor(k, ref_k)
                v = self._rescale_tensor(v, ref_v)
            # Match the local HF path: statistics are aligned in the source
            # frame first, then K is rotated into the target position.
            if not copy_only and transform.rope_shift in ("on", "auto") and delta != 0:
                k = self._rope_shift_tensor(k, delta, origin_start=origin_start)
            self.kv_pool.get_key_buffer(layer_id)[dst_indices] = k
            self.kv_pool.get_value_buffer(layer_id)[dst_indices] = v
            if copy_only and layer_id == layer_ids[0]:
                copied_k = self.kv_pool.get_key_buffer(layer_id)[dst_indices]
                copied_v = self.kv_pool.get_value_buffer(layer_id)[dst_indices]
                k_diff = (copied_k.float() - orig_k.float()).abs().max().item() if copied_k.numel() > 0 else 0.0
                v_diff = (copied_v.float() - orig_v.float()).abs().max().item() if copied_v.numel() > 0 else 0.0
                logger.info(
                    "[kv_graft copy_check MHA after] layer=%s max_k_diff=%s max_v_diff=%s",
                    layer_id,
                    k_diff,
                    v_diff,
                )


class MLAGraftMaterializer(BaseKVGraftMaterializer):
    def transform_segment(
        self,
        *,
        source_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        transform: KVTransformSpec,
        origin_start: int,
        current_prefix_len_before_append: int,
        layer_ids: Sequence[int],
        reference_indices: Optional[torch.Tensor] = None,
    ):
        delta = current_prefix_len_before_append - origin_start
        copy_only = bool(getattr(transform, "rescale_params", None) and transform.rescale_params.get("copy_only"))
        alias_bypass = bool(getattr(transform, "rescale_params", None) and transform.rescale_params.get("alias_bypass"))
        logger.info(
            "[kv_graft materialize MLA] src_len=%s dst_len=%s ref_len=%s origin_start=%s current_prefix_len=%s delta=%s rope=%s rescale=%s copy_only=%s alias_bypass=%s",
            int(source_indices.numel()),
            int(dst_indices.numel()),
            int(reference_indices.numel()) if reference_indices is not None else 0,
            origin_start,
            current_prefix_len_before_append,
            delta,
            getattr(transform, "rope_shift", None),
            getattr(transform, "rescale_profile", None),
            copy_only,
            alias_bypass,
        )
        sample_src = source_indices[: min(8, int(source_indices.numel()))].tolist()
        sample_dst = dst_indices[: min(8, int(dst_indices.numel()))].tolist()
        for layer_id in layer_ids:
            layer_stub = type("LayerStub", (), {"layer_id": layer_id})()
            k_nope, k_rope = self.kv_pool.get_mla_kv_buffer(layer_stub, source_indices)
            orig_k_nope = k_nope if copy_only else None
            orig_k_rope = k_rope if copy_only else None
            if copy_only and layer_id == layer_ids[0]:
                logger.info(
                    "[kv_graft copy_check MLA before] layer=%s sample_src=%s sample_dst=%s k_nope_shape=%s k_rope_shape=%s",
                    layer_id,
                    sample_src,
                    sample_dst,
                    tuple(k_nope.shape),
                    tuple(k_rope.shape),
                )
            if not copy_only and (
                transform.rescale_profile == "match_stats"
                and reference_indices is not None
                and reference_indices.numel() > 0
            ):
                dst_k_nope, dst_k_rope = self.kv_pool.get_mla_kv_buffer(
                    layer_stub, reference_indices, dst_dtype=k_nope.dtype
                )
                k_nope = self._rescale_tensor(k_nope, dst_k_nope)
                k_rope = self._rescale_tensor(k_rope, dst_k_rope)
            if not copy_only and transform.rope_shift in ("on", "auto") and delta != 0:
                k_rope = self._rope_shift_tensor(k_rope, delta, origin_start=origin_start)
            self.kv_pool.set_mla_kv_buffer(layer_stub, dst_indices, k_nope, k_rope)
            if copy_only and layer_id == layer_ids[0]:
                copied_k_nope, copied_k_rope = self.kv_pool.get_mla_kv_buffer(layer_stub, dst_indices, dst_dtype=k_nope.dtype)
                nope_diff = (copied_k_nope.float() - orig_k_nope.float()).abs().max().item() if copied_k_nope.numel() > 0 else 0.0
                rope_diff = (copied_k_rope.float() - orig_k_rope.float()).abs().max().item() if copied_k_rope.numel() > 0 else 0.0
                logger.info(
                    "[kv_graft copy_check MLA after] layer=%s max_nope_diff=%s max_rope_diff=%s",
                    layer_id,
                    nope_diff,
                    rope_diff,
                )


def resolve_rope_theta_from_hf_config(hf_config) -> float:
    if hf_config is None:
        return 10000.0
    try:
        text_config = get_hf_text_config(hf_config)
        rope_theta, _rope_scaling = get_rope_config(text_config)
        if rope_theta is not None:
            return float(rope_theta)
    except Exception:
        pass
    try:
        rope_theta, _rope_scaling = get_rope_config(hf_config)
        if rope_theta is not None:
            return float(rope_theta)
    except Exception:
        pass
    return 10000.0
