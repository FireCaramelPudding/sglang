from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
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
# Diagnostic-only: when enabled, _rescale_tensor logs the per-token norm
# distribution of src/tgt before collapsing to a single global scale. Used to
# judge whether a global scalar rescale loses meaningful per-token detail (large
# std/mean ratio => segmented rescale would matter). Default off to avoid spam.
_RESCALE_NORM_DEBUG = get_bool_env_var("SGLANG_KV_GRAFT_RESCALE_NORM_DEBUG")


@dataclass
class GraftMaterializeResult:
    device_indices: torch.Tensor
    token_ids: List[int]
    composite: bool
    transform: Optional[KVTransformSpec]


class BaseKVGraftMaterializer:
    _context_map_cache = {}

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
    def _amplify_k_tensor(
        k: torch.Tensor, amplify_token_count: int, amplify_ratio: float
    ) -> torch.Tensor:
        if amplify_ratio == 1.0 or amplify_token_count <= 0:
            return k
        actual = min(amplify_token_count, int(k.shape[0]))
        k_out = k.clone()
        k_out[:actual] = k_out[:actual] * amplify_ratio
        return k_out

    @staticmethod
    def _compute_k_amplify_ratio(
        k: torch.Tensor, sys_token_count: int, eps: float = 1e-6
    ) -> float:
        if sys_token_count <= 0 or sys_token_count > int(k.shape[0]):
            return 1.0
        sys_k = k[:sys_token_count].float()
        overall_k = k.float()
        sys_norm = torch.norm(sys_k, p=2, dim=-1).mean()
        overall_norm = torch.norm(overall_k, p=2, dim=-1).mean()
        ratio = overall_norm / (sys_norm + eps)
        return max(1.0, min(float(ratio.item()), 5.0))

    @staticmethod
    def _oracle_layer_selected(layer_id: int, params: dict) -> bool:
        """Return whether a half-open diagnostic layer interval contains layer_id."""
        layer_start = int(params.get("oracle_layer_start", 0) or 0)
        layer_end_raw = params.get("oracle_layer_end")
        layer_end = int(layer_end_raw) if layer_end_raw is not None else 1 << 30
        if layer_start < 0 or layer_end < layer_start:
            raise ValueError(
                "oracle_mix requires 0 <= oracle_layer_start <= oracle_layer_end"
            )
        selected = layer_start <= int(layer_id) < layer_end
        if bool(params.get("oracle_layer_invert")):
            selected = not selected
        return selected

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

        if _RESCALE_NORM_DEBUG:
            # Report per-token norm spread *before* collapsing to a global scale.
            # A large std/mean ratio means tokens differ a lot in magnitude, so a
            # single global scale under/over-scales many of them (Bug 1 evidence).
            def _spread(norms: torch.Tensor) -> tuple[float, float, float, float]:
                flat = norms.float().reshape(-1)
                mean = flat.mean().item()
                std = flat.std(unbiased=False).item() if flat.numel() > 1 else 0.0
                return mean, std, flat.min().item(), flat.max().item()

            s_mean, s_std, s_min, s_max = _spread(src_norm)
            t_mean, t_std, t_min, t_max = _spread(tgt_norm)
            logger.info(
                "[kv_graft rescale norm_debug] "
                "src(mean=%.4f std=%.4f min=%.4f max=%.4f std/mean=%.3f) "
                "tgt(mean=%.4f std=%.4f min=%.4f max=%.4f std/mean=%.3f) "
                "global_scale=%.4f",
                s_mean, s_std, s_min, s_max, (s_std / (s_mean + eps)),
                t_mean, t_std, t_min, t_max, (t_std / (t_mean + eps)),
                (t_mean / (s_mean + eps)),
            )

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
        self,
        k: torch.Tensor,
        delta: float,
        origin_start: int = 0,
        compute_mode: str = "bf16_absolute",
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

        if compute_mode == "fp32_relative":
            # Standard RoPE is a block-diagonal rotation, therefore
            # R(target) R(source)^-1 == R(target - source).  Applying the
            # relative rotation once in FP32 avoids an unnecessary inverse and
            # second BF16 rotation of an already-quantized K tensor.
            angles = float(delta) * inv_freq
            cos_delta = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
            sin_delta = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
            while cos_delta.dim() < k.dim():
                cos_delta = cos_delta.unsqueeze(0)
                sin_delta = sin_delta.unsqueeze(0)
            k_fp32 = k.float()
            return (
                k_fp32 * cos_delta + self._rotate_half(k_fp32) * sin_delta
            ).to(dtype=dtype)

        if compute_mode not in ("bf16_absolute", "fp32_absolute"):
            raise ValueError(
                "rope_compute_mode must be bf16_absolute, fp32_absolute, or "
                f"fp32_relative; got {compute_mode!r}"
            )

        # Absolute control: undo the source-position rotation and then apply
        # the target-position rotation.  bf16_absolute preserves the historical
        # implementation; fp32_absolute isolates the effect of arithmetic
        # precision while keeping the formula unchanged.
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

        work_dtype = dtype if compute_mode == "bf16_absolute" else torch.float32
        cos_old = torch.cos(old_angles).to(dtype=work_dtype).unsqueeze(1)
        sin_old = torch.sin(old_angles).to(dtype=work_dtype).unsqueeze(1)
        cos_new = torch.cos(new_angles).to(dtype=work_dtype).unsqueeze(1)
        sin_new = torch.sin(new_angles).to(dtype=work_dtype).unsqueeze(1)

        cos_old = torch.cat([cos_old, cos_old], dim=-1)
        sin_old = torch.cat([sin_old, sin_old], dim=-1)
        cos_new = torch.cat([cos_new, cos_new], dim=-1)
        sin_new = torch.cat([sin_new, sin_new], dim=-1)

        k_work = k if compute_mode == "bf16_absolute" else k.float()
        k_base = k_work * cos_old - self._rotate_half(k_work) * sin_old
        return (
            k_base * cos_new + self._rotate_half(k_base) * sin_new
        ).to(dtype=dtype)

    @staticmethod
    def _rope_layer_selected(layer_id: int, params: dict) -> bool:
        """Whether RoPE should be applied to this layer.

        An omitted or empty ``rope_layer_ranges`` means all layers.  Explicit
        ranges are half-open and may be disjoint, which lets mechanism probes
        test broad layer groups without changing or re-prefilling the source KV.
        """
        ranges = params.get("rope_layer_ranges")
        if not ranges:
            return True
        for interval in ranges:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(
                    "rope_layer_ranges must contain [start, end] pairs"
                )
            start, end = int(interval[0]), int(interval[1])
            if start < 0 or end <= start:
                raise ValueError(
                    "rope_layer_ranges requires 0 <= start < end"
                )
            if start <= int(layer_id) < end:
                return True
        return False

    @staticmethod
    def _rope_shift_scale(params: dict) -> float:
        scale = float(params.get("rope_shift_scale", 1.0))
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("rope_shift_scale must be finite and non-negative")
        return scale

    @classmethod
    def _load_context_map(
        cls, params: dict, layer_id: int, device: torch.device
    ) -> torch.Tensor:
        path_value = str(params.get("context_map_path", ""))
        if not path_value:
            raise ValueError("context_linear_map requires context_map_path")
        path = str(Path(path_value).resolve())
        payload = cls._context_map_cache.get(path)
        if payload is None:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if int(payload.get("schema_version", 0)) != 1:
                raise ValueError("unsupported context map schema")
            cls._context_map_cache[path] = payload
        tp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        key = f"layer_{int(layer_id):02d}_rank_{int(tp_rank):02d}"
        matrix = payload.get("maps", {}).get(key)
        if matrix is None:
            raise ValueError(f"context map missing {key}")
        return matrix.to(device=device, dtype=torch.float32)

    def _apply_context_map_tensor(
        self,
        k: torch.Tensor,
        *,
        delta: float,
        origin_start: int,
        matrix: torch.Tensor,
    ) -> torch.Tensor:
        """De-RoPE, apply a head-wise linear map, then re-RoPE at target positions."""
        dim = int(k.shape[-1])
        if dim % 2 != 0 or k.dim() != 3:
            raise ValueError("context map requires token-major [token, head, even_dim] K")
        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)
        if tuple(matrix.shape) != (int(k.shape[1]), dim, dim):
            raise ValueError(
                f"context map shape {tuple(matrix.shape)} does not match K {tuple(k.shape)}"
            )
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, dim, 2, device=k.device, dtype=torch.float32)
                / dim
            )
        )
        old_positions = torch.arange(
            int(origin_start),
            int(origin_start) + int(k.shape[0]),
            device=k.device,
            dtype=torch.float32,
        )
        new_positions = old_positions + float(delta)

        def cos_sin(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            angles = positions[:, None] * inv_freq[None, :]
            cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
            sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
            return cos.unsqueeze(1), sin.unsqueeze(1)

        cos_old, sin_old = cos_sin(old_positions)
        cos_new, sin_new = cos_sin(new_positions)
        k_fp32 = k.float()
        canonical = k_fp32 * cos_old - self._rotate_half(k_fp32) * sin_old
        mapped = torch.einsum(
            "thd,hde->the", canonical.float(), matrix
        )
        rotated = mapped * cos_new + self._rotate_half(mapped) * sin_new
        return rotated.to(dtype=k.dtype)

    @classmethod
    def _load_receiver_bridge_map(
        cls, params: dict, layer_id: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        path_value = str(params.get("bridge_map_path", ""))
        if not path_value:
            raise ValueError("receiver_bridge_map requires bridge_map_path")
        path = str(Path(path_value).resolve())
        payload = cls._context_map_cache.get(path)
        if payload is None:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if int(payload.get("schema_version", 0)) != 2:
                raise ValueError("unsupported receiver bridge map schema")
            cls._context_map_cache[path] = payload
        tp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        key = f"layer_{int(layer_id):02d}_rank_{int(tp_rank):02d}"
        block = payload.get("maps", {}).get(key)
        if block is None:
            raise ValueError(f"receiver bridge map missing {key}")
        return {
            name: value.to(device=device, dtype=torch.float32)
            for name, value in block.items()
        }

    @staticmethod
    def _apply_receiver_bridge_map_tensor(
        value: torch.Tensor, matrix: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        if value.dim() != 3:
            raise ValueError("receiver bridge map requires [token, head, dim]")
        heads, dim = int(value.shape[1]), int(value.shape[2])
        if tuple(matrix.shape) != (heads, dim, dim):
            raise ValueError(
                f"receiver bridge matrix {tuple(matrix.shape)} != {(heads, dim, dim)}"
            )
        if tuple(bias.shape) != (heads, dim):
            raise ValueError(
                f"receiver bridge bias {tuple(bias.shape)} != {(heads, dim)}"
            )
        mapped = torch.einsum("thd,hde->the", value.float(), matrix) + bias
        return mapped.to(dtype=value.dtype)

    @staticmethod
    def _synchronize_materialization(device: torch.device) -> None:
        """Make newly materialized pages visible before a forward can consume them."""
        if device.type == "cuda":
            torch.cuda.current_stream(device=device).synchronize()


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
        _rescale_params = getattr(transform, "rescale_params", None) or {}
        verify_copy = bool(_rescale_params.get("verify_copy"))
        diagnostic_k_scale = float(
            _rescale_params.get("diagnostic_k_scale", 1.0) or 0.0
        )
        diagnostic_v_scale = float(
            _rescale_params.get("diagnostic_v_scale", 1.0) or 0.0
        )
        rope_compute_mode = str(
            _rescale_params.get("rope_compute_mode", "bf16_absolute")
            or "bf16_absolute"
        )
        rope_shift_scale = self._rope_shift_scale(_rescale_params)
        scaled_delta = float(delta) * rope_shift_scale
        context_map_enabled = transform.rescale_profile == "context_linear_map"
        # receiver_bridge_map_v1: closed-form map over receiver-computed T KV.
        bridge_map_enabled = transform.rescale_profile == "receiver_bridge_map"
        k_amplify_token_count = int(_rescale_params.get("k_amplify_token_count", 0) or 0)
        k_amplify_ratio = float(_rescale_params.get("k_amplify_ratio", 0) or 0)
        k_amplify_mode = str(_rescale_params.get("k_amplify_mode", "fixed") or "fixed")
        k_amplify_enabled = (
            not copy_only
            and k_amplify_token_count > 0
            and (k_amplify_ratio > 0 or k_amplify_mode == "auto")
        )
        logger.info(
            "[kv_graft materialize MHA] src_len=%s dst_len=%s ref_len=%s origin_start=%s current_prefix_len=%s delta=%s scaled_delta=%s rope=%s rope_scale=%s rope_ranges=%s rescale=%s copy_only=%s alias_bypass=%s diagnostic_k_scale=%s diagnostic_v_scale=%s k_amplify=%s k_amplify_ratio=%s k_amplify_mode=%s k_amplify_tokens=%s",
            int(source_indices.numel()),
            int(dst_indices.numel()),
            int(reference_indices.numel()) if reference_indices is not None else 0,
            origin_start,
            current_prefix_len_before_append,
            delta,
            scaled_delta,
            getattr(transform, "rope_shift", None),
            rope_shift_scale,
            _rescale_params.get("rope_layer_ranges"),
            getattr(transform, "rescale_profile", None),
            copy_only,
            alias_bypass,
            diagnostic_k_scale,
            diagnostic_v_scale,
            k_amplify_enabled,
            k_amplify_ratio,
            k_amplify_mode,
            k_amplify_token_count,
        )
        sample_src = source_indices[: min(8, int(source_indices.numel()))].tolist()
        sample_dst = dst_indices[: min(8, int(dst_indices.numel()))].tolist()
        max_copy_k_diff = 0.0
        max_copy_v_diff = 0.0
        for layer_id in layer_ids:
            # Explicit clones make the source snapshot independent of allocator
            # reuse and of any overlapping advanced-index assignment.
            k = self.kv_pool.get_key_buffer(layer_id)[source_indices].clone()
            v = self.kv_pool.get_value_buffer(layer_id)[source_indices].clone()
            oracle_mix = transform.rescale_profile == "oracle_mix"
            oracle_layer_selected = self._oracle_layer_selected(
                layer_id, _rescale_params
            )
            oracle_replace_k = bool(_rescale_params.get("oracle_replace_k"))
            oracle_replace_v = bool(_rescale_params.get("oracle_replace_v"))
            oracle_replace_k_this_layer = (
                oracle_mix and oracle_layer_selected and oracle_replace_k
            )
            oracle_replace_v_this_layer = (
                oracle_mix and oracle_layer_selected and oracle_replace_v
            )
            if oracle_mix and reference_indices is None:
                raise ValueError("oracle_mix requires reference_handle KV indices")
            if oracle_mix and reference_indices.numel() != source_indices.numel():
                raise ValueError(
                    "oracle_mix source/reference token lengths must be identical"
                )
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
            # CHANGED ORDER: k_amplify → rescale → RoPE
            # This ensures k_amplify works on original KV baseline,
            # not affected by rescale's scale factor.
            if k_amplify_enabled:
                actual_count = min(k_amplify_token_count, int(k.shape[0]))
                if actual_count > 0:
                    if k_amplify_mode == "auto":
                        k_amplify_ratio = self._compute_k_amplify_ratio(k, actual_count)
                    k[:actual_count] = k[:actual_count] * k_amplify_ratio
                    if layer_id == layer_ids[0]:
                        logger.info(
                            "[kv_graft k_amplify MHA] mode=%s ratio=%.4f sys_tokens=%s/%s",
                            k_amplify_mode, k_amplify_ratio, actual_count, k_amplify_token_count,
                        )
            if not copy_only and (
                transform.rescale_profile == "match_stats"
                and reference_indices is not None
                and reference_indices.numel() > 0
            ):
                ref_k = self.kv_pool.get_key_buffer(layer_id)[reference_indices]
                ref_v = self.kv_pool.get_value_buffer(layer_id)[reference_indices]
                k_before = k if layer_id == layer_ids[0] else None
                k = self._rescale_tensor(k, ref_k)
                v = self._rescale_tensor(v, ref_v)
                if layer_id == layer_ids[0] and k_before is not None:
                    src_norm = torch.norm(k_before.float(), p=2, dim=-1).mean().item()
                    rescaled_norm = torch.norm(k.float(), p=2, dim=-1).mean().item()
                    scale_factor = rescaled_norm / (src_norm + 1e-6)
                    logger.info(
                        "[kv_graft rescale MHA] src_norm=%.4f rescaled_norm=%.4f scale=%.4f",
                        src_norm, rescaled_norm, scale_factor,
                    )
            if oracle_replace_k_this_layer or oracle_replace_v_this_layer:
                ref_k = self.kv_pool.get_key_buffer(layer_id)[reference_indices]
                ref_v = self.kv_pool.get_value_buffer(layer_id)[reference_indices]
                if tuple(ref_k.shape) != tuple(k.shape) or tuple(ref_v.shape) != tuple(v.shape):
                    raise ValueError(
                        "oracle_mix source/reference KV shapes must be identical: "
                        f"source_k={tuple(k.shape)} reference_k={tuple(ref_k.shape)} "
                        f"source_v={tuple(v.shape)} reference_v={tuple(ref_v.shape)}"
                    )
                if oracle_replace_k_this_layer:
                    k = ref_k
                if oracle_replace_v_this_layer:
                    v = ref_v
                logger.info(
                    "[kv_graft oracle_mix MHA] layer=%s replace_k=%s replace_v=%s interval=[%s,%s)",
                    layer_id,
                    oracle_replace_k_this_layer,
                    oracle_replace_v_this_layer,
                    int(_rescale_params.get("oracle_layer_start", 0) or 0),
                    _rescale_params.get("oracle_layer_end"),
                )
            # Match the local HF path: statistics are aligned in the source
            # frame first, then K is rotated into the target position.
            # A locally sourced oracle K is already encoded at the receiver's
            # target positions, so only rotate K that still comes from foreign KV.
            if bridge_map_enabled:
                if oracle_replace_k_this_layer or oracle_replace_v_this_layer:
                    raise ValueError("receiver_bridge_map cannot be combined with oracle KV")
                bridge_map = self._load_receiver_bridge_map(
                    _rescale_params, layer_id, device=k.device
                )
                k = self._apply_receiver_bridge_map_tensor(
                    k, bridge_map["k_matrix"], bridge_map["k_bias"]
                )
                v = self._apply_receiver_bridge_map_tensor(
                    v, bridge_map["v_matrix"], bridge_map["v_bias"]
                )
            if context_map_enabled:
                if oracle_replace_k_this_layer:
                    raise ValueError("context_linear_map cannot be combined with oracle K")
                matrix = self._load_context_map(
                    _rescale_params, layer_id, device=k.device
                )
                k = self._apply_context_map_tensor(
                    k,
                    delta=delta,
                    origin_start=origin_start,
                    matrix=matrix,
                )
            elif (
                not copy_only
                and not oracle_replace_k_this_layer
                and transform.rope_shift in ("on", "auto")
                and scaled_delta != 0
                and self._rope_layer_selected(layer_id, _rescale_params)
            ):
                k = self._rope_shift_tensor(
                    k,
                    scaled_delta,
                    origin_start=origin_start,
                    compute_mode=rope_compute_mode,
                )
            if diagnostic_k_scale != 1.0 or diagnostic_v_scale != 1.0:
                k_before_diagnostic = k
                v_before_diagnostic = v
                if diagnostic_k_scale != 1.0:
                    k = k * diagnostic_k_scale
                if diagnostic_v_scale != 1.0:
                    v = v * diagnostic_v_scale
                if layer_id == layer_ids[0]:
                    k_change = (
                        (k.float() - k_before_diagnostic.float()).abs().max().item()
                        if k.numel() > 0
                        else 0.0
                    )
                    v_change = (
                        (v.float() - v_before_diagnostic.float()).abs().max().item()
                        if v.numel() > 0
                        else 0.0
                    )
                    logger.info(
                        "[kv_graft diagnostic MHA] layer=%s k_scale=%s v_scale=%s max_k_change=%s max_v_change=%s",
                        layer_id,
                        diagnostic_k_scale,
                        diagnostic_v_scale,
                        k_change,
                        v_change,
                    )
            self.kv_pool.get_key_buffer(layer_id)[dst_indices] = k
            self.kv_pool.get_value_buffer(layer_id)[dst_indices] = v
            if copy_only and verify_copy:
                copied_k = self.kv_pool.get_key_buffer(layer_id)[dst_indices]
                copied_v = self.kv_pool.get_value_buffer(layer_id)[dst_indices]
                k_diff = (copied_k.float() - orig_k.float()).abs().max().item() if copied_k.numel() > 0 else 0.0
                v_diff = (copied_v.float() - orig_v.float()).abs().max().item() if copied_v.numel() > 0 else 0.0
                max_copy_k_diff = max(max_copy_k_diff, k_diff)
                max_copy_v_diff = max(max_copy_v_diff, v_diff)
                if k_diff != 0.0 or v_diff != 0.0:
                    raise RuntimeError(
                        "materialized MHA copy identity failed: "
                        f"layer={layer_id} max_k_diff={k_diff} max_v_diff={v_diff}"
                    )
        self._synchronize_materialization(dst_indices.device)
        if copy_only and verify_copy:
            logger.info(
                "[kv_graft copy_check MHA complete] layers=%s max_k_diff=%s max_v_diff=%s",
                len(layer_ids),
                max_copy_k_diff,
                max_copy_v_diff,
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
        _rescale_params = getattr(transform, "rescale_params", None) or {}
        verify_copy = bool(_rescale_params.get("verify_copy"))
        rope_compute_mode = str(
            _rescale_params.get("rope_compute_mode", "bf16_absolute")
            or "bf16_absolute"
        )
        rope_shift_scale = self._rope_shift_scale(_rescale_params)
        scaled_delta = float(delta) * rope_shift_scale
        if transform.rescale_profile == "context_linear_map":
            raise ValueError("context_linear_map is currently supported only for MHA/GQA KV")
        if transform.rescale_profile == "receiver_bridge_map":
            raise ValueError("receiver_bridge_map is currently supported only for MHA/GQA KV")
        k_amplify_token_count = int(_rescale_params.get("k_amplify_token_count", 0) or 0)
        k_amplify_ratio = float(_rescale_params.get("k_amplify_ratio", 0) or 0)
        k_amplify_mode = str(_rescale_params.get("k_amplify_mode", "fixed") or "fixed")
        k_amplify_enabled = (
            not copy_only
            and k_amplify_token_count > 0
            and (k_amplify_ratio > 0 or k_amplify_mode == "auto")
        )
        logger.info(
            "[kv_graft materialize MLA] src_len=%s dst_len=%s ref_len=%s origin_start=%s current_prefix_len=%s delta=%s scaled_delta=%s rope=%s rope_scale=%s rope_ranges=%s rescale=%s copy_only=%s alias_bypass=%s k_amplify=%s k_amplify_ratio=%s k_amplify_mode=%s k_amplify_tokens=%s",
            int(source_indices.numel()),
            int(dst_indices.numel()),
            int(reference_indices.numel()) if reference_indices is not None else 0,
            origin_start,
            current_prefix_len_before_append,
            delta,
            scaled_delta,
            getattr(transform, "rope_shift", None),
            rope_shift_scale,
            _rescale_params.get("rope_layer_ranges"),
            getattr(transform, "rescale_profile", None),
            copy_only,
            alias_bypass,
            k_amplify_enabled,
            k_amplify_ratio,
            k_amplify_mode,
            k_amplify_token_count,
        )
        sample_src = source_indices[: min(8, int(source_indices.numel()))].tolist()
        sample_dst = dst_indices[: min(8, int(dst_indices.numel()))].tolist()
        max_copy_nope_diff = 0.0
        max_copy_rope_diff = 0.0
        for layer_id in layer_ids:
            layer_stub = type("LayerStub", (), {"layer_id": layer_id})()
            k_nope, k_rope = self.kv_pool.get_mla_kv_buffer(layer_stub, source_indices)
            k_nope = k_nope.clone()
            k_rope = k_rope.clone()
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
            # CHANGED ORDER: k_amplify → rescale → RoPE (same as MHA path)
            if k_amplify_enabled:
                actual_count = min(k_amplify_token_count, int(k_nope.shape[0]))
                if actual_count > 0:
                    if k_amplify_mode == "auto":
                        k_amplify_ratio = self._compute_k_amplify_ratio(k_nope, actual_count)
                    k_nope[:actual_count] = k_nope[:actual_count] * k_amplify_ratio
                    k_rope[:actual_count] = k_rope[:actual_count] * k_amplify_ratio
                    if layer_id == layer_ids[0]:
                        logger.info(
                            "[kv_graft k_amplify MLA] mode=%s ratio=%.4f sys_tokens=%s/%s",
                            k_amplify_mode, k_amplify_ratio, actual_count, k_amplify_token_count,
                        )
            if not copy_only and (
                transform.rescale_profile == "match_stats"
                and reference_indices is not None
                and reference_indices.numel() > 0
            ):
                dst_k_nope, dst_k_rope = self.kv_pool.get_mla_kv_buffer(
                    layer_stub, reference_indices, dst_dtype=k_nope.dtype
                )
                k_nope_before = k_nope if layer_id == layer_ids[0] else None
                k_nope = self._rescale_tensor(k_nope, dst_k_nope)
                k_rope = self._rescale_tensor(k_rope, dst_k_rope)
                if layer_id == layer_ids[0] and k_nope_before is not None:
                    src_norm = torch.norm(k_nope_before.float(), p=2, dim=-1).mean().item()
                    rescaled_norm = torch.norm(k_nope.float(), p=2, dim=-1).mean().item()
                    scale_factor = rescaled_norm / (src_norm + 1e-6)
                    logger.info(
                        "[kv_graft rescale MLA] src_norm=%.4f rescaled_norm=%.4f scale=%.4f",
                        src_norm, rescaled_norm, scale_factor,
                    )
            if (
                not copy_only
                and transform.rope_shift in ("on", "auto")
                and scaled_delta != 0
                and self._rope_layer_selected(layer_id, _rescale_params)
            ):
                k_rope = self._rope_shift_tensor(
                    k_rope,
                    scaled_delta,
                    origin_start=origin_start,
                    compute_mode=rope_compute_mode,
                )
            self.kv_pool.set_mla_kv_buffer(layer_stub, dst_indices, k_nope, k_rope)
            if copy_only and verify_copy:
                copied_k_nope, copied_k_rope = self.kv_pool.get_mla_kv_buffer(layer_stub, dst_indices, dst_dtype=k_nope.dtype)
                nope_diff = (copied_k_nope.float() - orig_k_nope.float()).abs().max().item() if copied_k_nope.numel() > 0 else 0.0
                rope_diff = (copied_k_rope.float() - orig_k_rope.float()).abs().max().item() if copied_k_rope.numel() > 0 else 0.0
                max_copy_nope_diff = max(max_copy_nope_diff, nope_diff)
                max_copy_rope_diff = max(max_copy_rope_diff, rope_diff)
                if nope_diff != 0.0 or rope_diff != 0.0:
                    raise RuntimeError(
                        "materialized MLA copy identity failed: "
                        f"layer={layer_id} max_nope_diff={nope_diff} "
                        f"max_rope_diff={rope_diff}"
                    )
        self._synchronize_materialization(dst_indices.device)
        if copy_only and verify_copy:
            logger.info(
                "[kv_graft copy_check MLA complete] layers=%s max_nope_diff=%s max_rope_diff=%s",
                len(layer_ids),
                max_copy_nope_diff,
                max_copy_rope_diff,
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
