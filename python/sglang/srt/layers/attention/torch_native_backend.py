from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


def build_graft_attention_mask(
    spans,
    seq_len_kv: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    """Build a broadcastable additive mask for foreign KV spans."""
    active = [span for span in (spans or []) if float(span[2]) != 0.0]
    if not active:
        return None
    mask = torch.zeros((1, 1, 1, seq_len_kv), dtype=dtype, device=device)
    for start, end, bias in active:
        start = max(0, min(int(start), seq_len_kv))
        end = max(start, min(int(end), seq_len_kv))
        if start < end:
            mask[..., start:end] += float(bias)
    return mask


def build_sdpa_graft_mask(
    spans,
    query_len: int,
    seq_len_kv: int,
    *,
    query_start: int = 0,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    """Return an SDPA mask and whether native causal masking remains enabled."""
    bias_mask = build_graft_attention_mask(
        spans, seq_len_kv, dtype=dtype, device=device
    )
    if not causal:
        return bias_mask, False
    if query_start == 0 and query_len == seq_len_kv and bias_mask is None:
        return None, True

    query_positions = query_start + torch.arange(query_len, device=device)
    key_positions = torch.arange(seq_len_kv, device=device)
    causal_allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if bias_mask is None:
        return causal_allowed.unsqueeze(0).unsqueeze(0), False

    causal_mask = torch.zeros(
        (query_len, seq_len_kv), dtype=dtype, device=device
    ).masked_fill_(~causal_allowed, float("-inf"))
    return bias_mask + causal_mask.unsqueeze(0).unsqueeze(0), False


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    @staticmethod
    def _load_kv_diagnostic_control():
        """Load an explicitly armed, local-only diagnostic capture request."""
        control_path = os.getenv("SGLANG_KV_DIAGNOSTIC_CONTROL", "")
        root_path = os.getenv("SGLANG_KV_DIAGNOSTIC_ROOT", "")
        if not control_path or not root_path:
            return None
        path = Path(control_path)
        if not path.is_file():
            return None
        try:
            control = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        capture_id = str(control.get("capture_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", capture_id):
            raise ValueError("invalid KV diagnostic capture_id")
        root = Path(root_path).resolve()
        output_dir = (root / capture_id).resolve()
        if root not in output_dir.parents:
            raise ValueError("KV diagnostic output escaped configured root")
        control["output_dir"] = output_dir
        return control

    def _maybe_capture_kv_diagnostics(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        scaling: float | None = None,
    ) -> None:
        """Persist one last-token Q and sampled cached K/V shard per TP rank.

        This is disabled unless both diagnostic environment variables are set
        and the external driver atomically arms a control JSON file. Scientific
        runs are serial, so a capture id corresponds to exactly one request.
        """
        control = self._load_kv_diagnostic_control()
        if control is None:
            return
        selected_layers = control.get("layer_ids")
        if selected_layers is not None and int(layer.layer_id) not in {
            int(value) for value in selected_layers
        }:
            return
        if int(forward_batch.req_pool_indices.numel()) != 1:
            raise RuntimeError("KV diagnostics require exactly one request per batch")
        if int(forward_batch.extend_seq_lens.numel()) != 1:
            raise RuntimeError("KV diagnostics require one extend sequence")

        seq_len = int(forward_batch.seq_lens[0].item())
        body_start = int(control["body_start"])
        body_end = int(control["body_end"])
        if body_start < 0 or body_end <= body_start or body_end > seq_len:
            raise ValueError(
                f"invalid diagnostic body span [{body_start},{body_end}) for seq_len={seq_len}"
            )
        relative_positions = [int(value) for value in control["sample_relative_positions"]]
        if not relative_positions or min(relative_positions) < 0:
            raise ValueError("diagnostic sample positions must be non-negative")
        if max(relative_positions) >= body_end - body_start:
            raise ValueError("diagnostic sample position exceeds body span")

        req_pool_idx = int(forward_batch.req_pool_indices[0].item())
        logical_positions = torch.tensor(
            [body_start + value for value in relative_positions],
            dtype=torch.int64,
            device=forward_batch.req_to_token_pool.req_to_token.device,
        )
        cache_indices = forward_batch.req_to_token_pool.req_to_token[
            req_pool_idx, logical_positions
        ]
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        tp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        output_dir = control["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"layer_{int(layer.layer_id):02d}_rank_{tp_rank:02d}.pt"
        temporary_path = output_dir / (
            f".{output_path.name}.tmp.{os.getpid()}"
        )
        # kv_diagnostic_attention_mass_v1: save aggregates, never the full matrix.
        attention_mass = {}
        configured_spans = control.get("attention_spans") or {}
        if configured_spans:
            all_positions = torch.arange(seq_len, device=cache_indices.device)
            all_indices = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, all_positions
            ]
            all_k = k_cache[all_indices].float()
            q_last = q[-1].float()
            if all_k.shape[1] != q_last.shape[0]:
                if q_last.shape[0] % all_k.shape[1] != 0:
                    raise ValueError("diagnostic GQA head ratio is not integral")
                all_k = all_k.repeat_interleave(
                    q_last.shape[0] // all_k.shape[1], dim=1
                )
            score_scale = float(scaling) if scaling is not None else all_k.shape[-1] ** -0.5
            probs = torch.einsum("hd,thd->ht", q_last, all_k) * score_scale
            probs = probs.softmax(dim=-1)
            for name, span in configured_spans.items():
                start, end = int(span[0]), int(span[1])
                start, end = max(0, start), min(seq_len, end)
                attention_mass[str(name)] = (
                    probs[:, start:end].sum(dim=-1).detach().to(device="cpu")
                )

        payload = {
            "schema_version": 1,
            "capture_id": control["capture_id"],
            "case": str(control.get("case", "")),
            "sample_id": int(control.get("sample_id", -1)),
            "layer_id": int(layer.layer_id),
            "tp_rank": int(tp_rank),
            "seq_len": seq_len,
            "body_start": body_start,
            "body_end": body_end,
            "sample_relative_positions": relative_positions,
            "sample_logical_positions": logical_positions.detach().cpu(),
            "q_last": q[-1].detach().to(dtype=torch.bfloat16, device="cpu"),
            "k_body": k_cache[cache_indices]
            .detach()
            .to(dtype=torch.bfloat16, device="cpu"),
            "v_body": v_cache[cache_indices]
            .detach()
            .to(dtype=torch.bfloat16, device="cpu"),
            "attention_mass": attention_mass,
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        attention_bias_spans=None,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
            attn_mask, use_native_causal = build_sdpa_graft_mask(
                (attention_bias_spans or [])[seq_idx]
                if attention_bias_spans
                else None,
                int(extend_seq_len_q),
                int(seq_len_kv),
                query_start=int(prefill_seq_len_q),
                causal=causal,
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    attn_mask=attn_mask,
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=use_native_causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        attention_bias_spans=None,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
            attn_mask, use_native_causal = build_sdpa_graft_mask(
                (attention_bias_spans or [])[seq_idx]
                if attention_bias_spans
                else None,
                int(seq_len_q),
                int(seq_len_kv),
                query_start=int(seq_len_kv) - int(seq_len_q),
                causal=causal,
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    attn_mask=attn_mask,
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=use_native_causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._maybe_capture_kv_diagnostics(
            q_, layer, forward_batch, scaling=layer.scaling
        )

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
            attention_bias_spans=forward_batch.graft_attention_bias_spans,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
            attention_bias_spans=forward_batch.graft_attention_bias_spans,
        )

        return o

    def support_triton(self):
        return False
