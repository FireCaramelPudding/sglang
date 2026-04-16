import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.managers.io_struct import KVExportSpec
from sglang.srt.managers.kv_graft_materializer import (
    MHAGraftMaterializer,
    MLAGraftMaterializer,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _FakeKVPool:
    def __init__(self):
        self.key_buffers = {
            0: torch.tensor(
                [
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 2.0, 2.0, 2.0]],
                    [[0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
        self.value_buffers = {
            0: torch.tensor(
                [
                    [[3.0, 3.0, 3.0, 3.0]],
                    [[4.0, 4.0, 4.0, 4.0]],
                    [[0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
        self.mla_nope = {
            0: torch.tensor(
                [
                    [[5.0, 5.0, 5.0, 5.0]],
                    [[6.0, 6.0, 6.0, 6.0]],
                    [[0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
        self.mla_rope = {
            0: torch.tensor(
                [
                    [[7.0, 7.0, 7.0, 7.0]],
                    [[8.0, 8.0, 8.0, 8.0]],
                    [[0.0, 0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }

    def get_key_buffer(self, layer_id):
        return self.key_buffers[layer_id]

    def get_value_buffer(self, layer_id):
        return self.value_buffers[layer_id]

    def get_mla_kv_buffer(self, layer_stub, indices, dst_dtype=None):
        layer_id = layer_stub.layer_id
        k_nope = self.mla_nope[layer_id][indices]
        k_rope = self.mla_rope[layer_id][indices]
        if dst_dtype is not None:
            k_nope = k_nope.to(dst_dtype)
            k_rope = k_rope.to(dst_dtype)
        return k_nope, k_rope

    def set_mla_kv_buffer(self, layer_stub, indices, k_nope, k_rope):
        layer_id = layer_stub.layer_id
        self.mla_nope[layer_id][indices] = k_nope
        self.mla_rope[layer_id][indices] = k_rope


class _RecordingMHAMaterializer(MHAGraftMaterializer):
    def __init__(self, kv_pool):
        super().__init__(kv_pool=kv_pool, rope_theta=10000.0)
        self.ops = []

    def _rescale_tensor(self, src, tgt, eps=1e-6):
        self.ops.append("rescale")
        return src + 10.0

    def _rope_shift_tensor(self, k, delta, origin_start=0):
        self.ops.append("rope")
        return k + 100.0


class _RecordingMLAMaterializer(MLAGraftMaterializer):
    def __init__(self, kv_pool):
        super().__init__(kv_pool=kv_pool, rope_theta=10000.0)
        self.ops = []

    def _rescale_tensor(self, src, tgt, eps=1e-6):
        self.ops.append("rescale")
        return src + 10.0

    def _rope_shift_tensor(self, k, delta, origin_start=0):
        self.ops.append("rope")
        return k + 100.0


class _RecordingRegistry:
    def __init__(self):
        self.register_calls = []

    def register(self, **kwargs):
        self.register_calls.append(kwargs)
        return {"handle": "kvh_test"}


class _LookupRegistry:
    def __init__(self, entry):
        self.entry = entry
        self.model_key = "model"
        self.backend = "mha"

    def lookup(self, handle, tree_cache=None):
        return self.entry


class _AllocRecorder:
    def __init__(self):
        self.held = []
        self.next_alloc = torch.tensor([20], dtype=torch.int64)

    def hold(self, indices):
        self.held.append(indices.clone())

    def alloc(self, numel):
        assert numel == self.next_alloc.numel()
        return self.next_alloc.clone()


class _MaterializerRecorder:
    def __init__(self):
        self.calls = []

    def transform_segment(self, **kwargs):
        self.calls.append(kwargs)


class _GuardedReq:
    def __init__(self):
        self.rid = "req_guarded_export"
        self.kv_export_spec = SimpleNamespace(
            token_start=0,
            token_end=None,
            origin_start=7,
            ttl_seconds=300,
            persist=True,
            name="guarded-export",
        )
        self.kv_exports = []
        self.kv_committed_len = 5
        self.req_pool_idx = 0
        self.synthetic_prefix_physical_len = 2
        self.kv_graft_spec = SimpleNamespace(
            segments=[SimpleNamespace(transform=None)]
        )
        self.helper_called_with = None

    @property
    def logical_fill_ids(self):
        raise AssertionError("scheduler should use get_exportable_logical_token_ids")

    def get_exportable_logical_token_ids(self, committed_len=None):
        self.helper_called_with = committed_len
        return [11, 12, 13]


class _PrefillExportReq:
    def __init__(self):
        self.graft_export_after_prefill = True
        self.prompt_token_count = 9


class _ApplyGraftReq:
    def __init__(self):
        self.kv_export_spec = None
        self.graft_export_after_prefill = False
        self.disable_radix_match = False


class _MixedRunningReq:
    def __init__(self):
        self.origin_input_ids = [1, 2]
        self.output_ids = [3]
        self.logical_fill_ids = [90, 1, 2, 3]
        self.prompt_token_count = 3
        self.fill_ids = None
        self.extend_input_len_calls = []

    def set_extend_input_len(self, value):
        self.extend_input_len_calls.append(value)


class _RunningBatchStub:
    def __init__(self, reqs):
        self.reqs = reqs
        self.input_ids = torch.tensor([7], dtype=torch.int64)
        self.out_cache_loc = torch.tensor([70], dtype=torch.int64)

    def batch_size(self):
        return len(self.reqs)


class _MixedBatchStub:
    def __init__(self):
        self.forward_mode = None
        self.enable_overlap = False
        self.input_ids = torch.tensor([5], dtype=torch.int64)
        self.out_cache_loc = torch.tensor([50], dtype=torch.int64)
        self.prefix_lens = []
        self.extend_lens = []
        self.extend_num_tokens = 0
        self.extend_logprob_start_lens = []
        self.is_prefill_only = True
        self.reqs = []

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)


class _PrebuiltReq:
    def __init__(self):
        self.fill_ids = [90, 1, 2, 8, 9]
        self.prefix_indices = torch.tensor([101, 102, 103], dtype=torch.int64)
        self.req_pool_idx = 4
        self.extend_input_len = 2
        self.origin_input_ids = [1, 2]
        self.output_ids = [8, 9]
        self.prompt_token_count = 3
        self.retracted_stain = False
        self.cached_tokens = 0
        self.already_computed = 0
        self.is_retracted = True
        self.extend_logprob_start_len = None
        self.top_logprobs_num = 0
        self.token_ids_logprob = [1, 2]
        self.multimodal_inputs = None


class _PrebuiltBatchStub(ScheduleBatchDisaggregationDecodeMixin):
    def __init__(self, reqs):
        self.reqs = reqs
        self.device = torch.device("cpu")
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [11, 12, 13, 14, 15],
                ],
                dtype=torch.int64,
            )
        )
        self.return_logprob = False
        self.model_config = SimpleNamespace(vocab_size=32000)
        self.enable_overlap = False
        self.spec_algorithm = None
        self.tree_cache = object()


class _DecodePreallocReq:
    def __init__(self):
        self.prompt_token_count = 3
        self.origin_input_ids = [1, 2]
        self.output_ids = [8, 9]
        self.logical_fill_ids = [90, 1, 2, 8, 9]
        self.req_pool_idx = 0
        self.kv_allocated_len = None
        self.kv_committed_len = None
        self.extend_input_len_calls = []

    @property
    def seqlen(self):
        return self.prompt_token_count + len(self.output_ids)

    def set_extend_input_len(self, value):
        self.extend_input_len_calls.append(value)


class _InitNextRoundReq:
    def __init__(self):
        self.is_dllm = lambda: False
        self.logical_fill_ids = [90, 1, 2, 8, 9]
        self.fill_ids = None
        self.return_logprob = False
        self.logprob_start_len = -1
        self.session = None
        self.disable_radix_match = True
        self.synthetic_prefix_indices = torch.tensor([101, 102, 103], dtype=torch.int64)
        self.synthetic_prefix_token_ids = [90, 1, 2]
        self.req_pool_idx = 0
        self.kv_committed_len = 4
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.cache_protected_len = 0
        self.last_node = object()
        self.last_host_node = object()
        self.host_hit_length = 7
        self.mamba_branching_seqlen = 11
        self.is_retracted = False
        self.multimodal_inputs = None
        self.output_ids = [8, 9]
        self.rid = "req_init_round"
        self.extend_input_len_calls = []

    def set_extend_input_len(self, value):
        self.extend_input_len_calls.append(value)


class _FinishedCacheReq:
    def __init__(self):
        self.origin_input_ids = [1, 2]
        self.output_ids = [8, 9]
        self.synthetic_prefix_token_ids = [90]
        self.req_pool_idx = 0
        self.cache_protected_len = 0
        self.extra_key = None
        self.last_node = None
        self.kv_committed_len = 4
        self.kv_committed_freed = False

    @property
    def logical_fill_ids(self):
        return self.synthetic_prefix_token_ids + self.origin_input_ids + self.output_ids

    def get_exportable_logical_token_ids(self, committed_len=None):
        if committed_len is None:
            committed_len = self.kv_committed_len
        return self.logical_fill_ids[:committed_len]

    def pop_committed_kv_cache(self):
        self.kv_committed_freed = True
        return self.kv_committed_len


class TestKVGraftRegressions(unittest.TestCase):
    def test_mha_transform_rescales_before_rope(self):
        kv_pool = _FakeKVPool()
        materializer = _RecordingMHAMaterializer(kv_pool)

        materializer.transform_segment(
            source_indices=torch.tensor([0, 1], dtype=torch.int64),
            dst_indices=torch.tensor([2, 3], dtype=torch.int64),
            transform=SimpleNamespace(
                rope_shift="on",
                rescale_profile="match_stats",
            ),
            origin_start=0,
            current_prefix_len_before_append=4,
            layer_ids=[0],
            reference_indices=torch.tensor([0, 1], dtype=torch.int64),
        )

        self.assertEqual(materializer.ops, ["rescale", "rescale", "rope"])

    def test_mla_transform_rescales_before_rope(self):
        kv_pool = _FakeKVPool()
        materializer = _RecordingMLAMaterializer(kv_pool)

        materializer.transform_segment(
            source_indices=torch.tensor([0, 1], dtype=torch.int64),
            dst_indices=torch.tensor([2, 3], dtype=torch.int64),
            transform=SimpleNamespace(
                rope_shift="on",
                rescale_profile="match_stats",
            ),
            origin_start=0,
            current_prefix_len_before_append=4,
            layer_ids=[0],
            reference_indices=torch.tensor([0, 1], dtype=torch.int64),
        )

        self.assertEqual(materializer.ops, ["rescale", "rescale", "rope"])

    def test_kv_export_uses_committed_helper(self):
        registry = _RecordingRegistry()
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[101, 102, 103, 104, 105]], dtype=torch.int64)
        )
        fake_scheduler = SimpleNamespace(
            kv_handle_registry=registry,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=object(),
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(kv_cache_dtype=torch.bfloat16)
            ),
        )
        req = _GuardedReq()

        Scheduler._maybe_register_kv_export(fake_scheduler, req)

        self.assertEqual(req.helper_called_with, 5)
        self.assertEqual(len(registry.register_calls), 1)
        call = registry.register_calls[0]
        self.assertEqual(call["token_ids"], [11, 12, 13])
        self.assertEqual(call["device_indices"].tolist(), [101, 102, 103])
        self.assertTrue(call["composite"])
        self.assertEqual(req.kv_exports, [{"handle": "kvh_test"}])

    def test_prefill_graft_export_uses_prompt_prefix_len(self):
        recorded = {}

        fake_scheduler = SimpleNamespace()

        def _record_export(req, *, committed_len_override=None):
            recorded["req"] = req
            recorded["committed_len_override"] = committed_len_override

        fake_scheduler._maybe_register_kv_export = _record_export
        req = _PrefillExportReq()

        Scheduler._maybe_register_prefill_graft_export(fake_scheduler, req)

        self.assertIs(recorded["req"], req)
        self.assertEqual(recorded["committed_len_override"], 9)

    def test_prompt_only_kv_export_disables_radix_match(self):
        fake_scheduler = SimpleNamespace()
        fake_scheduler._should_export_after_prefill = lambda export_spec, prompt_len: (
            Scheduler._should_export_after_prefill(
                fake_scheduler, export_spec, prompt_len
            )
        )
        req = _ApplyGraftReq()
        recv_req = SimpleNamespace(
            kv_export=KVExportSpec(token_start=0, token_end=4, origin_start=0),
            input_ids=[1, 2, 3, 4],
            kv_graft=None,
        )

        Scheduler._apply_kv_graft(fake_scheduler, req, recv_req)

        self.assertEqual(req.kv_export_spec, recv_req.kv_export)
        self.assertTrue(req.graft_export_after_prefill)
        self.assertTrue(req.disable_radix_match)

    def test_non_prompt_only_kv_export_keeps_radix_match(self):
        fake_scheduler = SimpleNamespace()
        fake_scheduler._should_export_after_prefill = lambda export_spec, prompt_len: (
            Scheduler._should_export_after_prefill(
                fake_scheduler, export_spec, prompt_len
            )
        )
        req = _ApplyGraftReq()
        recv_req = SimpleNamespace(
            kv_export=KVExportSpec(token_start=0, token_end=None, origin_start=0),
            input_ids=[1, 2, 3, 4],
            kv_graft=None,
        )

        Scheduler._apply_kv_graft(fake_scheduler, req, recv_req)

        self.assertEqual(req.kv_export_spec, recv_req.kv_export)
        self.assertFalse(req.graft_export_after_prefill)
        self.assertFalse(req.disable_radix_match)

    def test_graft_transform_offsets_origin_start_by_token_start(self):
        entry = SimpleNamespace(
            meta=SimpleNamespace(model_key="model", backend="mha", origin_start=697),
            device_indices=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            token_ids=[100, 101, 102, 103],
        )
        allocator = _AllocRecorder()
        materializer = _MaterializerRecorder()
        fake_scheduler = SimpleNamespace(
            kv_handle_registry=_LookupRegistry(entry),
            tree_cache=None,
            token_to_kv_pool_allocator=allocator,
            kv_graft_materializer=materializer,
            _graft_layer_ids=lambda: [0],
        )
        segment = SimpleNamespace(
            handle="kvh_test",
            token_start=3,
            token_end=None,
            origin_start=697,
            transform=SimpleNamespace(
                rope_shift="on",
                rescale_profile=None,
                rescale_params=None,
            ),
        )

        new_indices, token_ids, is_owned = Scheduler._resolve_graft_segment(
            fake_scheduler,
            req=None,
            segment=segment,
            current_prefix_indices=[],
            current_prefix_tokens=[1, 2, 3, 4],
        )

        self.assertTrue(is_owned)
        self.assertEqual(token_ids, [103])
        self.assertEqual(new_indices.tolist(), [20])
        self.assertEqual(len(materializer.calls), 1)
        self.assertEqual(materializer.calls[0]["origin_start"], 700)
        self.assertEqual(materializer.calls[0]["source_indices"].tolist(), [13])

    def test_graft_transform_uses_segment_origin_for_pre_sliced_handle(self):
        entry = SimpleNamespace(
            meta=SimpleNamespace(model_key="model", backend="mha", origin_start=0),
            device_indices=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            token_ids=[100, 101, 102, 103],
        )
        allocator = _AllocRecorder()
        materializer = _MaterializerRecorder()
        fake_scheduler = SimpleNamespace(
            kv_handle_registry=_LookupRegistry(entry),
            tree_cache=None,
            token_to_kv_pool_allocator=allocator,
            kv_graft_materializer=materializer,
            _graft_layer_ids=lambda: [0],
        )
        segment = SimpleNamespace(
            handle="kvh_test",
            token_start=3,
            token_end=None,
            origin_start=700,
            transform=SimpleNamespace(
                rope_shift="on",
                rescale_profile=None,
                rescale_params=None,
            ),
        )

        new_indices, token_ids, is_owned = Scheduler._resolve_graft_segment(
            fake_scheduler,
            req=None,
            segment=segment,
            current_prefix_indices=[],
            current_prefix_tokens=[1, 2, 3, 4],
        )

        self.assertTrue(is_owned)
        self.assertEqual(token_ids, [103])
        self.assertEqual(new_indices.tolist(), [20])
        self.assertEqual(len(materializer.calls), 1)
        self.assertEqual(materializer.calls[0]["origin_start"], 700)
        self.assertEqual(materializer.calls[0]["source_indices"].tolist(), [13])

    def test_mix_with_running_uses_logical_fill_ids_and_prefix_len(self):
        req = _MixedRunningReq()
        mixed_batch = _MixedBatchStub()
        running_batch = _RunningBatchStub([req])

        ScheduleBatch.mix_with_running(mixed_batch, running_batch)

        self.assertEqual(mixed_batch.forward_mode, ForwardMode.MIXED)
        self.assertEqual(req.fill_ids, [90, 1, 2, 3])
        self.assertEqual(req.extend_input_len_calls, [1])
        self.assertEqual(mixed_batch.prefix_lens, [3])
        self.assertEqual(mixed_batch.extend_lens, [1])
        self.assertEqual(mixed_batch.extend_num_tokens, 1)
        self.assertEqual(mixed_batch.extend_logprob_start_lens, [0])
        self.assertFalse(mixed_batch.is_prefill_only)

    def test_prepare_ngram_embedding_uses_logical_fill_ids(self):
        recorded = {}

        def _record_update_token_table(**kwargs):
            recorded["tokens"] = kwargs["tokens"].cpu().tolist()
            recorded["column_starts"] = kwargs["column_starts"].cpu().tolist()
            recorded["req_lens"] = kwargs["req_lens"].cpu().tolist()

        req = SimpleNamespace(
            prefix_indices=torch.tensor([11], dtype=torch.int64),
            extend_input_len=2,
            origin_input_ids=[1, 2],
            output_ids=[3],
            logical_fill_ids=[90, 1, 2, 3],
        )
        batch = SimpleNamespace(
            reqs=[req],
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            ne_token_table=None,
        )
        fake_scheduler = SimpleNamespace(
            use_ngram_embedding=True,
            token_table=torch.zeros((1, 8), dtype=torch.int64),
            ngram_embedding_n=2,
        )

        with patch(
            "sglang.srt.managers.scheduler.update_token_table",
            new=_record_update_token_table,
        ):
            result = Scheduler._maybe_prepare_ngram_embedding(fake_scheduler, batch)

        self.assertIs(result, batch)
        self.assertIs(batch.ne_token_table, fake_scheduler.token_table)
        self.assertEqual(recorded["tokens"], [90, 1, 2])
        self.assertEqual(recorded["column_starts"], [0])
        self.assertEqual(recorded["req_lens"], [3])

    def test_hisparse_decode_batch_seq_lens_include_synthetic_prefix(self):
        req = SimpleNamespace(
            req_pool_idx=4,
            prompt_token_count=5,
            origin_input_ids=[1, 2],
            output_ids=[8, 9],
            top_logprobs_num=0,
        )
        fake_scheduler = SimpleNamespace(
            device=torch.device("cpu"),
            req_to_token_pool=SimpleNamespace(device=torch.device("cpu")),
            token_to_kv_pool_allocator=object(),
            tree_cache=object(),
            model_config=SimpleNamespace(vocab_size=32000),
            enable_overlap=False,
            spec_algorithm=None,
        )

        with patch.object(
            ScheduleBatch,
            "init_new",
            return_value=SimpleNamespace(return_logprob=False),
        ) as init_new_mock, patch(
            "sglang.srt.managers.scheduler.SamplingBatchInfo.from_schedule_batch",
            return_value="sampling-info",
        ):
            batch = Scheduler._build_hisparse_decode_batch(fake_scheduler, [req])

        init_new_mock.assert_called_once()
        self.assertEqual(batch.seq_lens.tolist(), [6])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [6])
        self.assertEqual(batch.orig_seq_lens.tolist(), [6])
        self.assertEqual(batch.seq_lens_sum, 6)
        self.assertEqual(batch.output_ids.tolist(), [9])
        self.assertEqual(batch.sampling_info, "sampling-info")

    def test_prebuilt_batch_seq_lens_include_synthetic_prefix(self):
        req = _PrebuiltReq()
        batch = _PrebuiltBatchStub([req])

        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin.SamplingBatchInfo.from_schedule_batch",
            return_value="sampling-info",
        ):
            batch.prepare_for_prebuilt()

        self.assertEqual(batch.input_ids.tolist(), [8, 9])
        self.assertEqual(batch.seq_lens.tolist(), [4])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [4])
        self.assertEqual(batch.orig_seq_lens.tolist(), [4])
        self.assertEqual(batch.seq_lens_sum, 4)
        self.assertEqual(batch.prefix_lens, [3])
        self.assertEqual(batch.extend_lens, [2])
        self.assertEqual(req.cached_tokens, 3)
        self.assertEqual(req.already_computed, 4)
        self.assertFalse(req.is_retracted)
        self.assertEqual(req.extend_logprob_start_len, 0)
        self.assertEqual(batch.sampling_info, "sampling-info")

    def test_decode_prealloc_uses_logical_fill_ids_and_prompt_len(self):
        req = _DecodePreallocReq()
        kv_pool = SimpleNamespace(
            page_size=1,
            alloc=lambda fill_len: torch.arange(200, 200 + fill_len, dtype=torch.int64),
        )
        prealloc = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                alloc=lambda reqs: [0],
                write=lambda indices, values: None,
            ),
            token_to_kv_pool_allocator=kv_pool,
        )

        kv_loc = DecodePreallocQueue._pre_alloc(prealloc, req)

        self.assertEqual(kv_loc.tolist(), [200, 201, 202, 203])
        self.assertEqual(req.kv_allocated_len, 4)
        self.assertEqual(req.kv_committed_len, 4)
        self.assertEqual(req.fill_ids, [90, 1, 2, 8, 9])
        self.assertEqual(req.extend_input_len_calls, [5])

    def test_init_next_round_input_uses_committed_graft_prefix_indices(self):
        req = _InitNextRoundReq()
        tree_cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor([[101, 102, 103, 204, 205]], dtype=torch.int64)
            )
        )

        req.init_next_round_input = Req.init_next_round_input.__get__(req, type(req))
        req.init_next_round_input(tree_cache)

        self.assertEqual(req.fill_ids, [90, 1, 2, 8, 9])
        self.assertEqual(req.prefix_indices.tolist(), [101, 102, 103, 204])
        self.assertEqual(req.cache_protected_len, 3)
        self.assertEqual(req.host_hit_length, 0)
        self.assertIsNone(req.last_node)
        self.assertIsNone(req.last_host_node)
        self.assertIsNone(req.mamba_branching_seqlen)
        self.assertEqual(req.extend_input_len_calls, [1])

    def test_schedule_policy_uses_committed_graft_prefix_indices(self):
        req = SimpleNamespace(
            disable_radix_match=True,
            synthetic_prefix_indices=torch.tensor([101, 102, 103], dtype=torch.int64),
            req_pool_idx=0,
            kv_committed_len=4,
            logical_fill_ids=[90, 1, 2, 8, 9],
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            cache_protected_len=0,
            last_node=object(),
            last_host_node=object(),
            host_hit_length=9,
        )
        policy = SimpleNamespace(
            waiting_queue_radix_tree=SimpleNamespace(reset=lambda: None),
            tree_cache=SimpleNamespace(
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.tensor([[101, 102, 103, 204, 205]], dtype=torch.int64)
                )
            ),
        )

        temporary = type("_TmpRadix", (), {"reset": lambda self: None})()
        policy.waiting_queue_radix_tree = temporary
        SchedulePolicy._compute_prefix_matches(policy, [req], CacheAwarePolicy.LPM)

        self.assertEqual(req.prefix_indices.tolist(), [101, 102, 103, 204])
        self.assertEqual(req.cache_protected_len, 3)
        self.assertIsNone(req.last_node)
        self.assertIsNone(req.last_host_node)
        self.assertEqual(req.host_hit_length, 0)

    def test_cache_finished_req_uses_logical_token_ids(self):
        req = _FinishedCacheReq()
        freed = []
        radix = SimpleNamespace(
            disable_finished_insert=False,
            disable=False,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor([[301, 302, 303, 304, 305]], dtype=torch.int64)
            ),
            token_to_kv_pool_allocator=SimpleNamespace(free=lambda indices: freed.append(indices.clone())),
            is_eagle=False,
            page_size=1,
            maybe_bigram_convert=lambda key, value: (key, value),
            dec_lock_ref=lambda node: None,
        )
        recorded = {}

        def _insert(params):
            recorded["keys"] = list(params.key.token_ids)
            recorded["values"] = params.value.tolist()
            return SimpleNamespace(prefix_len=0)

        radix.insert = _insert

        RadixCache.cache_finished_req(radix, req)

        self.assertEqual(recorded["keys"], [90, 1, 2, 8])
        self.assertEqual(recorded["values"], [301, 302, 303, 304])
        self.assertTrue(req.kv_committed_freed)
        self.assertEqual(len(freed), 2)
        self.assertEqual(freed[0].tolist(), [])
        self.assertEqual(freed[1].tolist(), [])

    def test_decode_allocatable_tokens_counts_logical_lengths_for_retracted(self):
        req = SimpleNamespace(seqlen=5)
        prealloc = SimpleNamespace(
            scheduler=SimpleNamespace(
                running_batch=SimpleNamespace(reqs=[]),
                waiting_queue=[],
                last_batch=None,
            ),
            transfer_queue=SimpleNamespace(queue=[]),
            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 20),
            num_reserved_decode_tokens=2,
            retracted_queue=[req],
        )

        allocatable = DecodePreallocQueue._allocatable_tokens(prealloc, count_retracted=True)

        self.assertEqual(allocatable, 13)

    def test_decode_allocatable_tokens_counts_logical_lengths_for_running(self):
        running_req = SimpleNamespace(
            prompt_token_count=3,
            sampling_params=SimpleNamespace(max_new_tokens=7),
        )
        prealloc = SimpleNamespace(
            scheduler=SimpleNamespace(
                running_batch=SimpleNamespace(reqs=[running_req]),
                waiting_queue=[],
                last_batch=None,
            ),
            transfer_queue=SimpleNamespace(queue=[]),
            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 20),
            num_reserved_decode_tokens=2,
            retracted_queue=[],
        )

        allocatable = DecodePreallocQueue._allocatable_tokens(
            prealloc, retractable_tokens=5, count_retracted=False
        )

        self.assertEqual(allocatable, 15)

    def test_pop_preallocated_uses_prompt_len_for_capacity_check(self):
        req = SimpleNamespace(
            rid="req1",
            finished_reason=None,
            waiting_for_input=True,
            req_pool_idx=0,
            prompt_token_count=3,
            origin_input_ids=[1, 2],
            sampling_params=SimpleNamespace(max_new_tokens=9),
            time_stats=SimpleNamespace(set_decode_transfer_queue_entry_time=lambda: None),
        )
        decode_req = SimpleNamespace(req=req, waiting_for_input=True)
        calls = []
        prealloc = SimpleNamespace(
            queue=[decode_req],
            scheduler=SimpleNamespace(
                running_batch=SimpleNamespace(reqs=[]),
                stream_output=lambda reqs, return_logprob: None,
            ),
            req_to_token_pool=SimpleNamespace(
                available_size=lambda: 1,
                req_to_token=torch.tensor([[31, 32, 33]], dtype=torch.int64),
            ),
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(
                available_size=lambda: 1,
                alloc=lambda: 5,
            ),
            _resolve_pending_reqs=lambda: None,
            _update_handshake_waiters=lambda rids_to_check=None: None,
            _allocatable_tokens=lambda retractable_tokens, count_retracted: 5,
            _pre_alloc=lambda req: calls.append(req),
            num_reserved_decode_tokens=2,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            token_to_kv_pool=object(),
            transfer_queue=SimpleNamespace(queue=[]),
        )
        decode_req.kv_receiver = SimpleNamespace(
            send_metadata=lambda page_indices, metadata_buffer_index, state_indices: None
        )

        preallocated, failed = DecodePreallocQueue.pop_preallocated(prealloc)

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        self.assertEqual(calls, [])
        self.assertEqual(len(prealloc.queue), 1)

    def test_pop_preallocated_uses_logical_running_retractable_tokens(self):
        running_req = SimpleNamespace(seqlen=5)
        req = SimpleNamespace(
            rid="req1",
            finished_reason=None,
            waiting_for_input=True,
            req_pool_idx=0,
            prompt_token_count=3,
            origin_input_ids=[1, 2],
            sampling_params=SimpleNamespace(max_new_tokens=9),
            time_stats=SimpleNamespace(set_decode_transfer_queue_entry_time=lambda: None),
        )
        decode_req = SimpleNamespace(req=req, waiting_for_input=True)
        recorded = {}
        prealloc = SimpleNamespace(
            queue=[decode_req],
            scheduler=SimpleNamespace(
                running_batch=SimpleNamespace(reqs=[running_req]),
                stream_output=lambda reqs, return_logprob: None,
            ),
            req_to_token_pool=SimpleNamespace(available_size=lambda: 0),
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(available_size=lambda: 1),
            _resolve_pending_reqs=lambda: None,
            _update_handshake_waiters=lambda rids_to_check=None: None,
            _allocatable_tokens=lambda retractable_tokens, count_retracted: recorded.update(
                {
                    "retractable_tokens": retractable_tokens,
                    "count_retracted": count_retracted,
                }
            )
            or 0,
            num_reserved_decode_tokens=2,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            token_to_kv_pool=object(),
            transfer_queue=SimpleNamespace(queue=[]),
        )

        preallocated, failed = DecodePreallocQueue.pop_preallocated(prealloc)

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        self.assertEqual(recorded, {"retractable_tokens": 5, "count_retracted": True})


if __name__ == "__main__":
    unittest.main()
