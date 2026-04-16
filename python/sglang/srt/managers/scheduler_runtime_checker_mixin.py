from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.srt.observability.metrics_collector import QueueCount
from sglang.srt.utils.common import ceil_align, raise_error_or_warn
from sglang.srt.utils.request_logger import disable_request_logging
from sglang.srt.utils.watchdog import WatchdogRaw

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerRuntimeCheckerMixin:
    def _external_held_tokens(self: Scheduler) -> int:
        registry = getattr(self, "kv_handle_registry", None)
        if registry is not None:
            return registry.external_uncached_size(self.tree_cache)

        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        if allocator is None:
            return 0
        getter = getattr(allocator, "external_held_size", None)
        if getter is None:
            return 0
        return getter()

    def _external_cached_held_tokens(self: Scheduler) -> int:
        registry = getattr(self, "kv_handle_registry", None)
        if registry is None:
            return 0
        getter = getattr(registry, "external_cached_size", None)
        if getter is None:
            return 0
        return getter(self.tree_cache)

    def _session_held_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_tokens()
        return 0

    def _session_held_full_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_full_tokens()
        return 0

    def _session_held_swa_tokens(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_swa_tokens()
        return 0

    def _session_held_req_count(self: Scheduler) -> int:
        if isinstance(self.tree_cache, SessionAwareCache):
            return self.tree_cache.session_held_req_count()
        return 0

    def _get_token_info(self: Scheduler):
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.tree_cache.evictable_size()
        num_used = self.max_total_num_tokens - (available_size + evictable_size)
        token_usage = num_used / self.max_total_num_tokens
        return num_used, token_usage, available_size, evictable_size

    def _get_mamba_token_info(self: Scheduler):
        is_mamba_radix_cache = (
            self.tree_cache.supports_mamba() and self.tree_cache.is_tree_cache()
        )
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = (
            self.tree_cache.full_evictable_size() if is_mamba_radix_cache else 0
        )
        mamba_available_size = self.req_to_token_pool.mamba_pool.available_size()
        mamba_evictable_size = (
            self.tree_cache.mamba_evictable_size() if is_mamba_radix_cache else 0
        )
        full_num_used = self.token_to_kv_pool_allocator.size - (
            full_available_size + full_evictable_size
        )
        mamba_num_used = self.req_to_token_pool.mamba_pool.size - (
            mamba_available_size + mamba_evictable_size
        )
        full_token_usage = full_num_used / self.token_to_kv_pool_allocator.size
        mamba_usage = mamba_num_used / self.req_to_token_pool.mamba_pool.size
        return (
            full_num_used,
            mamba_num_used,
            full_token_usage,
            mamba_usage,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        )

    def _get_swa_token_info(self: Scheduler):
        full_available_size = self.token_to_kv_pool_allocator.full_available_size()
        full_evictable_size = self.tree_cache.full_evictable_size()
        swa_available_size = self.token_to_kv_pool_allocator.swa_available_size()
        swa_evictable_size = self.tree_cache.swa_evictable_size()
        full_num_used = self.full_tokens_per_layer - (
            full_available_size + full_evictable_size
        )
        swa_num_used = self.swa_tokens_per_layer - (
            swa_available_size + swa_evictable_size
        )
        full_token_usage = full_num_used / self.full_tokens_per_layer
        swa_token_usage = swa_num_used / self.swa_tokens_per_layer
        return (
            full_num_used,
            swa_num_used,
            full_token_usage,
            swa_token_usage,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        )

    def _check_hybrid_memory(self: Scheduler):
        (
            full_num_used,
            swa_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            swa_available_size,
            swa_evictable_size,
        ) = self._get_swa_token_info()
        session_held_full = self._session_held_full_tokens()
        session_held_swa = self._session_held_swa_tokens()

        # Streaming sessions hold tree locks during idle, so tree-protected
        # tokens must be accounted for alongside session-held tokens.
        full_protected = self.tree_cache.full_protected_size()
        swa_protected = self.tree_cache.swa_protected_size()
        full_leaked = full_num_used - full_protected - session_held_full
        swa_leaked = swa_num_used - swa_protected - session_held_swa
        memory_leak = full_leaked != 0 or swa_leaked != 0
        token_msg = (
            f"{full_leaked=}, {swa_leaked=}\n"
            f"{self.full_tokens_per_layer=}, {full_available_size=}, {full_evictable_size=}, {full_protected=}, {session_held_full=}\n"
            f"{self.swa_tokens_per_layer=}, {swa_available_size=}, {swa_evictable_size=}, {swa_protected=}, {session_held_swa=}\n"
        )
        return memory_leak, token_msg

    def _check_mamba_memory(self: Scheduler):
        (
            full_num_used,
            mamba_num_used,
            _,
            _,
            full_available_size,
            full_evictable_size,
            mamba_available_size,
            mamba_evictable_size,
        ) = self._get_mamba_token_info()
        session_held = self._session_held_tokens()
        memory_leak = (
            full_num_used != self.tree_cache.full_protected_size() + session_held
            or mamba_num_used != self.tree_cache.mamba_protected_size()
        )
        if memory_leak:
            free_full_pages = set(
                self.token_to_kv_pool_allocator.free_pages.tolist()
                + self.token_to_kv_pool_allocator.release_pages.tolist()
            )
            cached_full_pages = set(self.tree_cache.all_values_flatten().tolist())
            expected_full_pages = set(
                range(1, self.token_to_kv_pool_allocator.size + 1)
            )
            leaked_full_pages = (
                expected_full_pages - free_full_pages - cached_full_pages
            )
            free_mamba_pages = set(
                self.req_to_token_pool.mamba_pool.free_slots.tolist()
            )
            cached_mamba_pages = set(
                self.tree_cache.all_mamba_values_flatten().tolist()
            )
            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))
            leaked_mamba_pages = (
                expected_mamba_pages - free_mamba_pages - cached_mamba_pages
            )
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}, leaked_full_pages={leaked_full_pages if len(leaked_full_pages) > 0 else None}, leaked_mamba_pages={leaked_mamba_pages if len(leaked_mamba_pages) > 0 else None}\n"
            )
        else:
            token_msg = (
                f"{full_available_size=}, {full_evictable_size=}, {self.token_to_kv_pool_allocator.size=}, {self.tree_cache.full_protected_size()=}\n"
                f"{mamba_available_size=}, {mamba_evictable_size=}, {self.req_to_token_pool.mamba_pool.size=}, {self.tree_cache.mamba_protected_size()=}\n"
            )
        return memory_leak, token_msg

    def _check_radix_cache_memory(self: Scheduler):
        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()
        session_held = self._session_held_tokens()
        external_uncached_held = self._external_held_tokens()
        external_cached_held = self._external_cached_held_tokens()
        effective_evictable_size = evictable_size
        expected_free_or_evictable = (
            self.max_total_num_tokens
            - protected_size
            - session_held
            - external_uncached_held
        )
        if protected_size == 0 and session_held == 0 and external_cached_held > 0:
            effective_evictable_size = max(0, evictable_size - external_cached_held)
            expected_free_or_evictable -= external_cached_held
        memory_leak = (available_size + effective_evictable_size) != expected_free_or_evictable
        token_msg = (
            f"{self.max_total_num_tokens=}, {available_size=}, {evictable_size=}, "
            f"{protected_size=}, {session_held=}, "
            f"{external_uncached_held=}, {external_cached_held=}, "
            f"{effective_evictable_size=}\n"
        )
        if memory_leak:
            allocator = getattr(self, "token_to_kv_pool_allocator", None)
            if allocator is not None:
                deduped = self._dedupe_allocator_free_items(allocator)
                if deduped > 0:
                    available_size = self.token_to_kv_pool_allocator.available_size()
                    memory_leak = (
                        available_size + effective_evictable_size
                    ) != expected_free_or_evictable
                    token_msg += (
                        f"auto_deduped_free_items={deduped}, {available_size=}\n"
                    )
                if memory_leak and (
                    available_size + effective_evictable_size
                ) > expected_free_or_evictable:
                    purged = self._purge_allocator_free_conflicts(allocator)
                    if purged > 0:
                        available_size = self.token_to_kv_pool_allocator.available_size()
                        memory_leak = (
                            available_size + effective_evictable_size
                        ) != expected_free_or_evictable
                        token_msg += (
                            f"auto_purged_conflicting_free_items={purged}, {available_size=}\n"
                        )
                hold_counts = getattr(allocator, "external_hold_counts", None)
                pending_counts = getattr(allocator, "pending_free_counts", None)
                if hold_counts is not None and pending_counts is not None:
                    held_items = int((hold_counts > 0).sum().item())
                    pending_items = int((pending_counts > 0).sum().item())
                    ready_pending_items = int(
                        ((pending_counts > 0) & (hold_counts == 0)).sum().item()
                    )
                    token_msg += (
                        f"allocator_debug: held_items={held_items}, "
                        f"pending_items={pending_items}, "
                        f"ready_pending_items={ready_pending_items}\n"
                    )
                leaked_items_full = self._collect_radix_leaked_items_for_debug(
                    allocator, sample_limit=None
                )
                if leaked_items_full:
                    reclaimed = self._reclaim_radix_orphan_items(
                        allocator, leaked_items_full
                    )
                    if reclaimed > 0:
                        # Re-evaluate after reclaiming orphan items.
                        available_size = self.token_to_kv_pool_allocator.available_size()
                        external_uncached_held = self._external_held_tokens()
                        external_cached_held = self._external_cached_held_tokens()
                        effective_evictable_size = evictable_size
                        expected_free_or_evictable = (
                            self.max_total_num_tokens
                            - protected_size
                            - session_held
                            - external_uncached_held
                        )
                        if (
                            protected_size == 0
                            and session_held == 0
                            and external_cached_held > 0
                        ):
                            effective_evictable_size = max(
                                0, evictable_size - external_cached_held
                            )
                            expected_free_or_evictable -= external_cached_held
                        memory_leak = (
                            available_size + effective_evictable_size
                        ) != expected_free_or_evictable
                        token_msg += (
                            f"auto_reclaimed_orphan_items={reclaimed}, "
                            f"{available_size=}, {external_uncached_held=}, "
                            f"{external_cached_held=}, {effective_evictable_size=}\n"
                        )
                        if memory_leak and (
                            available_size + effective_evictable_size
                        ) > expected_free_or_evictable:
                            purged = self._purge_allocator_free_conflicts(allocator)
                            if purged > 0:
                                available_size = (
                                    self.token_to_kv_pool_allocator.available_size()
                                )
                                memory_leak = (
                                    available_size + effective_evictable_size
                                ) != expected_free_or_evictable
                                token_msg += (
                                    "auto_purged_conflicting_free_items_after_reclaim="
                                    f"{purged}, {available_size=}\n"
                                )
                leaked_items = self._collect_radix_leaked_items_for_debug(
                    allocator, sample_limit=64
                )
                if leaked_items is not None:
                    token_msg += f"leaked_items_sample={leaked_items}\n"
        return memory_leak, token_msg

    def _collect_radix_leaked_items_for_debug(
        self: Scheduler, allocator, sample_limit: int | None = 64
    ) -> list[int] | None:
        try:
            hold_counts = getattr(allocator, "external_hold_counts", None)
            if hold_counts is None:
                return None
            num_items = int(hold_counts.numel())
            if num_items <= 1:
                return None

            expected = set(range(1, num_items))

            free_items = set()
            for attr in ("free_pages", "release_pages"):
                pages = getattr(allocator, attr, None)
                if pages is None:
                    continue
                if len(pages) == 0:
                    continue
                free_items.update(int(x) for x in pages.tolist())

            cached_items = set()
            flatten_fn = getattr(self.tree_cache, "all_values_flatten", None)
            if callable(flatten_fn):
                try:
                    cached = flatten_fn()
                    if cached is not None and cached.numel() > 0:
                        cached = cached.to(device=allocator.device, dtype=torch.int64)
                        if allocator.page_size > 1:
                            cached = cached // allocator.page_size
                        cached_items.update(int(x) for x in torch.unique(cached).tolist())
                except Exception:
                    pass

            held_items = set(
                int(x)
                for x in torch.nonzero(hold_counts > 0, as_tuple=False).flatten().tolist()
                if int(x) > 0
            )

            leaked = sorted(expected - free_items - cached_items - held_items)
            if not leaked:
                return []
            if sample_limit is None:
                return leaked
            return leaked[:sample_limit]
        except Exception:
            return None

    def _purge_allocator_free_conflicts(self: Scheduler, allocator) -> int:
        """Remove invalid/conflicting items from allocator free lists.

        When memory accounting is higher than expected, it usually means some items are
        counted as "free" while still being referenced by the cache or held externally.
        This helper removes such items from the allocator free pages to avoid double
        counting (and potential double-allocation).
        """
        try:
            free_pages = getattr(allocator, "free_pages", None)
            release_pages = getattr(allocator, "release_pages", None)
            if free_pages is None or release_pages is None:
                return 0

            hold_counts = getattr(allocator, "external_hold_counts", None)
            if hold_counts is None:
                return 0

            # Normalize to allocator device/dtype.
            free_pages = free_pages.to(device=allocator.device, dtype=torch.int64)
            release_pages = release_pages.to(device=allocator.device, dtype=torch.int64)

            before = len(free_pages) + len(release_pages)

            # Valid allocator items are [1, num_items - 1]. (0 is padded dummy slot.)
            num_items = int(hold_counts.numel())
            if num_items <= 1:
                return 0
            valid_min = 1
            valid_max = num_items - 1
            if len(free_pages) > 0:
                free_pages = free_pages[
                    (free_pages >= valid_min) & (free_pages <= valid_max)
                ]
            if len(release_pages) > 0:
                release_pages = release_pages[
                    (release_pages >= valid_min) & (release_pages <= valid_max)
                ]

            held_items = torch.nonzero(hold_counts > 0, as_tuple=False).flatten().to(
                device=allocator.device, dtype=torch.int64
            )
            if held_items.numel() > 0:
                held_items = held_items[held_items > 0]

            cached_items = torch.empty((0,), dtype=torch.int64, device=allocator.device)
            flatten_fn = getattr(self.tree_cache, "all_values_flatten", None)
            if callable(flatten_fn):
                try:
                    cached = flatten_fn()
                    if cached is not None and cached.numel() > 0:
                        cached = cached.to(device=allocator.device, dtype=torch.int64)
                        if allocator.page_size > 1:
                            cached = cached // allocator.page_size
                        cached_items = torch.unique(cached)
                        if cached_items.numel() > 0:
                            cached_items = cached_items[cached_items > 0]
                except Exception:
                    cached_items = torch.empty(
                        (0,), dtype=torch.int64, device=allocator.device
                    )

            conflict_parts = []
            if held_items.numel() > 0:
                conflict_parts.append(held_items)
            if cached_items.numel() > 0:
                conflict_parts.append(cached_items)
            if conflict_parts:
                conflicts = torch.unique(torch.cat(conflict_parts))
                if len(free_pages) > 0:
                    free_pages = free_pages[~torch.isin(free_pages, conflicts)]
                if len(release_pages) > 0:
                    release_pages = release_pages[~torch.isin(release_pages, conflicts)]

            allocator.free_pages = free_pages
            allocator.release_pages = release_pages

            after = len(free_pages) + len(release_pages)
            purged = before - after
            if purged > 0:
                logger.warning(
                    "Purged allocator free items conflicting with cache/holds during idle self-check: %d",
                    purged,
                )
            return purged
        except Exception:
            return 0

    def _reclaim_radix_orphan_items(
        self: Scheduler, allocator, leaked_items: list[int]
    ) -> int:
        if not leaked_items:
            return 0
        append_fn = getattr(allocator, "_append_freed_items", None)
        if append_fn is None:
            return 0
        try:
            leaked_tensor = torch.tensor(
                leaked_items, dtype=torch.int64, device=allocator.device
            )
            append_fn(leaked_tensor)
            logger.warning(
                "Recovered orphan KV allocator items during idle self-check: %d",
                len(leaked_items),
            )
            return len(leaked_items)
        except Exception:
            return 0

    def _dedupe_allocator_free_items(self: Scheduler, allocator) -> int:
        try:
            free_pages = getattr(allocator, "free_pages", None)
            release_pages = getattr(allocator, "release_pages", None)
            if free_pages is None or release_pages is None:
                return 0

            before = len(free_pages) + len(release_pages)

            if len(free_pages) > 0:
                free_pages = torch.unique(free_pages.to(dtype=torch.int64))
            else:
                free_pages = free_pages.to(dtype=torch.int64)

            if len(release_pages) > 0:
                release_pages = torch.unique(release_pages.to(dtype=torch.int64))
                if len(free_pages) > 0:
                    release_pages = release_pages[
                        ~torch.isin(release_pages, free_pages)
                    ]
            else:
                release_pages = release_pages.to(dtype=torch.int64)

            allocator.free_pages = free_pages
            allocator.release_pages = release_pages

            after = len(free_pages) + len(release_pages)
            deduped = before - after
            if deduped > 0:
                logger.warning(
                    "Deduplicated allocator free items during idle self-check: %d",
                    deduped,
                )
            return deduped
        except Exception:
            return 0

    def _get_batch_uncached_size(self: Scheduler, batch: ScheduleBatch) -> int:
        ret = 0
        for req in batch.reqs:
            assert req.kv_committed_freed == req.kv_overallocated_freed
            uncached_len = 0
            if not req.kv_committed_freed:
                allocated_len = req.kv_allocated_len
                if self.page_size > 1:
                    allocated_len = ceil_align(allocated_len, self.page_size)
                    assert req.cache_protected_len % self.page_size == 0
                uncached_len = allocated_len - req.cache_protected_len

            ret += uncached_len

        return ret

    def self_check_during_busy(self: Scheduler):
        current_batch: ScheduleBatch = self.last_batch

        if current_batch is None:
            return

        spec_topk = self.server_args.speculative_eagle_topk or 1
        if spec_topk > 1:
            warnings.warn(
                "Runtime memory check (busy) is not supported when speculation topk > 1."
            )
            return

        _, _, available_size, evictable_size = self._get_token_info()
        protected_size = self.tree_cache.protected_size()

        uncached_size = self._get_batch_uncached_size(current_batch)

        if (
            current_batch.forward_mode.is_extend()
            and self.running_batch is not None
            and not self.running_batch.is_empty()
        ):
            uncached_size += self._get_batch_uncached_size(self.running_batch)

        if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get() > 1:
            log_msg = f"[Mem Check (BUSY)] {available_size=}, {evictable_size=}, {protected_size=}, {uncached_size=}"
            logger.info(log_msg)

        session_held = self._session_held_tokens()
        external_uncached_held = self._external_held_tokens()
        total_tokens = (
            available_size
            + evictable_size
            + protected_size
            + uncached_size
            + session_held
            + external_uncached_held
        )
        assert (
            total_tokens == self.max_total_num_tokens
        ), f"Mem Leak Detected! {total_tokens=} vs {self.max_total_num_tokens=}"

    def _check_req_pool(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req_total_size = (
                self.req_to_token_pool.size + self.req_to_token_pool.pre_alloc_size
            )
        else:
            req_total_size = self.req_to_token_pool.size

        session_req_count = self._session_held_req_count()
        if len(self.req_to_token_pool.free_slots) + session_req_count != req_total_size:
            msg = (
                "req_to_token_pool memory leak detected!"
                f"available_size={len(self.req_to_token_pool.free_slots)}, "
                f"session_held={session_req_count}, "
                f"total_size={self.req_to_token_pool.size}\n"
            )
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_req_pool_leak_warnings",
                msg,
            )

    def check_memory(self: Scheduler):
        if self.is_hybrid_swa:
            memory_leak, token_msg = self._check_hybrid_memory()
        elif self.is_hybrid_ssm and self.tree_cache.supports_mamba():
            memory_leak, token_msg = self._check_mamba_memory()
        else:
            memory_leak, token_msg = self._check_radix_cache_memory()

        if memory_leak:
            msg = "token_to_kv_pool_allocator memory leak detected! " f"{token_msg}"
            raise_error_or_warn(
                self,
                envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.get(),
                "count_memory_leak_warnings",
                msg,
            )

        self._check_req_pool()

        if (
            self.current_scheduler_metrics_enabled
            and time.perf_counter() > self.metrics_collector.last_log_time + 30
        ):
            # During idle time, also collect metrics every 30 seconds.
            if self.is_hybrid_swa:
                (
                    full_num_used,
                    swa_num_used,
                    full_token_usage,
                    swa_token_usage,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_swa_token_info()
                num_used = max(full_num_used, swa_num_used)
                token_usage = max(full_token_usage, swa_token_usage)
            elif self.is_hybrid_ssm:
                (
                    num_used,
                    _,
                    token_usage,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = self._get_mamba_token_info()
            else:
                num_used, token_usage, _, _ = self._get_token_info()

            priority_enabled = self.enable_priority_scheduling
            self.stats.num_running_reqs = QueueCount.from_reqs(
                self.running_batch.reqs, priority_enabled
            )
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.gen_throughput = 0
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )
            self.metrics_collector.log_stats(self.stats)
        self._publish_kv_events()

    def check_tree_cache(self: Scheduler):
        if (
            self.tree_cache.is_tree_cache()
            and (self.is_hybrid_swa and self.tree_cache.supports_swa())
            or (self.is_hybrid_ssm and self.tree_cache.supports_mamba())
        ):
            self.tree_cache.sanity_check()

    def self_check_during_idle(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if len(self.disagg_prefill_inflight_queue) > 0:
                return
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            queue_size = (
                len(self.waiting_queue)
                + len(self.disagg_decode_transfer_queue.queue)
                + len(self.disagg_decode_prealloc_queue.queue)
            )
            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                queue_size += len(self.decode_offload_manager.ongoing_offload)
            if queue_size:
                return
        elif self.enable_hisparse:
            if self.hisparse_coordinator.has_ongoing_staging():
                return

        self.check_memory()
        self.check_tree_cache()
        self.new_token_ratio = self.init_new_token_ratio
        self.maybe_sleep_on_idle()


def create_scheduler_watchdog(
    scheduler: Scheduler, watchdog_timeout: float, soft: bool = False
) -> WatchdogRaw:
    def dump_info() -> str:
        if scheduler.is_initializing or disable_request_logging():
            return ""
        if scheduler.is_hybrid_swa:
            _, info_msg = scheduler._check_hybrid_memory()
        elif scheduler.is_hybrid_ssm and scheduler.tree_cache.supports_mamba():
            _, info_msg = scheduler._check_mamba_memory()
        else:
            _, info_msg = scheduler._check_radix_cache_memory()
        return (
            f"{scheduler.cur_batch.batch_size()=}\n"
            f"{scheduler.cur_batch.reqs=}\n"
            f"{info_msg}"
        )

    return WatchdogRaw(
        debug_name="Scheduler",
        get_counter=lambda: getattr(scheduler, "forward_ct", 0),
        is_active=lambda: scheduler.is_initializing
        or getattr(scheduler, "cur_batch", None) is not None,
        watchdog_timeout=watchdog_timeout,
        soft=soft,
        dump_info=dump_info,
    )
