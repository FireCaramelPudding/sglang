from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from sglang.srt.managers.io_struct import KVHandleMeta, KVTransformSpec
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


@dataclass
class KVHandleEntry:
    meta: KVHandleMeta
    device_indices: torch.Tensor
    token_ids: List[int]
    allocator: BaseTokenToKVPoolAllocator
    created_at: float = field(default_factory=time.monotonic)
    ttl_expiry: Optional[float] = None
    external_refcount: int = 1
    released: bool = False
    transform_provenance: Optional[List[KVTransformSpec]] = None

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_expiry is None:
            return False
        if now is None:
            now = time.monotonic()
        return now >= self.ttl_expiry


class KVHandleRegistry:
    def __init__(self, model_key: str, backend: str):
        self.model_key = model_key
        self.backend = backend
        self._entries: Dict[str, KVHandleEntry] = {}

    def _empty_items(
        self, allocator: BaseTokenToKVPoolAllocator
    ) -> torch.Tensor:
        return torch.empty((0,), dtype=torch.int64, device=allocator.device)

    def _normalize_to_allocator_items(
        self,
        indices: torch.Tensor,
        allocator: BaseTokenToKVPoolAllocator,
    ) -> torch.Tensor:
        if indices is None or indices.numel() == 0:
            return self._empty_items(allocator)
        items = indices.to(device=allocator.device, dtype=torch.int64)
        if allocator.page_size > 1:
            items = items // allocator.page_size
        return torch.unique(items)

    def _active_allocator_items(
        self, tree_cache: Any = None
    ) -> Dict[BaseTokenToKVPoolAllocator, torch.Tensor]:
        allocator_set: Dict[BaseTokenToKVPoolAllocator, None] = {}
        for entry in self._entries.values():
            if entry.released:
                continue
            allocator_set[entry.allocator] = None
        if tree_cache is not None:
            allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
            if allocator is not None:
                allocator_set[allocator] = None

        grouped: Dict[BaseTokenToKVPoolAllocator, List[torch.Tensor]] = {}
        for entry in self._entries.values():
            if entry.released:
                continue
            items = self._normalize_to_allocator_items(
                entry.device_indices, entry.allocator
            )
            if items.numel() == 0:
                continue
            grouped.setdefault(entry.allocator, []).append(items)

        merged: Dict[BaseTokenToKVPoolAllocator, torch.Tensor] = {}
        for allocator in allocator_set.keys():
            hold_counts = getattr(allocator, "external_hold_counts", None)
            if hold_counts is not None:
                held_items = torch.nonzero(
                    hold_counts > 0, as_tuple=False
                ).flatten().to(device=allocator.device, dtype=torch.int64)
                if held_items.numel() > 0:
                    held_items = held_items[held_items > 0]
                if held_items.numel() > 0:
                    merged[allocator] = held_items
                    continue

            item_groups = grouped.get(allocator, [])
            if item_groups:
                merged[allocator] = torch.unique(torch.cat(item_groups))

        return merged

    def _active_allocator_items_from_entries_only(
        self,
    ) -> Dict[BaseTokenToKVPoolAllocator, torch.Tensor]:
        grouped: Dict[BaseTokenToKVPoolAllocator, List[torch.Tensor]] = {}
        for entry in self._entries.values():
            if entry.released:
                continue
            items = self._normalize_to_allocator_items(
                entry.device_indices, entry.allocator
            )
            if items.numel() == 0:
                continue
            grouped.setdefault(entry.allocator, []).append(items)

        merged: Dict[BaseTokenToKVPoolAllocator, torch.Tensor] = {}
        for allocator, item_groups in grouped.items():
            merged[allocator] = torch.unique(torch.cat(item_groups))
        return merged

    def _cached_allocator_items(
        self,
        tree_cache: Any,
        allocator: BaseTokenToKVPoolAllocator,
    ) -> torch.Tensor:
        if tree_cache is None:
            return self._empty_items(allocator)

        total_size_fn = getattr(tree_cache, "total_size", None)
        if callable(total_size_fn):
            try:
                if total_size_fn() == 0:
                    return self._empty_items(allocator)
            except Exception:
                pass

        flatten_fn = getattr(tree_cache, "all_values_flatten", None)
        if not callable(flatten_fn):
            return self._empty_items(allocator)

        try:
            cached_indices = flatten_fn()
        except RuntimeError:
            return self._empty_items(allocator)

        return self._normalize_to_allocator_items(cached_indices, allocator)

    def _session_allocator_items(
        self,
        tree_cache: Any,
        allocator: BaseTokenToKVPoolAllocator,
    ) -> torch.Tensor:
        slots = getattr(tree_cache, "slots", None)
        req_to_token_pool = getattr(tree_cache, "req_to_token_pool", None)
        if not slots or req_to_token_pool is None:
            return self._empty_items(allocator)

        page_size = allocator.page_size
        held_indices: List[torch.Tensor] = []
        for slot in slots.values():
            if not getattr(slot, "is_holding_kv", False):
                continue
            if getattr(slot, "req_pool_idx", None) is None:
                continue

            start = int(getattr(slot, "cache_protected_len", 0) or 0)
            end = int(getattr(slot, "kv_allocated_len", 0) or 0)
            if page_size > 1 and end % page_size:
                end = ((end + page_size - 1) // page_size) * page_size
            if start >= end:
                continue

            held_indices.append(
                req_to_token_pool.req_to_token[slot.req_pool_idx, start:end].to(
                    dtype=torch.int64
                )
            )

        if not held_indices:
            return self._empty_items(allocator)

        return self._normalize_to_allocator_items(torch.cat(held_indices), allocator)

    def external_uncached_size(self, tree_cache: Any) -> int:
        total = 0
        for allocator, held_items in self._active_allocator_items(tree_cache).items():
            if held_items.numel() == 0:
                continue

            excluded_items: List[torch.Tensor] = []

            cached_items = self._cached_allocator_items(tree_cache, allocator)
            if cached_items.numel() > 0:
                excluded_items.append(cached_items)

            session_items = self._session_allocator_items(tree_cache, allocator)
            if session_items.numel() > 0:
                excluded_items.append(session_items)

            if excluded_items:
                accounted_items = torch.unique(torch.cat(excluded_items))
                held_items = held_items[~torch.isin(held_items, accounted_items)]

            total += int(held_items.numel()) * allocator.page_size

        return total

    def external_cached_size(self, tree_cache: Any) -> int:
        total = 0
        for allocator, held_items in self._active_allocator_items(tree_cache).items():
            if held_items.numel() == 0:
                continue

            cached_items = self._cached_allocator_items(tree_cache, allocator)
            if cached_items.numel() == 0:
                continue

            cached_held_items = held_items[torch.isin(held_items, cached_items)]
            if cached_held_items.numel() == 0:
                continue

            session_items = self._session_allocator_items(tree_cache, allocator)
            if session_items.numel() > 0:
                cached_held_items = cached_held_items[
                    ~torch.isin(cached_held_items, session_items)
                ]
            total += int(torch.unique(cached_held_items).numel()) * allocator.page_size

        return total

    def _entry_uncached_allocator_items(
        self, entry: KVHandleEntry, tree_cache: Any
    ) -> torch.Tensor:
        held_items = self._normalize_to_allocator_items(
            entry.device_indices, entry.allocator
        )
        if held_items.numel() == 0:
            return held_items

        excluded_items: List[torch.Tensor] = []
        cached_items = self._cached_allocator_items(tree_cache, entry.allocator)
        if cached_items.numel() > 0:
            excluded_items.append(cached_items)

        session_items = self._session_allocator_items(tree_cache, entry.allocator)
        if session_items.numel() > 0:
            excluded_items.append(session_items)

        if excluded_items:
            accounted_items = torch.unique(torch.cat(excluded_items))
            held_items = held_items[~torch.isin(held_items, accounted_items)]

        return held_items

    def _new_handle(
        self, name: Optional[str] = None, created_from_rid: Optional[str] = None
    ) -> str:
        prefix = "kvh"
        if name:
            return f"{prefix}_{name}_{uuid.uuid4().hex}"
        if created_from_rid:
            return f"{prefix}_{created_from_rid}_{uuid.uuid4().hex}"
        return f"{prefix}_{uuid.uuid4().hex}"

    def register(
        self,
        *,
        allocator: BaseTokenToKVPoolAllocator,
        device_indices: torch.Tensor,
        token_ids: Sequence[int],
        origin_start: int,
        dtype: str,
        created_from_rid: Optional[str],
        ttl_seconds: int,
        persist: bool,
        name: Optional[str] = None,
        composite: bool = False,
        transform: Optional[KVTransformSpec] = None,
        materialized: bool = False,
        transform_provenance: Optional[List[KVTransformSpec]] = None,
        compressed: bool = False,
        compression_type: Optional[str] = None,
        original_token_count: Optional[int] = None,
        compressed_token_count: Optional[int] = None,
        compression_spans: Optional[List[tuple[int, int]]] = None,
        handle: Optional[str] = None,
    ) -> KVHandleMeta:
        handle = handle or self._new_handle(name, created_from_rid=created_from_rid)
        if handle in self._entries and not self._entries[handle].released:
            raise ValueError(f"KV handle already exists: {handle}")
        indices = device_indices.detach().clone().to(dtype=torch.int64)
        allocator.hold(indices)
        meta = KVHandleMeta(
            handle=handle,
            backend=self.backend,
            token_count=len(token_ids),
            origin_start=origin_start,
            dtype=dtype,
            model_key=self.model_key,
            composite=composite,
            created_from_rid=created_from_rid,
            ttl_seconds=ttl_seconds,
            name=name,
            transform=transform,
            materialized=materialized,
            transform_provenance=transform_provenance,
            compressed=compressed,
            compression_type=compression_type,
            original_token_count=original_token_count,
            compressed_token_count=compressed_token_count,
            compression_spans=compression_spans,
        )
        entry = KVHandleEntry(
            meta=meta,
            device_indices=indices,
            token_ids=list(token_ids),
            allocator=allocator,
            ttl_expiry=(
                time.monotonic() + ttl_seconds if persist and ttl_seconds > 0 else None
            ),
            external_refcount=1 if persist else 0,
            transform_provenance=transform_provenance,
        )
        self._entries[handle] = entry
        return meta

    def lookup(
        self, handle: str, *, allow_expired: bool = False, tree_cache: Any = None
    ) -> KVHandleEntry:
        self.cleanup_expired(tree_cache=tree_cache)
        entry = self._entries.get(handle)
        if entry is None or entry.released:
            raise KeyError(f"Unknown KV handle: {handle}")
        if not allow_expired and entry.is_expired():
            self.release([handle], tree_cache=tree_cache)
            raise KeyError(f"Expired KV handle: {handle}")
        return entry

    def get_meta(self, handle: str) -> Optional[KVHandleMeta]:
        try:
            return self.lookup(handle).meta
        except KeyError:
            return None

    def export_debug_info(self, handle: str) -> Dict[str, Any]:
        entry = self.lookup(handle)
        return {
            "meta": entry.meta,
            "released": entry.released,
            "external_refcount": entry.external_refcount,
            "ttl_expiry": entry.ttl_expiry,
            "token_ids": entry.token_ids,
            "device_indices": entry.device_indices.tolist(),
            "transform_provenance": entry.transform_provenance,
        }

    def add_ref(self, handle: str):
        entry = self.lookup(handle)
        entry.external_refcount += 1

    def release(
        self, handles: Sequence[str], tree_cache: Any = None
    ) -> tuple[List[str], List[str]]:
        released: List[str] = []
        missing: List[str] = []
        for handle in handles:
            entry = self._entries.get(handle)
            if entry is None or entry.released:
                missing.append(handle)
                continue
            if entry.external_refcount > 0:
                entry.external_refcount -= 1
            if entry.external_refcount <= 0:
                if tree_cache is not None:
                    uncached_items = self._entry_uncached_allocator_items(
                        entry, tree_cache
                    )
                    if uncached_items.numel() > 0:
                        # allocator.free() expects token indices. Convert allocator
                        # items back to token-space representatives.
                        entry.allocator.free(
                            uncached_items.to(torch.int64) * entry.allocator.page_size
                        )
                entry.allocator.release_hold(entry.device_indices)
                entry.released = True
                self._entries.pop(handle, None)
            released.append(handle)
        return released, missing

    def cleanup_expired(self, tree_cache: Any = None):
        now = time.monotonic()
        expired = [
            handle
            for handle, entry in self._entries.items()
            if not entry.released and entry.is_expired(now)
        ]
        if expired:
            self.release(expired, tree_cache=tree_cache)

    def clear(self, tree_cache: Any = None):
        handles = list(self._entries.keys())
        if handles:
            self.release(handles, tree_cache=tree_cache)
