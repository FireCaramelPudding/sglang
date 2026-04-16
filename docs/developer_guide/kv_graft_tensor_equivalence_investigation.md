# KV Graft Tensor 等价性问题排查记录

本文档记录当前 `kv_graft` 基础链路中的一个关键问题：

**同一份逻辑文本，在服务端 direct prefill/export 与 graft prefill/export 两条路径下，导出的真实 KV tensor 不一致。**

这不是最终生成文本层面的现象，而是对 handle 背后真实 K/V 张量的直接比对结果。

## 背景

在 `debate` 场景中，Stage2 会复用前序阶段导出的 KV handle。此前已经观察到：

- `graft`
- `graft + rope`
- `graft + rescaling`
- `graft + rope + rescaling`

四组结果完全一致。

这意味着问题优先级应从 rope/rescaling 具体实现，回退到更基础的 graft/export/handle 映射链路。

换句话说，如果基础 graft 就已经把 KV 复用错了，那么无论后面再开不开 rope 或 rescaling，结果都可能一样地错。

## 目标

不再依赖最终生成文本做间接判断，而是直接验证三份 KV 是否一致：

1. **服务端 direct export**：对完整 `full_ids` 直接 prefill，然后导出 KV。
2. **服务端 graft export**：先导出 `prefix_ids`，再 graft `prefix handle + suffix_ids`，再导出 merged KV。
3. **本地 HuggingFace prefill**：对相同完整 `full_ids` 做一次前向，取 `past_key_values` 作为参考。

## 新增测试

测试文件：

- `test/manual/entrypoints/http_server/test_kv_graft_tensor_equivalence.py`

核心设计：

### Case A: direct export

- `input_ids = full_ids`
- `max_new_tokens = 0`
- 导出 `[0, len(full_ids))` 的整段 KV

### Case B: graft export

- 先对 `prefix_ids` 预填充并导出 `prefix_handle`
- 再发起：
  - `input_ids = suffix_ids`
  - `kv_graft.segments = [{handle: prefix_handle, origin_start: 0}]`
  - 不启用 transform
  - 再导出 `[0, len(full_ids))` 的 merged KV

### Case C: local HF

- 使用 `AutoModelForCausalLM(..., use_cache=True)`
- 对相同 `full_ids` 做 prefill
- 取 `past_key_values`
- 转换成与服务端一致的 token-major 视图后比较

## 调试链路状态

排查中确认，服务端最小调试能力其实已经存在，无需再新加接口：

- `GET /kv_handles/{handle}`
- `GET /kv_handles/{handle}/tensors`
- `POST /kv_handles/release`

相关实现已位于：

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_communicator_mixin.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/entrypoints/http_server.py`

## 已修复的独立问题

在运行 tensor 对比测试时，调试接口曾先因以下问题崩溃：

- `scheduler.py` 中 `get_kv_handle_tensors()` 使用了不存在的 `self.is_mla`

实际修复方式：

- 改为使用 `self.kv_handle_registry.backend == "mla"` 判断后端类型

这个问题只影响 debug tensor dump，**不是本次 KV 语义错误的根因**。

## 当前结论

当前最重要结论如下：

1. 测试首先失败在 **direct vs graft**。
2. 失败发生在与 HF 对比之前。
3. 因此，问题已经可以在**纯服务端两条路径之间**复现。

这说明：

**同一逻辑文本，通过 direct export 与 graft export 得到的真实 KV 已经不一致。**

因此当前最强判断是：

- 问题更可能位于 `sglang` 服务端基础 graft/export/handle 映射链路
- 而不是 debate 端 prompt 语义拼接
- 也不是 rope/rescaling 变换细节
- 也还不是 HF 参考实现差异

## 问题性质判断

这个失败意味着以下至少一类问题存在：

- graft 后 merged handle 的 `device_indices` 指向了错误 KV 槽位
- composite export 导出时，逻辑 token 顺序与物理 KV 顺序不一致
- prefix 段或 suffix 段在 graft 路径被错误拼接、错读或覆盖
- `req_to_token_pool` / `token_to_kv_pool` 的索引来源在 graft/export 场景下不正确
- `token_ids` 看起来正确，但其对应的真实 KV 内容并不是预期那一批

## 为什么这条证据很强

本测试是极简 alias graft：

- 同一模型
- 同一完整字符串
- 不启用 transform
- 不比较生成文本
- 直接比较真实 K/V 张量

在这种条件下，`direct` 与 `graft` 理应逐层逐 token 等价。

现在它们不等，说明问题已经发生在最基础的 KV 复用路径，而不是更上层的语义链路。

## 下一步排查重点

下一步不应优先继续查 rope/rescaling，而应优先看基础 graft/export：

1. `composite export` 构造 merged handle 时，token 顺序是否正确
2. handle 的 `device_indices` 是否与 merged logical token 一一对应
3. prefix 与 suffix 的边界 token 是否在 graft/export 时错位
4. `req_to_token_pool` 与 `token_to_kv_pool` 在 graft 场景下是否存在错误索引来源
5. `token_ids` 与真实 KV 内容之间是否脱钩

## 日志判读要点

本轮测试日志已落盘到：

- `/ssd/home/xiaoliangyang/debate/kv_graft_tensor_equivalence.log`

后续解读时，重点看 direct vs graft 失败项中的：

- `layer`
- `tensor` (`k` / `v`)
- `max_abs_diff`
- `first_bad_index`

其中最关键是 `first_bad_index[0]`，因为当前比较视图是 token-major：

- 若 `< len(prefix_ids)`，说明 prefix 段本身就已错
- 若 `>= len(prefix_ids)`，说明问题可能从 graft 后追加的 suffix 段开始
- 若所有层都从同一 token 开始坏，优先怀疑边界/offset/export range
- 若几乎所有 token 都大幅偏离，优先怀疑读错槽位，而不是数值误差

## 相关文件

### 测试与日志

- `test/manual/entrypoints/http_server/test_kv_graft_tensor_equivalence.py`
- `/ssd/home/xiaoliangyang/debate/kv_graft_tensor_equivalence.log`

### 调试链路

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_communicator_mixin.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/entrypoints/http_server.py`

### 重点嫌疑位置

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/kv_graft_materializer.py`
- `python/sglang/srt/managers/kv_handle_registry.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/mem_cache/allocator.py`

## 当前阶段结论

阶段性结论可简写为：

> `graft` 基础链路已被 tensor 级测试直接证伪。
> 当前应优先排查服务端 graft/export/handle 映射，而不是继续怀疑 rope/rescaling 或 debate 端 prompt 组装。
