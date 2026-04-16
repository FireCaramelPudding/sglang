# KV Graft Stage2 `other analyst` 丢失问题排查与修复记录

本文档记录本轮 `sglang` 侧问题、定位过程、代码修改点、以及建议验证方式。

## 1. 问题现象

在 `debate` 的 Stage2 easy 模式下，使用：

- `METHOD=kv_graft`
- `ROPE_SHIFT=on`
- `ENABLE_KV_RESCALING=1`

运行真实实验时，模型经常输出类似内容：

- `The other analyst's response is not provided`
- `The other analyst's response is missing`

这说明 Stage2 第二段 graft 进来的 `other_answer` 没有被模型正确当作当前上下文的一部分。

## 2. 最初证据

相关运行日志显示，Stage2 easy 组装语义本身是正确的：

- `self_full` 作为第一段 graft
- `other_answer` 作为第二段 graft
- 只对第二段做 transform

`debate` 侧关键位置：

- `src/orchestrator_parallel.py`
  - `_slice_kv_cache()` 会在切 answer-only handle 时，把 `origin_start` 直接推进到切片后的真实绝对位置。
  - `_generate_with_interleaved_server_parts()` 会在 Stage2 easy 中用 `[False, True]` 作为 transform 标记。

对应代码位置：

- `debate/src/orchestrator_parallel.py:4593-4600`
- `debate/src/orchestrator_parallel.py:720-724`
- `debate/src/orchestrator_parallel.py:765-770`
- `debate/src/orchestrator_parallel.py:1892-1896`
- `debate/src/orchestrator_parallel.py:2014-2018`

日志中还能直接看到：

- `self_full=tuple(origin_start=0, ...)`
- `other_answer=tuple(origin_start=697, ...)`

这说明调用方已经把第二段 answer-only handle 的绝对起点算好了。

## 3. 第一轮已修问题

本轮之前，已经修过一个独立真实 bug：

### 3.1 问题

当 graft segment 同时满足：

- `token_start > 0`
- `transform` 开启
- `rope_shift` 开启

`scheduler` 传给 materializer 的 `origin_start` 少加了 `token_start` 偏移。

旧逻辑等价于：

```python
source_origin_start = segment.origin_start
```

但真实 source slice 位置应为：

```python
source_origin_start = segment.origin_start + token_start
```

### 3.2 修复

修复位置：

- `python/sglang/srt/managers/scheduler.py`

并补了 CPU regression：

- `test/srt/cpu/test_kv_graft_regressions.py`

### 3.3 结果

这修复了“base handle + token_start 切片”场景下的 rope 绝对位置错误。

但真实 `debate` 场景里，Stage2 仍然会出现 `not provided / missing`。说明还有第二个剩余问题。

## 4. 第二轮定位：为什么仍然错

继续排查后发现，Stage2 easy 的 `other_answer` 在 server-native graft 路径下，很多时候不是“base handle 再切片”的语义，而是“已经切好的 answer-only handle”。

也就是说：

- `segment.origin_start` 已经是 **切片后真实绝对起点**
- 同时 `segment.token_start` 仍然可能非零

而 `sglang` 之前统一采用：

```python
source_origin_start = segment.origin_start + token_start
```

这会导致：

- 对 base handle 场景：正确
- 对 pre-sliced handle 场景：重复加偏移，double count

## 5. 根因

根因在 `Scheduler._resolve_graft_segment()`。

文件：

- `python/sglang/srt/managers/scheduler.py`

旧逻辑无法区分两种不同语义：

1. **base handle 语义**
   - `segment.origin_start` 表示原始 handle 起点
   - 需要再加 `token_start`

2. **pre-sliced handle 语义**
   - `segment.origin_start` 已表示切片后真实起点
   - 不能再加 `token_start`

结果是 Stage2 第二段 `other_answer` 在做 rope transform 时位置再次漂移，模型感知到的是一段错位 KV，而不是正确的对方回答内容。

这正好解释了为什么模型会说：

- `other analyst's response is not provided`
- `missing`

本质不是“文本没传到”，而是“graft 进去的 KV 位置语义错了”。

## 6. 最终修复

修复位置：

- `python/sglang/srt/managers/scheduler.py:393-425`

新逻辑：

```python
entry_origin_start = int(
    getattr(entry.meta, "origin_start", segment.origin_start)
)
segment_origin_start = int(segment.origin_start)
source_origin_start = max(
    segment_origin_start, entry_origin_start + int(token_start)
)
```

### 6.1 修复含义

这个逻辑兼容两类调用：

- 如果 `segment.origin_start` 还是 base handle 起点，
  那么 `entry_origin_start + token_start` 会更大，得到正确 slice 起点。
- 如果 `segment.origin_start` 已经是 pre-sliced handle 的真实绝对起点，
  那么 `segment_origin_start` 会更大，避免再次加偏移。

### 6.2 为什么用 `max(...)`

因为当前两类调用都合法，而且都已在真实链路里出现。

`max(...)` 保持 KISS：

- 不新增协议字段
- 不要求 debate 与 sglang 同步升级 schema
- 只在 scheduler 内修正 source 绝对位置推导

## 7. 回归测试补充

测试文件：

- `test/srt/cpu/test_kv_graft_regressions.py`

保留旧回归：

### 7.1 base handle 场景

- `entry.meta.origin_start = 697`
- `segment.origin_start = 697`
- `token_start = 3`
- 期望 `origin_start == 700`

新增新回归：

### 7.2 pre-sliced handle 场景

- `entry.meta.origin_start = 0`
- `segment.origin_start = 700`
- `token_start = 3`
- 期望仍然 `origin_start == 700`
- 不能误算成 `703`

关键测试位置：

- `test/srt/cpu/test_kv_graft_regressions.py:313-395`

## 8. 本轮修改文件

### 8.1 运行时修复

- `python/sglang/srt/managers/scheduler.py`

### 8.2 回归测试

- `test/srt/cpu/test_kv_graft_regressions.py`

## 9. 定位过程摘要

本轮定位顺序如下：

1. 先确认 Stage2 easy 真实失败现象仍存在，错误样式为 `not provided / missing`。
2. 复查 `debate` 侧 Stage2 easy 组装逻辑，确认语义上确实在插入 `self_full + other_answer`。
3. 复查 `debate/src/orchestrator_parallel.py:_slice_kv_cache()`，发现 answer-only handle 的 `origin_start` 已经是切片后真实绝对起点。
4. 回看 `sglang` 的 `Scheduler._resolve_graft_segment()`，发现仍统一按 `segment.origin_start + token_start` 计算。
5. 判断这里对 pre-sliced handle 会 double count。
6. 修改 scheduler 推导逻辑，并补两个方向的回归测试，覆盖 base-handle 与 pre-sliced-handle 两类语义。

## 10. 验证命令

### 10.1 CPU regression

```bash
PYTHONPATH="/ssd/home/xiaoliangyang/sglang/python" "/home/xiaoliangyang/miniconda3/envs/sglang/bin/python" "/ssd/home/xiaoliangyang/sglang/test/srt/cpu/test_kv_graft_regressions.py"
```

预期：

- 新旧相关回归全部通过
- 总测试数比之前多 1 条

### 10.2 真实 debate 场景复跑

```bash
cd "/ssd/home/xiaoliangyang/debate" && METHOD=kv_graft ROPE_SHIFT=on ENABLE_KV_RESCALING=1 RUN_BASE="/ssd/home/xiaoliangyang/debate/runs/20260415" SAMPLES=50 DATA_PATH="/ssd/home/xiaoliangyang/debate/data/mmlu/mmlu_college_computer_science_50" AGENT1=cs AGENT2=math bash run_minimal_sglang_debate.sh
```

预期：

- `wrong_cases_*.html` 中 `The other analyst's response is not provided` / `missing` 应明显下降，理想情况消失。
- 如果仍有残留，再继续排查 rescale reference 是否还有独立问题。

## 11. 当前结论

当前可以明确下结论：

1. 本轮真实剩余问题在 `sglang`，不是 `debate` 文本拼接层。
2. 问题本质是 graft transform 的 source absolute position 计算同时要兼容两种 handle 语义。
3. 第一轮修复解决了“base handle + token_start”的 rope 偏移错误。
4. 第二轮修复解决了“pre-sliced handle + token_start”被重复加偏移的问题。
5. 这两个修复合起来，才完整覆盖 Stage2 easy 的真实 server-native graft 链路。
