# 跨上下文 KV Graft/Export 开发者指南

本文档说明本地 SGLang 分支中，为支持服务端原生跨上下文 KV 复用而引入的 `kv_export` 与 `kv_graft` 改造。

这次工作的目标是：让一个请求可以把可复用的 KV 切片导出为 handle，而后续请求可以把一个或多个已导出的 KV 切片 graft 到新的逻辑 prompt 中，而不需要调用方自己重建 `past_key_values`。

## 范围

本文档覆盖以下内容：

- 内部数据模型与请求生命周期
- 内存所有权与释放语义
- Native HTTP、Engine 与 OpenAI 兼容接口
- KV handle 的调试与释放接口
- 验证流程，以及为什么 release 后看到 `404` 是预期行为
- 本次改造中顺手修复的 FastAPI `ORJSONResponse` 兼容问题

本文档不假定存在某个特定的外部 orchestrator。就当前仓库快照而言，服务端已经暴露出跨上下文链式调用所需的契约。

## 为什么需要这个能力

在这次改造之前，跨上下文复用通常意味着调用方必须自己持有模型本地 KV 状态，或者在服务端之外重建 prompt 状态。这对下面这些场景都不太友好：

- 多轮 agent 流水线
- fan-out / fan-in 编排
- 跨进程请求接力
- 长链条推理流程中对 answer-only 或 merged-prefix KV 的复用

新的设计把复用逻辑下沉到了 serving 层：

1. 请求 A 可以把一个逻辑 KV 切片导出成 handle。
2. 请求 B 可以把一个或多个 handle graft 成 synthetic prefix。
3. 请求 B 还可以继续把它的 merged prefix 再导出给下一跳使用。

## 关键术语

- `KV handle`：服务端侧对一段 KV 切片及其元数据的命名引用
- `kv_export`：请求期指令，告诉服务端把哪一段逻辑 token 范围发布成 handle
- `kv_graft`：请求期指令，告诉服务端把哪些已有 handle 拼接进新请求
- `synthetic prefix`：由 graft 片段拼出来、位于当前 live prompt 之前的逻辑前缀
- `aliased segment`：直接复用源 KV 页的 graft 片段
- `owned segment`：会重新分配 KV 页，并把源 KV 拷贝/变换到新页中的 graft 片段
- `logical token ids`：API 视角下可见的 token ids，包含 graft 得到的 synthetic prefix
- `physical indices`：真正承载这些逻辑 token 的 allocator / KV-pool 索引
- `composite handle`：导出的范围与 graft synthetic prefix 有重叠的 exported handle

## 高层架构

这个能力横跨四层实现：

1. 请求 schema 与协议层
   - 新增 `kv_graft`、`kv_export`、handle 元数据，以及 OpenAI 扩展字段。
2. Scheduler 请求装配层
   - 解析 handle 引用、物化 synthetic prefix，并决定所有权模式。
3. 内存与 handle registry 层
   - 跟踪已导出 KV 页、TTL、引用 hold 与调试元数据。
4. 输出回传层
   - 在合适的时机注册 export，并把 handle 元数据返回给调用方。

## 文件地图

本次工作新增或修改的核心文件如下：

- `python/sglang/srt/managers/io_struct.py`
  - 新增 `KVTransformSpec`、`KVGraftSegment`、`KVGraftSpec`、`KVExportSpec`、`KVHandleMeta`，以及 release/debug 相关请求输出结构。
- `python/sglang/srt/managers/kv_handle_registry.py`
  - 新增 exported handle 注册表，负责 TTL 清理、debug 信息与 allocator hold 跟踪。
- `python/sglang/srt/managers/kv_graft_materializer.py`
  - 新增 MHA 与 MLA 后端的 KV 变换层。
- `python/sglang/srt/managers/scheduler.py`
  - 负责解析 graft 片段、装配 synthetic prefix、注册 export。
- `python/sglang/srt/managers/schedule_batch.py`
  - 扩展 `Req`，加入 synthetic prefix、graft 所有权以及逻辑 token 辅助函数。
- `python/sglang/srt/mem_cache/allocator.py`
  - 新增 external hold 计数与延迟 free 逻辑。
- `python/sglang/srt/mem_cache/common.py`
  - 新增 graft 请求的特殊清理逻辑。
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
  - 触发 export 注册，并通过 scheduler output 传递 `kv_exports`。
- `python/sglang/srt/managers/detokenizer_manager.py`
  - 保留字符串输出路径上的 `kv_exports`。
- `python/sglang/srt/managers/tokenizer_manager.py`
  - 在入口处解析 typed spec，并把 `kv_exports` 镜像回 response metadata 与 native 顶层输出。
- `python/sglang/srt/managers/tokenizer_communicator_mixin.py`
  - 新增 handle release 与 handle debug RPC。
- `python/sglang/srt/entrypoints/engine.py`
  - 新增离线 Engine 的 release/debug 辅助方法。
- `python/sglang/srt/entrypoints/http_server.py`
  - 新增 `/kv_handles/release` 与 `/kv_handles/{handle}` 两个 endpoint。
- `python/sglang/srt/entrypoints/openai/protocol.py`
  - 新增 `sgl_kv_graft`、`sgl_kv_export` 与 `sgl_kv_exports`。
- `python/sglang/srt/entrypoints/openai/serving_*.py`
  - 把 OpenAI 扩展字段向内透传，并把 exported handles 向外返回。
- `python/sglang/srt/utils/json_response.py`
  - 替换了对 FastAPI 已弃用 `ORJSONResponse` 的依赖。
- `test/manual/entrypoints/http_server/test_kv_graft_smoke.py`
  - export、graft、re-export、debug、release 的端到端 smoke test。

## 数据模型

### `KVTransformSpec`

`KVTransformSpec` 用来控制 graft 时可选的变换行为：

- `rope_shift`
  - `off`：不做 rope 位置调整
  - `on`：总是按新逻辑位置去平移 KV
  - `auto`：当前实现上走与 `on` 相同的路径，用作调用方友好的默认值
- `rescale_profile`
  - 当前支持在存在 reference prefix 时，将源统计量匹配到参考前缀
- `rescale_params`
  - 为后续扩展预留的变换参数

### `KVGraftSpec`

`KVGraftSpec` 由一个或多个 `KVGraftSegment` 组成。每个 segment 指定：

- `handle`
- 可选 `token_start`
- 可选 `token_end`
- `origin_start`
- 可选 `transform`

这意味着调用方既可以复用整个 exported handle，也可以只取其中一个切片。

### `KVExportSpec`

`KVExportSpec` 定义了当前请求中要导出的逻辑范围：

- 可选 `token_start`
- 可选 `token_end`
- `origin_start`
- `persist`
- `ttl_seconds`
- 可选 `name`

这里的导出范围基于请求的逻辑 token 空间，而不是底层物理 allocator 空间。

### `KVHandleMeta`

每个 exported handle 会记录：

- `handle`
- `backend`
- `token_count`
- `origin_start`
- `dtype`
- `model_key`
- `composite`
- `created_from_rid`
- `ttl_seconds`
- 可选 `name`
- 可选 `transform`

## 端到端请求流程

### 流程 A：导出一个 Handle

请求 A 发送普通 generate 请求，并携带 `kv_export`。

```text
caller
  -> tokenizer manager
  -> scheduler
  -> normal prefill/decode
  -> scheduler registers export on committed logical KV
  -> detokenizer/tokenizer manager attach kv_exports to response
  -> caller receives handle metadata
```

### 流程 B：把一个 Handle Graft 到新请求中

请求 B 发送 `kv_graft`，并可选地附带 `kv_export`。

```text
caller
  -> tokenizer manager parses kv_graft/kv_export
  -> scheduler resolves each handle from registry
  -> scheduler builds synthetic prefix token ids + physical indices
  -> request runs with disable_radix_match=True
  -> synthetic prefix counts as already-cached prompt KV
  -> request may export a merged prefix again
```

### 流程 C：释放一个 Handle

当调用方不再需要某些 handle 时，可以显式释放它们。

```text
caller
  -> POST /kv_handles/release
  -> registry drops exported entry
  -> allocator hold count is decremented
  -> physical KV pages are freed if no request/cache/export still owns them
```

## 详细内部流程

### 1. 入口：解析 Typed Spec

`GenerateReqInput` 现在接受 `kv_graft` 和 `kv_export`。

`TokenizerManager` 会把它们归一化成 typed object：

- `KVGraftSpec`
- `KVExportSpec`

这一步发生在请求进入 scheduler 之前，因此 scheduler 内部始终处理结构化 spec。

### 2. Scheduler：组装 Synthetic Prefix

`Scheduler.handle_generate_request()` 在创建 `Req` 之后，会调用 `_apply_kv_graft()`。

`_apply_kv_graft()` 会：

- 保存 `req.kv_export_spec`
- 解析每一个 graft segment
- 把 segment token ids 拼接到 `req.synthetic_prefix_token_ids`
- 把 segment physical indices 拼接到 `req.synthetic_prefix_indices`
- 设置 `req.disable_radix_match = True`
- 记录每个 segment 是 aliased 还是 owned

这一步把“跨上下文复用”转换成了 runtime 后续可以像普通请求一样处理的内部表示。

### 3. Segment 解析：Alias 与 Transform

对每个 graft segment，`_resolve_graft_segment()` 都会先从 `KVHandleRegistry` 查到 handle，然后校验：

- `model_key` 一致
- `backend` 一致
- slice 范围合法

之后分成两条路径。

Alias 路径：

- 当没有请求 transform 时走这条路
- 直接对现有 exported indices 做 allocator hold
- 不发生 KV copy

Owned 路径：

- 当 segment 请求 transform 时走这条路
- 新分配 KV 页
- `KVGraftMaterializer` 把源 KV 拷贝到新页
- 可选 rope shift 与 rescale 也发生在这一步

这个区分非常重要，因为 alias 代价很低，而 transform graft 需要真实的分配与拷贝。

### 4. 逻辑 Prompt 与物理 KV 的区别

现在一个请求同时拥有两种 prompt 视图：

- 逻辑视图
  - `synthetic_prefix_token_ids + origin_input_ids + output_ids`
- 物理视图
  - 实际承载 committed KV 的 allocator indices

为了让下游在计算导出范围与输入长度时都使用逻辑视图，新增了：

- `Req.prompt_token_count`
- `Req.logical_input_ids`
- `Req.logical_fill_ids`
- `Req.get_exportable_logical_token_ids()`

这避免了这样的 bug：当前 live prompt 前面有 graft 进来的 KV，但这些 token 并不是这次 tokenizer pass 产生的，结果导出范围或长度判断发生偏移。

### 5. Prefix Cache 集成

一旦请求拥有 grafted prefix，它就不能再回到普通 radix match 路径上。为此改了两个地方：

- `SchedulePolicy._compute_prefix_matches()`
- `Req.init_next_round_input()`

当 `disable_radix_match=True` 时，这两条路径都会直接使用 `synthetic_prefix_indices`，并据此设置 `cache_protected_len`。

这样 scheduler 就不会再试图通过 radix tree 去重新匹配或重新“接管” synthetic prefix。

### 6. Export 注册时机

export 注册基于 committed KV，而不是 speculative 或未来 KV。

当前有两个注册时机：

- `_maybe_register_prefill_graft_export(req)`
  - 在 prefill 路径上，第一个 sampled token 被 append 后立即触发
  - 适合 graft 请求想尽早发布 merged prefix 的场景
- `_maybe_register_kv_export(req)`
  - 在请求 finished 时触发

导出路径具体会：

1. 选择请求指定的逻辑 token 范围
2. 通过 `req_to_token_pool` 把这段逻辑范围映射成 physical indices
3. 判断这次 export 是否为 `composite`
4. 调用 `KVHandleRegistry` 注册 handle
5. 把得到的 metadata 存入 `req.kv_exports`

这里特意按 logical token ids 来计算范围，保证 exported handle 描述的是“调用方认为自己导出的那一段”。

### 7. 输出回传

一旦 `req.kv_exports` 被设置，它会沿着以下链路一路向上透传：

- scheduler batch output
- detokenizer output
- tokenizer manager response assembly

对于 native `/generate`，`TokenizerManager` 会把 handle metadata 同时镜像到：

- `meta_info.kv_exports`
- 顶层 `kv_exports`

顶层镜像是一个便利契约，让调用方不必总是去 nested metadata 里找。

## 内存所有权与释放语义

这部分是运行时最关键的内容。

### 问题本身

跨上下文复用意味着同一批 KV 页可以活得比最初生成它们的请求更久。这打破了旧假设：请求完成后，就可以立即把它对应的 KV 页全部释放掉。

### Allocator Hold 跟踪

`BaseTokenToKVPoolAllocator` 现在会跟踪：

- `external_hold_counts`
- `pending_free_counts`

新增的方法有：

- `hold(indices)`
- `release_hold(indices)`
- 当页面仍被外部持有时的 deferred free 行为

这让“请求本身”“radix cache”和“handle registry”三者可以安全地共享或移交所有权。

### Registry 持有的 KV

`KVHandleRegistry.register()` 会：

- clone 导出的 physical indices
- 调用 `allocator.hold(indices)`
- 保存 token ids、metadata、TTL 和 provenance

因此 registry 不再只是一个 metadata map，而是 exported KV 的真实 owner 之一。

### Graft 请求的清理逻辑

`release_kv_cache()` 对 `disable_radix_match=True` 的请求有专门处理。

它会把请求涉及的区域分成三类：

- owned transformed graft pages
- aliased graft pages
- 当前请求自己生成出来的 live pages

清理规则如下：

- owned transformed pages 正常 free，除非它们仍被某个 export 持有
- aliased graft pages 不直接 free，而是 release 它们的 hold
- 当前请求自己的 pages 正常 free

这正是“纯 graft 请求不泄漏 transformed pages，同时已 export 页也不会被过早释放”的关键。

### TTL 与显式 Release

handle 有两种消失方式：

- 通过 `/kv_handles/release` 显式释放
- 在 scheduler 处理请求时 opportunistic 地做 TTL cleanup

一旦 handle 被释放或过期，`GET /kv_handles/{handle}` 就会返回 `404`。

这里的 `404` 是预期行为。在当前 smoke test 中，测试会在 release 之后故意检查 `404`，用来证明这个 handle 已经真正消失了。

## 调试与控制接口

### Native HTTP

#### generate 时导出

```json
{
  "input_ids": [100, 101, 102, 103, 104],
  "sampling_params": {
    "temperature": 0,
    "max_new_tokens": 2,
    "min_new_tokens": 2,
    "ignore_eos": true
  },
  "kv_export": {
    "token_start": 5,
    "origin_start": 0,
    "persist": true,
    "ttl_seconds": 300,
    "name": "answer-a"
  }
}
```

#### Graft 并再次导出

```json
{
  "input_ids": [201, 202],
  "sampling_params": {
    "temperature": 0,
    "max_new_tokens": 1,
    "min_new_tokens": 1,
    "ignore_eos": true
  },
  "kv_graft": {
    "segments": [
      {
        "handle": "kvh_answer-a_xxx",
        "origin_start": 0
      }
    ]
  },
  "kv_export": {
    "origin_start": 0,
    "token_end": 4,
    "persist": true,
    "ttl_seconds": 300,
    "name": "merged-prefix-b"
  }
}
```

#### 调试一个 handle

`GET /kv_handles/{handle}`

成功响应包含：

- `success`
- `handle_meta`
- `message`

#### 释放一批 handles

`POST /kv_handles/release`

```json
{
  "handles": ["kvh_answer-a_xxx", "kvh_merged-prefix-b_xxx"]
}
```

响应包含：

- `success`
- `released_handles`
- `missing_handles`
- `message`

### Offline Engine API

离线 Engine 现在暴露了：

- `Engine.release_kv_handles(handles)`
- `Engine.async_release_kv_handles(handles)`
- `Engine.get_kv_handle(handle)`
- `Engine.async_get_kv_handle(handle)`

这些方法封装的是与 HTTP endpoint 相同的服务端 registry 逻辑。

### OpenAI 兼容 API 扩展

OpenAI 请求模型现在接受：

- `sgl_kv_graft`
- `sgl_kv_export`

这些字段会由 `OpenAIServingBase._get_sgl_kv_fields()` 提取出来，然后透传到 `GenerateReqInput`。

导出的 handle metadata 会以如下形式返回：

- chat/completions：`sglext.sgl_kv_exports`
- Responses API 最终 metadata：`metadata.sgl_kv_exports`

这样就能在不改动 OpenAI 主 schema 字段的前提下，提供这项能力。

## Transform 语义

transform 层位于 `kv_graft_materializer.py`。

当前支持的 materializer：

- `MHAGraftMaterializer`
- `MLAGraftMaterializer`

当前支持的变换操作包括：

- 对 key 做 rope shift
- 在有 reference prefix 时做可选的统计匹配

只有在 graft segment 显式请求 transform 时，才会走这个层。普通复用仍然走最便宜的 alias 路径。

## 兼容性约束

一个 graft 进来的 handle 必须匹配：

- 当前模型的 `model_key`
- 当前运行时的 `backend`

因此 handle 被刻意设计成不跨以下边界复用：

- 不同模型
- 不同 attention backend
- 不同运行中的服务实例

这些约束会在 lookup 时被强制校验。

## 验证、指标与泄漏检查

为了把 exported handles 视为真实 owner，一系列 runtime check 也做了同步调整：

- memory leak 检查现在会扣除 externally held tokens
- schedule policy 能识别 synthetic prefix
- input validation 与 auto-truncate 使用逻辑 prompt 长度

如果没有这些改动，runtime 很容易出现以下问题：

- 报出假的 KV leak
- 过早释放 exported pages
- 在 graft 前缀存在时错误计算 prompt 长度

## Smoke Test

主要验证用例在：

- `test/manual/entrypoints/http_server/test_kv_graft_smoke.py`

这个 smoke test 会验证：

1. 请求 A 导出 answer-only handle。
2. 请求 B graft 这个 handle 并导出 merged prefix。
3. release 前，两者都能通过 debug endpoint 看到。
4. 两个 handle 都能成功释放。
5. 释放后，这两个 handle 会返回 `404`。

重要说明：当前测试日志里最后出现的 `404` 是预期行为，表示 release 真正生效了。

## FastAPI `ORJSONResponse` 兼容修复

在验证这些新 endpoint 时，FastAPI 报出了：

```text
FastAPIDeprecationWarning: ORJSONResponse is deprecated
```

这不是 KV graft 逻辑 bug，而是 response 层兼容问题：FastAPI 已经废弃了它内置的 `ORJSONResponse`。

修复方式如下：

- `python/sglang/srt/utils/json_response.py`
  - `SGLangORJSONResponse` 现在直接继承 `Response`
- `python/sglang/srt/entrypoints/http_server.py`
  - 现有的 `ORJSONResponse(...)` 调用被路由到本地兼容类

修复结果：

- KV graft smoke test 中不再出现 FastAPI 弃用警告
- 现有的 ORJSON 序列化选项保持不变

## 已知限制

- handle 是 server-local 且 backend-specific 的。
- TTL cleanup 是 opportunistic 的，发生在 scheduler 处理请求过程中，而不是独立后台清理器。
- 当前 smoke test 虽然证明了正确性，但它在 release 后主动检查 `404`，因此日志里会显得有点“吵”。
- 当前仓库快照主要落地了 server-native 能力。外部 orchestration 调用方已经可以接这个契约，但调用侧迁移代码可能位于本仓库之外。

## 建议的调用侧接入模式

对于外部 agent 或 orchestrator，推荐的调用流程是：

1. 发起一次带 `kv_export` 的 generate
2. 提取 `kv_exports[0].handle`
3. 在后续请求的 `kv_graft.segments[*].handle` 中使用这个 handle
4. 如有需要，再次导出 merged prefix
5. 当整条链路不再需要这些 handle 时，显式释放它们

这样可以把 KV 复用尽量留在服务端内部，而不必在用户态编排代码里传递原始 KV tensor。

## 后续开发建议

- 如果要新增 transform 类型，优先在 `kv_graft_materializer.py` 中实现，并明确区分 alias 与 owned 语义。
- 如果要扩展新的对外 API，请保持现有契约：
  - 请求字段可以透传进去
  - `kv_exports` metadata 能稳定返回出来
- 如果要修改请求清理逻辑，请重点复查 `release_kv_cache()`；这里最容易引入 leak 或 premature free。
- 如果要调整 handle schema，需要同步更新：
  - `io_struct.py`
  - native `/generate` response assembly
  - OpenAI `sglext` / `metadata` 透传
  - smoke tests

## 最小验证清单

- Native `/generate` 携带 `kv_export` 时能够返回 handle metadata。
- Native `/generate` 携带 `kv_graft` 时能够复用 cached prefix，并支持再次导出。
- `/kv_handles/{handle}` 在 release 前能返回 debug metadata。
- `/kv_handles/release` 能正确释放 handle，并返回正确的 bookkeeping。
- 已释放 handle 会返回 `404`。
- OpenAI chat/completions/responses 能正确透传 `sgl_kv_graft` 与 `sgl_kv_export`。
- OpenAI responses 能把 `sgl_kv_exports` 回传给调用方。
- smoke validation 期间不再出现 FastAPI `ORJSONResponse` 弃用警告。
