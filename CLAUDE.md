# SGLang 当前改动上下文

> 目的：本文件**只记录当前工作区相较于源码基线的改动点**，不重复介绍仓库原始架构。
> 记忆来源：`/home/xiaoliangyang/.codex/memories/*.md`
> 适用范围：帮助后续继续这批改动时，快速定位“我们额外加了什么、约束是什么、哪里容易出错”。

## 1. 改动主题总览

这一批改动主要围绕两条线：

1. **服务端原生跨上下文 KV 复用**
   - 在 SGLang 内引入 `kv_export` / `kv_graft` / handle 生命周期管理。
   - 目标是让调用方不必自己维护本地 `past_key_values`，而是通过服务端 handle 复用 KV。
2. **Qwen3 MoE 调优辅助链路**
   - 增加 topk ids 采集与 tuning 输入加载链路。

与记忆一致的外部背景：
- `debate` / `general-agentic-memory` 是这批能力的主要驱动方，但**本仓库只保留 serving/runtime 侧的实现与验证契约**。
- 当前文档只写本仓库实际变更，不展开外部仓库逻辑。

---

## 2. API / 协议层新增能力

### 2.1 Native 请求与返回结构扩展

核心文件：
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/detokenizer_manager.py`
- `python/sglang/srt/managers/tokenizer_communicator_mixin.py`

新增/扩展内容：
- `kv_graft`
- `kv_export`
- `kv_exports`
- handle release/debug 请求结构
- transform 相关 typed spec

当前语义：
- 请求可声明把已有 KV handle graft 为新的逻辑前缀。
- 请求也可在本次执行后导出一段逻辑 token 范围为新的 handle。
- `kv_exports` 需要穿过 tokenizer / detokenizer / HTTP 层完整返回。

### 2.2 HTTP / Engine / OpenAI 兼容入口扩展

核心文件：
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/entrypoints/engine.py`
- `python/sglang/srt/entrypoints/openai/protocol.py`
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/entrypoints/openai/serving_completions.py`
- `python/sglang/srt/entrypoints/openai/serving_responses.py`
- `python/sglang/srt/utils/json_response.py`

新增/扩展内容：
- Native endpoint：
  - `POST /kv_handles/release`
  - `GET /kv_handles/{handle}`
- OpenAI vendor 字段：
  - `sgl_kv_graft`
  - `sgl_kv_export`
  - `sgl_kv_exports`
- 修复 FastAPI `ORJSONResponse` 兼容问题。

当前约束：
- OpenAI 兼容层只做扩展字段透传，不改变原有主 schema。
- handle 被 release 后，再查 debug 接口返回 `404` 属于预期行为。

---

## 3. Scheduler / Runtime 主链路改动

核心文件：
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
- `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py`
- `python/sglang/srt/managers/utils.py`
- 新增 `python/sglang/srt/managers/kv_handle_registry.py`
- 新增 `python/sglang/srt/managers/kv_graft_materializer.py`

### 3.1 Synthetic prefix / graft 装配

当前实现不是把 graft 伪装成普通 radix prefix，而是显式走一条 handle-aware 路径：
- scheduler 解析 graft segments
- 构造 synthetic prefix
- 区分逻辑 token 空间与真实物理 KV 索引
- graft 请求显式禁用 radix match

这是本批改动的核心设计点。

### 3.2 Handle registry

`kv_handle_registry.py` 负责：
- exported handle 注册
- TTL 管理
- debug 元数据查询
- allocator hold / release_hold 协同
- 统计 external cached / uncached 占用

关键约束：
- handle 必须与当前 `model_key` / backend 匹配。
- TTL 是 opportunistic cleanup，不应假设严格实时回收。

### 3.3 Graft materialization / transform

`kv_graft_materializer.py` 支持：
- alias 复用
- owned segment 物化
- rope shift
- 统计 rescaling

结合记忆，需要始终保留以下约束：
- **只对新插入的 cross-context segment 做 transform**。
- **已处于当前 continuation 位置的 prefix 不能再次 transform**。
- `rope_theta` 必须取模型真实 HF config，不再写死 `10000`。
- SGLang KV 切片按 **token-major** 语义处理，不能直接套 HF 的 `[B, H, S, D]` 假设。

### 3.4 Export 注册时机

当前支持：
- prefill 后导出
- 完成后导出

并且导出范围基于**逻辑 token 范围**，不是直接暴露 allocator 物理范围。

---

## 4. 内存缓存与所有权语义改动

核心文件：
- `python/sglang/srt/mem_cache/allocator.py`
- `python/sglang/srt/mem_cache/common.py`

新增语义：
- allocator 支持 external `hold()` / `release_hold()`
- 支持 deferred free / pending free
- graft 请求释放路径区分：
  - alias segment
  - owned segment
  - live request segment

关键目的：
- 把“可复用 KV 的生命周期”从 radix tree 所有权中分离出来。
- 防止 exported handle 仍然有效时，底层 KV 页被提前认为可回收。

### 4.1 Runtime leak checker 修正

`scheduler_runtime_checker_mixin.py` 额外修了这类问题：
- external hold 导致的 accounting mismatch
- free list 与 live cache / hold 重叠
- orphan item 回收
- free list 去重与冲突清理

后续继续改这里时，优先保持以下不变量：
- session-held
- externally held uncached
- externally held cached
- pending free
- orphaned items

这些状态必须在检查逻辑里分开处理。

---

## 5. Qwen3 MoE / Kernel 调优辅助改动

核心文件：
- `python/sglang/srt/models/qwen3_moe.py`
- `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`
- `scripts/playground/moe_tuning/common.sh`
- `scripts/playground/moe_tuning/capture_qwen3_moe_topk_ids.sh`
- `scripts/playground/moe_tuning/install_qwen3_moe_configs.sh`
- `scripts/playground/moe_tuning/tune_qwen3_moe_configs.sh`
- `scripts/playground/moe_tuning/run_qwen3_moe_tuning_pipeline.sh`

当前新增能力：
- Qwen3 MoE 支持可选 topk ids 采样落盘。
- tuning 脚本改为从 `benchmark/kernels/fused_moe_triton/topk_ids/` 扫描并加载 `topk_ids_layer*_idx*.pt`。
- 仓库里当前已有一批 `Tongyi-DeepResearch-30B-A3B_tp2` 采样产物，可作为调优输入样本。

当前约束：
- topk ids 落盘受环境变量控制。
- 该链路是**调优辅助**，不是 serving 主路径必需逻辑。

---

## 6. 文档与测试补充

核心文件：
- `docs/developer_guide/cross_context_kv_graft.md`
- `docs/index.rst`
- `test/manual/entrypoints/http_server/test_kv_graft_smoke.py`
- `test/srt/cpu/test_kv_graft_regressions.py`

覆盖内容：
- 开发者文档：解释 `kv_export` / `kv_graft`、handle 生命周期、release/debug 语义。
- 手工 smoke：覆盖 export → graft → re-export → release 闭环。
- CPU 回归：覆盖 transform 顺序、export helper、prefill graft export 等回归点。

建议后续继续沿用：
- 任何修改 handle 生命周期、transform 顺序、export 注册逻辑的改动，都先补这里的测试。

---

## 7. 当前最重要的实现约束

这是后续继续修改时最不该丢的上下文：

1. **不要把 graft 重新塞回普通 radix prefix 语义。**
2. **导出 KV 的生命周期必须独立于 radix tree。**
3. **handle release 失败常常是下游症状，先看服务端是否更早崩溃。**
4. **transform 只作用于新 graft 段，不作用于 continuation 前缀。**
5. **rope theta 必须来自模型配置。**
6. **SGLang KV 是 token-major，不能直接照搬 HF 张量维度假设。**
7. **内存自检要把 cached / uncached / held / pending / orphan 分开算。**

---

## 8. 本仓库当前新增文件清单

功能性新增：
- `python/sglang/srt/managers/kv_handle_registry.py`
- `python/sglang/srt/managers/kv_graft_materializer.py`
- `docs/developer_guide/cross_context_kv_graft.md`
- `test/manual/entrypoints/http_server/test_kv_graft_smoke.py`
- `test/srt/cpu/test_kv_graft_regressions.py`
- `scripts/playground/moe_tuning/*`

数据/产物类新增：
- `benchmark/kernels/fused_moe_triton/topk_ids/Tongyi-DeepResearch-30B-A3B_tp2/*.pt`

非功能性本地工件：
- `.codex_write_probe`

处理建议：
- `.codex_write_probe` 不属于产品功能，不应写入正式设计说明或提交主逻辑文档。

---

## 9. 后续进入这批改动时的优先阅读顺序

1. `docs/developer_guide/cross_context_kv_graft.md`
2. `python/sglang/srt/managers/kv_handle_registry.py`
3. `python/sglang/srt/managers/kv_graft_materializer.py`
4. `python/sglang/srt/managers/scheduler.py`
5. `python/sglang/srt/mem_cache/allocator.py`
6. `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py`
7. `test/srt/cpu/test_kv_graft_regressions.py`
8. `test/manual/entrypoints/http_server/test_kv_graft_smoke.py`

如果目标是继续做 MoE tuning，再读：
- `python/sglang/srt/models/qwen3_moe.py`
- `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`
- `scripts/playground/moe_tuning/run_qwen3_moe_tuning_pipeline.sh`
