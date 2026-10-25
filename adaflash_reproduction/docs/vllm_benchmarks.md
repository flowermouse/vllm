# vLLM 五项 benchmark 复现

入口：[scripts/tools/reproduce_vllm.sh](../scripts/tools/reproduce_vllm.sh)，CLI：[bin/reproduce_vllm.py](../bin/reproduce_vllm.py)，实现：[pipeline/benchmark/vllm.py](../pipeline/benchmark/vllm.py)。

三种模式的目标模型固定为 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`：

| 模式 | 草拟模型 | 动态验证 |
| --- | --- | --- |
| vanilla | 无 | 无 speculative decoding |
| fixed | `AdaFlash/Qwen3-8B-AdaFlash` | 关闭，固定 15 个 draft tokens |
| adaptive | 同一 AdaFlash checkpoint | 开启，由发布的 length head 预测 |

两种 SD 模式都固定 drafter revision `6d9b5c2fd6a0c7bab8436bbca67c380dcdfc547f`，不将不同 drafter 的训练差异算成动态长度收益。

## 数据与默认协议

| 名称 | Hugging Face 来源 | 原始行数 / 选择 |
| --- | --- | --- |
| GSM8K | `openai/gsm8k`, main/test | 1319；固定 seed 42 抽取至 1024 |
| MATH500 | `HuggingFaceH4/MATH-500`, test | 沿用本仓库准备脚本的最后 50 题留出集 |
| HumanEval | `openai/openai_humaneval`, test | 164 |
| MBPP | `google-research-datasets/mbpp`, full/test | 500 |
| MT-Bench | `HuggingFaceH4/mt_bench_prompts`, train | 80，沿用原客户端只使用第一轮 |

复用 `pipeline/benchmark/run.py` 的 JSONL loader、GSM8K 推理指令与答案提取函数。新数据适配器将 HumanEval 包装成代码求解指令，MBPP 包含描述和公开测试。数据下载后固定 HF revision、保存来源与 SHA256；再次运行校验数据哈希。

每个集合每种模式测量 **256 条请求、64 并发上限、3 次**，另用 64 条请求、最多 64 个输出 tokens warmup。与原客户端一致，数据不足时循环使用，测量从 warmup 后的第 65 条开始。小集合存在重复，因此这是一组可复现吞吐负载，不是去重后的完整准确率评测。

所有模式统一：BF16、TP1、`TRITON_ATTN`、full decode CUDA Graph、最大上下文 8192、每条最多输出 2048 tokens、2048-token 调度预算、32 GiB KV、greedy、关闭 thinking 和 prefix caching。没有静默截断输入或跳过失败请求。达到输出上限的请求数会写入结果。

默认启用 `VLLM_BATCH_INVARIANT=1`，校验同模式各轮及不同模式的逐 token 输出，避免生成量变化影响速度比较。测量的是离线 `LLM.generate()` 引擎吞吐，包含 prefill/decode 和队列尾部，排除加载、编译和 warmup，不含 HTTP 网络。最终表格将报告三次吞吐的均值与样本标准差。

## 一键运行

在 A800 上：

```bash
source ~/adaflash-vllm/activate.sh
VLLM_DIR=~/adaflash-vllm/vllm CUDA_VISIBLE_DEVICES=2 \
  bash ~/adaflash-vllm/AdaFlash/scripts/tools/reproduce_vllm.sh
```

脚本默认顺序运行三种模式。每完成一个数据集即原子保存 JSON；相同配置可断点续跑。模式锁防止重复启动。更换代码 commit、模型、参数或数据后，应选择新的 `OUTPUT_DIR`。

可覆盖参数，例如：

```bash
OUTPUT_DIR=~/adaflash-vllm/results/full-c64 \
VLLM_DIR=~/adaflash-vllm/vllm \
  bash ~/adaflash-vllm/AdaFlash/scripts/tools/reproduce_vllm.sh \
  --num-prompts 1024 --repeats 3
```

为缩短本次等待，实测同时使用三张同型号 A800：vanilla 在 GPU 2、fixed 在 GPU 0、adaptive 在 GPU 1；每个模型仍为单卡 TP1，没有模型并行。三张 GPU 同处共享服务器，数据应视为当前环境下的验证结果，不能直接外推生产硬件。使用上述一键入口可在同一张卡顺序复测。

## 产物

默认目录为 `benchmark_results/vllm-c64/`：

- `<dataset>-<mode>.json`：输入配置、模型/代码版本、数据/prompt 哈希、逐轮计时、token 数、截断数、spec decode counters、完整文本和 token IDs。
- `summary.json`：输出一致性、实际平均验证长度、接受长度、逐集合统计。
- `table.md`：可直接放入 PR 的三模式吞吐及加速比表格。
- `logs/`：各模式原始日志。

AIME25 按用户后续要求已从默认运行与最终表格移除。

数学集合的辅助分数沿用仓库的 normalized extracted-answer exact match，不能视为严格的数学等价判分；代码集合没有执行生成代码，不报告 pass@1。

草稿 PR：[vllm-project/vllm#63532](https://github.com/vllm-project/vllm/pull/63532)。本次性能实验固定在移植提交 `c01935becfdc3d9fa7d0f3f3d26ce636271f1e87`。运行期间，草稿已单独同步到 upstream main `4e591657`，头提交为 `7634e544`，冲突已解决并通过 40 项 manager/scheduler 测试、2 个真实模型测试及全部 hooks；性能测量未更换版本或混用提交。正式提审前可在该 head 上重跑同一入口。
