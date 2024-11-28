# QwQ-32B-Preview

> QwQ-32B-Preview是由Qwen团队开发的实验性研究模型，旨在提升人工智能的推理能力。 [模型链接](https://modelscope.cn/models/Qwen/QwQ-32B-Preview/summary)

使用Speed Benchmark工具测试QwQ-32B-Preview模型在不同配置下的显存占用以及推理速度。以下测试生成2048 tokens时的速度与显存占用，输入长度分别为1、6144、14336、30720：

## 本地transformers推理速度


### 测试环境

- NVIDIA A100 80GB * 1
- CUDA 12.1
- Pytorch 2.3.1
- Flash Attention 2.5.8
- Transformers 4.46.0
- EvalScope 0.7.0


### 压测命令
```shell
pip install evalscope[perf] -U
```
```shell
CUDA_VISIBLE_DEVICES=0 evalscope perf \
 --parallel 1 \
 --model Qwen/QwQ-32B-Preview \
 --attn-implementation flash_attention_2 \
 --log-every-n-query 1 \
 --connect-timeout 60000 \
 --read-timeout 60000\
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local \
 --dataset speed_benchmark
```

### 测试结果
```text
+---------------+-----------------+----------------+
| Prompt Tokens | Speed(tokens/s) | GPU Memory(GB) |
+---------------+-----------------+----------------+
|       1       |      17.92      |     61.58      |
|     6144      |      12.61      |     63.72      |
|     14336     |      9.01       |     67.31      |
|     30720     |      5.61       |     74.47      |
+---------------+-----------------+----------------+
```

## vLLM 推理速度

### 测试环境
- NVIDIA A100 80GB * 2
- CUDA 12.1
- vLLM 0.6.3
- Pytorch 2.4.0
- Flash Attention 2.6.3
- Transformers 4.46.0

### 测试命令
```shell
CUDA_VISIBLE_DEVICES=0,1 evalscope perf \
 --parallel 1 \
 --model Qwen/QwQ-32B-Preview \
 --log-every-n-query 1 \
 --connect-timeout 60000 \
 --read-timeout 60000\
 --max-tokens 2048 \
 --min-tokens 2048 \
 --api local_vllm \
 --dataset speed_benchmark
```

### 测试结果
```text
+---------------+-----------------+
| Prompt Tokens | Speed(tokens/s) |
+---------------+-----------------+
|       1       |      38.17      |
|     6144      |      36.63      |
|     14336     |      35.01      |
|     30720     |      31.68      |
+---------------+-----------------+
```