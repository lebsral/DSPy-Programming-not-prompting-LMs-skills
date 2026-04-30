# vLLM API Reference

> Condensed from [docs.vllm.ai](https://docs.vllm.ai/en/latest/). Verify against upstream for latest.

## DSPy Connection

```python
lm = dspy.LM(
    "openai/<model-name>",
    api_base="http://localhost:8000/v1",
    api_key="none",    # required but any value works
)
dspy.configure(lm=lm)
```

`dspy.HFClientVLLM` is deprecated -- always use `dspy.LM("openai/...")` with `api_base`.

## vllm serve

```bash
vllm serve <model> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port number |
| `--tensor-parallel-size` / `-tp` | `1` | Number of GPUs for tensor parallelism |
| `--max-model-len` | auto | Max sequence length (supports "1k", "2M") |
| `--gpu-memory-utilization` | `0.92` | Fraction of GPU memory to use (0-1) |
| `--dtype` | `auto` | `auto`, `float16`, `bfloat16` |
| `--quantization` / `-q` | none | `awq`, `gptq`, or none |
| `--enable-prefix-caching` | off | Cache common prompt prefixes |
| `--max-num-seqs` | `256` | Max concurrent sequences |
| `--speculative-model` | none | Draft model for speculative decoding |
| `--num-speculative-tokens` | — | Tokens to speculate per step |

## GPU Sizing

| Model Size | FP16 VRAM | INT4 VRAM | Recommended GPU |
|-----------|-----------|----------|-----------------|
| 7-8B | ~16 GB | ~5 GB | 1x RTX 4090 / A10G |
| 13-14B | ~28 GB | ~8 GB | 1x A100-40GB |
| 70B | ~140 GB | ~40 GB | 2x A100-80GB |

Add ~20% overhead for KV cache.

## Health Check

```bash
curl http://localhost:8000/health  # 200 when ready
```

## Docker

```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```
