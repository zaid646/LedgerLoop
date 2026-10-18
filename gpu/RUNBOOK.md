# GPU Runbook — self-hosted vLLM on a rented RTX 4090

ExpenseIQ does **not** call a paid model API. The LLM is served by **vLLM** on a rented
GPU (Vast.ai RTX 4090, 24 GB). You spin the instance up; I consume the OpenAI-compatible
endpoint. **Destroy the instance when the session is done.**

Target model: **`Qwen/Qwen2.5-7B-Instruct`** (fits a 4090 comfortably, strong JSON).

---

## 1. Rent the instance (Vast.ai)

- GPU: **RTX 4090**, 1× GPU, ≥ 24 GB VRAM.
- Disk: **≥ 40 GB** (model weights ≈ 15 GB + CUDA image).
- Image / template: **`vllm/vllm-openai:latest`** (has vLLM + CUDA preinstalled) — avoids a
  slow `pip install` that bills while it runs.
- Recommended extra env on the instance: `HF_HUB_ENABLE_HF_TRANSFER=1` for fast pulls.

## 2. Launch vLLM

Two equivalent options. Pick one.

**A. Docker on the instance (recommended)**
```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --api-key "$VLLM_API_KEY" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

**B. vLLM on the host**
```bash
pip install "vllm>=0.6"
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90
```

Wait for the log line `Application startup complete.` and the `Uvicorn running on
http://0.0.0.0:8000` message. First boot also downloads the weights (~2–5 min on a good pipe).

## 3. Expose the port

Vast maps your container port `8000` to a **public host + port**, e.g.
`https://<host>.vast.ai:41000` or `http://<ip>:<mapped_port>`. In the Vast UI open the
*Connect* tab and copy the mapping for port 8000.

> Prefer an **SSH tunnel** over a public port when possible:
> ```bash
> ssh -p <ssh_port> -L 8000:localhost:8000 root@<host>
> # then use VLLM_BASE_URL=http://localhost:8000/v1
> ```
> Always set `--api-key` regardless — the endpoint must never be open.

## 4. Hand it to me

Send me **three things**:
1. Base URL, e.g. `https://<host>.vast.ai:41000` (or say "tunnelled → `http://localhost:8000`")
2. The API key you set (`$VLLM_API_KEY`)
3. The model name (`Qwen/Qwen2.5-7B-Instruct`)

I'll write them into `.env` and verify with `make gpu-check`.

## 5. Verify

```bash
make gpu-check           # lists /v1/models + runs a schema-guided test completion
```

Expected: `[ok] /v1/models -> Qwen/Qwen2.5-7B-Instruct` and `[ok] guided JSON completion`.

## 6. Cost tracking

Record the hourly rate in `runs/gpu_sessions.csv` (auto-appended by `check_gpu.py`) so W&B
can compute **cost-per-document** = `rate_usd_per_hr ÷ (documents_processed / hours_run)`.

## 7. Tear down

**Destroy the instance** (not just stop) when finished — stopped instances still bill for
storage. Sessions are cheap: a 4090 runs ~$0.22–0.40/hr, so a few eval hours cost ~$1–2.

---

### Optional: serving a QLoRA adapter (Phase 6)
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-lora --lora-modules expenseiq-gl=./adapters/expenseiq-gl \
  --api-key "$VLLM_API_KEY" --max-model-len 8192
```
Then point the coding step at the `expenseiq-gl` adapter via `VLLM_MODEL`.
