# vLLM: Original (`origin/main`) vs Current (local)

Differences in the **vLLM fork** (`/home/ubuntu/Ali_p/vllm_new/vllm`) required for **DeepSeek-V4-Flash online hidden-state extraction** during speculators `dspark-hg` training.

**Diff summary:** 3 files changed, 100 insertions(+), 52 deletions(-)

Speculators-side training code is documented separately in `/home/ubuntu/Ali_p/vllm_new/speculators/ORIGINAL_VS_CURRENT.md`.

---

## Quick comparison

| Area | Original (`origin/main`) | Current (local) |
|------|--------------------------|-----------------|
| DeepSeek-V4 + HS extraction | HS layers merged into MLA KV groups → connector fails | HS layers in **isolated** KV groups |
| Aborted HS requests | `KeyError` on missing filename | Safe `pop(req_id, None)` |
| `libcudart` loading | Prefer already-loaded library, then env | Prefer **`VLLM_CUDART_SO_PATH`** first |

---

## Why these changes exist

Online training calls vLLM with `extract_hidden_states` to write verifier hidden states to `/tmp/hidden_states/`. For **DeepSeek-V4-Flash**:

1. Target layers use **MLA** KV cache layout.
2. Hidden-state layers use `HiddenStateCacheSpec`, which **subclasses** MLA spec.
3. Original grouping merged HS layers into MLA groups → `ExampleHiddenStatesConnector` could not find a unique HS group (`"MLA verifiers are unsupported"`).
4. Under load, requests could abort before the connector registered filenames → crash.
5. tilelang could load a **stub** `libcudart` missing `cudaDeviceReset` → worker shutdown errors.

---

## Modified files (3)

### 1. `vllm/v1/core/kv_cache_utils.py` (+85 / −52 lines)

Largest change. Enables DeepSeek-V4 + hidden-state extraction together.

#### New helpers

**`_append_hidden_state_cache_groups(groups, hidden_specs)`**

- Appends each `HiddenStateCacheSpec` layer as its **own isolated** `KVCacheGroupSpec`.
- HS groups must not share MLA verifier groups so the connector can identify them uniquely.

**`_is_deepseek_v4_kv_groups(kv_cache_groups)`**

- Returns true for:
  - all groups are `UniformTypeKVCacheSpecs` (pure DSV4), **or**
  - mix of `UniformTypeKVCacheSpecs` + `HiddenStateCacheSpec` only.

#### `get_kv_cache_groups()` — reordered HS handling

**Original:**

- Pulled `HiddenStateCacheSpec` out only on the **general multi-group** path (end of function).
- DeepSeek-V4 branch returned early **without** isolated HS groups.
- HS layers re-added with page-size alignment to common page.

**Current:**

- Peel off `hidden_specs` **at the start**, before any hybrid / DSV4 MLA grouping.
- Every return path calls `_append_hidden_state_cache_groups(groups, hidden_specs)`.
- Removed late realignment block that padded HS page size to common MLA page.

**Effect:** DSV4 MLA groups unchanged; HS layers sit in separate groups the connector can target.

#### `_use_packed_kv_cache_config()`

**Original:** `is_dsv4 = all(UniformTypeKVCacheSpecs)`.

**Current:** `is_dsv4 = _is_deepseek_v4_kv_groups(...)` — true when HS groups are present alongside DSV4.

#### `_max_memory_usage_bytes_from_groups()`

**Original:** All groups assumed uniform DSV4 MLA; one memory formula.

**Current:**

- Split into `uniform_groups` (MLA) and `hidden_groups` (`HiddenStateCacheSpec`).
- MLA memory computed as before on uniform groups only.
- HS group memory added via `max_memory_usage_bytes()` per hidden group.

---

### 2. `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py` (+6 / −2 lines)

**Location:** Method that finishes a request and writes hidden states from KV cache.

**Original:**

```python
filename = self._request_filenames.pop(req_id)
```

**Current:**

```python
filename = self._request_filenames.pop(req_id, None)
if filename is None:
    return False, None
```

**Why:** Requests aborted or never scheduled (e.g. under `max-num-seqs` pressure) can finish before `build_connector_meta` registers them. Original code raised `KeyError`.

---

### 3. `vllm/distributed/device_communicators/cuda_wrapper.py` (+4 / −5 lines)

**Location:** `CudaRTLibrary.__init__` when resolving `libcudart` shared library.

**Original:**

```python
so_file = (
    find_loaded_library("libcudart")
    or envs.VLLM_CUDART_SO_PATH
)
```

**Current:**

```python
so_file = envs.VLLM_CUDART_SO_PATH or find_loaded_library("libcudart")
```

**Why:** tilelang may load `libcudart_stub.so` first (missing `cudaDeviceReset`). Prefer explicit real CUDA runtime from env.

**Used by:** `examples/train/server_run.sh` in speculators:

```bash
export VLLM_CUDART_SO_PATH="${VLLM_CUDART_SO_PATH:-/usr/local/cuda/lib64/libcudart.so}"
```

---

## How this connects to training

```
speculators train.py  --on-missing generate
        │
        ▼ HTTP completion (max_tokens=1)
vLLM server (DeepSeek-V4-Flash)
        │
        ├─ kv_cache_utils: isolated HiddenStateCacheSpec groups
        ├─ example_hidden_states_connector: write /tmp/hidden_states/*.safetensors
        └─ cuda_wrapper: stable libcudart on shutdown
        │
        ▼
speculators ArrowDataset loads HS → trains dspark-hg drafter
```

---

## Server launch (reference)

From `speculators/examples/train/server_run.sh` (not a vLLM repo file, but documents how this fork is run):

| Setting | Value |
|---------|-------|
| Model | `deepseek-ai/DeepSeek-V4-Flash` |
| GPUs | `1,2,3,4`, TP=4 |
| `TARGET_LAYER_IDS` | `41` |
| `max_model_len` | 256 |
| Quantization | `deepseek_v4_fp8` |
| KV cache | `fp8_ds_mla` |
| Expert parallel | enabled |

**Note:** `scripts/launch_vllm.py` defaults `--include-last-layer` to **True**, which may append layer `43` to target layers. Align with training `TARGET_LAYER_IDS` or pass `--no-include-last-layer`.

---

## Unchanged in this fork (for this effort)

- `vllm/models/deepseek_v4/nvidia/dspark.py` — inference drafter (no local diff vs origin/main in this changeset)
- Core DeepSeek-V4 model forward
- Standard non-HS serving paths for other models

---

## Modified file list

```
vllm/v1/core/kv_cache_utils.py
vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py
vllm/distributed/device_communicators/cuda_wrapper.py
```

---

*These vLLM changes are required companions to the speculators `dspark-hg` training work.*
