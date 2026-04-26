Here is the handoff context to paste at the start of the next session:

---

## Session Handoff — HMD-Poser Ablation Experiments

### Goal

Run three ablation experiments to empirically test two architectural claims in the HMD-Poser paper (CVPR 2024) that were only argued qualitatively:

1. **LSTM vs Transformer as temporal module** — is the unidirectional LSTM actually better than a causal Transformer?
2. **Module necessity** — do both the temporal (LSTM) and spatial (Transformer) modules individually contribute?

Three trained model variants are needed. Evaluate all with sliding-window inference on Protocol 1 AMASS, **HMD-only** scenario. Save per-sequence metrics for statistical analysis.

---

### Architecture notes (confirmed)

- **LSTM is unidirectional** (`bidirectional=False`, confirmed at `model/network.py:50` and `:360`) — so in-training validation with full sequences is causal, not offline.
- The `CausalTransformerTemporalSpatialBackbone` uses causal masking — also online/causal.
- `pred_shapes` is zeroed before returning in `hmd_imu_model.py` — shape head has no gradient (pre-existing issue, safe to ignore).

| `model_name` | Temporal | Spatial | Params |
|---|---|---|---|
| `HMD_imu_HME_Universe` | LSTM (unidirectional) | Transformer | 17.4M |
| `HMD_imu_HME_TemporalTransformer` | Causal Transformer | Transformer | 15.3M |
| `HMD_imu_HME_SpatialOnly` | None | Transformer | 9.0M |
| `HMD_imu_HME_TemporalOnly` | LSTM (unidirectional) | None | — |

---

### Key decisions (finalised)

**`compatible_inputs: ['HMD']` for training AND testing** (all 6 ablation YAMLs set).

**Checkpoint for evaluation: `checkpoint_latest.pt` (epoch 400)**
- Fixed training budget is the fairest ablation comparison.
- `run_ablation.py` now correctly uses `checkpoint_latest.pt` everywhere.
- Completion detection reads the `epoch` field from the `.pt` file (0-indexed); training is
  considered done when `stored_epoch == epochs - 1` (e.g., 399 for 400 epochs).

**Periodic snapshots every 20 epochs** saved to `<exp_dir>/snapshots/checkpoint_ep{N:04d}.pt`
- Weights-only (~59 MB each). 20 snapshots × 3 variants ≈ 3.5 GB total.
- Configured via `snapshot_interval: 20` in `train_config` block of each YAML.
- Epoch stored inside file is 0-indexed (0..399); filename label is 1-indexed (ep0020 = epoch 20).

---

### Timing measurements (RTX 4090, HMD-only, temporal_transformer)

| Phase | Time |
|---|---|
| Data loading + model init | ~82 s (one-time) |
| **Training epoch** (batch=128, 375 batches) | **89 s** |
| In-training validation (536 seq, HMD-only, batch=1) | **23 s** |
| **Sliding window eval** (536 seq, stride=1) | **321 s ≈ 5.3 min** |

**Full ablation estimate (3 variants × 400 epochs):**

| Component | Per run | × count | Total |
|---|---|---|---|
| Training epoch | 89 s | × 400 × 3 | 29.7 h |
| In-training validation | 23 s | × 40 × 3 | 0.8 h |
| Final sliding window eval | 321 s | × 3 variants | 0.3 h |
| **Grand total** | | | **~30.8 h** |

---

### Automated pipeline

`scripts/run_ablation.py` orchestrates the full workflow. State persists in `scripts/ablation_state.json`.

```bash
python scripts/run_ablation.py              # run / resume
python scripts/run_ablation.py --status     # inspect progress
python scripts/run_ablation.py --reset      # delete state, start fresh
```

**Current state:** all variants pending (state was reset 2026-04-12, no training has run yet).

---

### What to run

**Launch:**
```bash
python scripts/run_ablation.py
```

**Monitor (epoch 2 should appear ~112 s after epoch 1 log line):**
- "Training epoch: 1/400" → train ~89 s → val ~23 s → "Training epoch: 2/400"
- If epoch 2 doesn't appear within 3 min of epoch 1: process crashed (WSL2/CUDA silent death)

**After all training, run_ablation.py handles evaluation automatically.**
Manual eval commands if needed:
```bash
python test_sliding_window.py --config options/ablation/test_temporal_transformer.yaml
python test_sliding_window.py --config options/ablation/test_spatial_only.yaml
python test_sliding_window.py --config options/ablation/test_temporal_only.yaml
```
Each takes ~5.3 min. Results land in `results/ablation/<variant>/`.

**Generate summary:**
```bash
python scripts/run_ablation.py --summary-only
```

---

### Known issues / notes

- **Silent crash pattern**: training dies within seconds of epoch 1 start with no traceback (WSL2 + CUDA). GPU shows "No running processes" after crash. Root cause unknown. No fix found; retrying usually works. Auto-retry is built into the pipeline.
- Two pre-existing deprecation warnings: `torch.cross` (utils_transform.py:50) and `weight_norm` (PyTorch internals). Safe to ignore.
- The Python interpreter is `.venv/bin/python` (uv venv, Python 3.13). **Not** the old conda path.
- All commands should be run from `/home/yy/Code/HMD-Poser/`.

---

### Pipeline internals (for debugging)

**YAML operations** (`scripts/run_ablation.py`):
- Reads: `yaml.safe_load` (PyYAML) — handles nested fields correctly
- Writes: regex substitution — preserves comments in config files (`yaml.dump` would strip them)

**Completion detection** (`run_training()` early-exit):
- Reads `epoch` field (0-indexed) from `checkpoint_latest.pt` via `torch.load`
- Training complete if `stored_epoch == target_epochs - 1` (e.g., 399 == 399)
- Falls back to full re-train if checkpoint is missing or epoch < target

**Checkpoint path flow:**
1. Training completes → `checkpoint_latest.pt` written to experiment dir
2. `run_ablation.py` stores path in `ablation_state.json`
3. `patch_test_config_after_training()` writes path into `resume_model` in test YAML
4. `test_sliding_window.py` reads `configs.resume_model`

**Resume training flow:**
- On crash: `checkpoint_latest.pt` exists but epoch < target
- `run_ablation.py` patches `resume_checkpoint` in train YAML
- `train.py` reads `resume_checkpoint` via `getattr(configs, 'resume_checkpoint', None)`
- Restores weights, optimizer, LR scheduler, epoch counter, best metrics
