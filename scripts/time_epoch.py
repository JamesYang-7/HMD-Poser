#!/usr/bin/env python3
"""
scripts/time_epoch.py — Estimate per-epoch training time for all 4 network variants.

Runs each variant for up to TIME_LIMIT_SECS seconds, counting batches,
then extrapolates to a full epoch (375 batches at batch_size=128).

Usage:
    python scripts/time_epoch.py
"""

import sys
import time
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIME_LIMIT_SECS = 300   # 5 minutes per variant
BATCHES_PER_EPOCH = 375
BATCH_SIZE = 128
SEQ_LEN = 40
INPUT_DIM = 135

VARIANTS = [
    ("HMD_imu_HME_Universe",           "Original (LSTM + Transformer)"),
    ("HMD_imu_HME_SpatialOnly",        "Spatial-Only (no LSTM)"),
    ("HMD_imu_HME_TemporalOnly",       "Temporal-Only LSTM (no Transformer)"),
    ("HMD_imu_HME_TemporalTransformer","CausalTransformer Temporal + Transformer Spatial"),
]

MODEL_PARAMS = dict(number_layer=3, hidden_size=256, dropout=0.05, nhead=8, block_num=2)


def build_model(model_name, device):
    from model import network as net
    cls = getattr(net, model_name)
    model = cls(INPUT_DIM, **MODEL_PARAMS).to(device)
    return model


def dummy_batch(device):
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM, device=device)
    return x


def time_variant(model_name, label, device):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  ({model_name})")
    print(f"{'='*70}")

    model = build_model(model_name, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    batch_times = []
    n_batches = 0
    deadline = time.time() + TIME_LIMIT_SECS

    print(f"  Timing (up to {TIME_LIMIT_SECS}s) ...", flush=True)
    warmup = 3  # skip first N batches (CUDA warm-up)

    while time.time() < deadline:
        x = dummy_batch(device)
        # Minimal forward + scalar loss + backward (no SMPL body model)
        t0 = time.time()
        optimizer.zero_grad()
        # All model outputs are returned; use first output for loss
        out = model(x)
        if isinstance(out, (list, tuple)):
            primary = out[0]
        else:
            primary = out
        loss = primary.sum()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        n_batches += 1
        if n_batches > warmup:
            batch_times.append(elapsed)

        if n_batches % 20 == 0:
            avg_so_far = sum(batch_times) / len(batch_times) if batch_times else 0
            print(f"    batch {n_batches:4d}  avg={avg_so_far*1000:.1f}ms/batch", flush=True)

    if not batch_times:
        print("  ERROR: no batches completed after warm-up!")
        return None

    avg_batch_s = sum(batch_times) / len(batch_times)
    epoch_s = avg_batch_s * BATCHES_PER_EPOCH
    epoch_min = epoch_s / 60
    total_h = epoch_s * 400 / 3600

    print(f"\n  Batches timed (excl. warm-up): {len(batch_times)}")
    print(f"  Avg time/batch : {avg_batch_s*1000:.1f} ms")
    print(f"  Est. epoch time: {epoch_s:.0f}s  ({epoch_min:.1f} min)")
    print(f"  Est. total (400 epochs): {total_h:.1f} h")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return dict(
        model_name=model_name,
        label=label,
        n_params=n_params,
        avg_batch_ms=avg_batch_s * 1000,
        epoch_min=epoch_min,
        total_h=total_h,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    for model_name, label in VARIANTS:
        r = time_variant(model_name, label, device)
        if r:
            results.append(r)

    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Variant':<45}  {'Params':>7}  {'ms/batch':>9}  {'min/epoch':>10}  {'h (400ep)':>10}")
    print(f"  {'-'*45}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"  {r['label']:<45}  {r['n_params']/1e6:>6.1f}M  {r['avg_batch_ms']:>9.1f}  "
              f"{r['epoch_min']:>10.1f}  {r['total_h']:>10.1f}")

    print(f"\nNote: these times use a dummy batch (no data loading, no SMPL body model).")
    print("Real training will be somewhat slower due to data loading + body-model FK.")


if __name__ == "__main__":
    main()
