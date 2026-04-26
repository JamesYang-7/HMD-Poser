#!/usr/bin/env python3
"""
scripts/preflight.py — Pre-flight validation for all HMD-Poser configs.

Phase 1 — Config audit:
  Loads every YAML config via the same load_config() path used by train/test scripts.
  Prints the final resolved value of every key parameter. Checks:
    - model_name is registered in model_wraps
    - dataset_path and support_dir exist on disk
    - resume_model file exists on disk (test configs only)

Phase 2 — Synthetic dry run (train configs only):
  Instantiates model + loss + optimizer exactly as train.py does.
  Runs 1 forward pass + loss computation + backward pass + optimizer step
  using synthetic random data (no AMASS loading required).
  Checks:
    - All output shapes are correct
    - All tensors (loss components, predictions) are finite
    - total_loss > 0
    - Gradient norms are finite and non-zero

Usage (from the repo root):
  python scripts/preflight.py                       # all configs under options/
  python scripts/preflight.py --configs options/ablation/train_temporal_only.yaml
  python scripts/preflight.py --no-dry-run          # phase 1 only
"""

import sys
import os
import glob
import argparse
import traceback
from pathlib import Path

# Run from repo root so relative paths in configs resolve correctly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch

from utils.utils_config import load_config, LearningRateLambda
from model.hmd_imu_model import HMDIMUModel, model_wraps
from model.loss import PoseJointLoss

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 72
W    = 24   # label column width


def _exists(path):
    if path and str(path) not in ("null", "None", ""):
        return "[EXISTS]" if os.path.exists(str(path)) else "[MISSING]"
    return ""


def _file_exists(path):
    if path and str(path) not in ("null", "None", ""):
        return "[EXISTS]" if os.path.isfile(str(path)) else "[FILE NOT FOUND]"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: config audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_config(path):
    """Load and print all key parameters of one config. Returns (cfg, ok)."""
    print(f"\n{SEP}")
    print(f"  {path}")
    print(SEP)

    try:
        cfg = load_config(path)
    except Exception as exc:
        print(f"  ERROR: could not load config — {exc}")
        return None, False

    is_test = os.path.basename(path).startswith("test")
    ok = True

    def row(label, value, note=""):
        print(f"  {label:{W}} {value}  {note}".rstrip())

    # ── model ──────────────────────────────────────────────────────────────
    registered = cfg.model_name in model_wraps
    if not registered:
        ok = False
    row("model_name", cfg.model_name,
        "[REGISTERED]" if registered else f"[NOT REGISTERED — known: {list(model_wraps)}]")

    # ── paths ──────────────────────────────────────────────────────────────
    row("dataset_path", cfg.dataset_path, _exists(cfg.dataset_path))
    row("support_dir",  cfg.support_dir,  _exists(cfg.support_dir))

    rm = getattr(cfg, "resume_model", None)
    if is_test:
        note = _file_exists(rm) if rm and str(rm) not in ("null", "None") else "[NOT SET]"
        if "[NOT SET]" in note:
            ok = False   # test config must have a checkpoint
        row("resume_model", rm, note)
    else:
        row("resume_model", rm)   # null is fine for train configs

    # ── I/O ───────────────────────────────────────────────────────────────
    row("compatible_inputs",  cfg.compatible_inputs)
    row("sparse_dim",         cfg.sparse_dim)
    row("input_motion_length",cfg.input_motion_length)

    if is_test:
        row("save_dir",    getattr(cfg, "save_dir", "N/A"))
        row("save_output", getattr(cfg, "save_output", "N/A"))

    # ── model_params ───────────────────────────────────────────────────────
    print(f"  {'model_params':{W}}")
    mp = cfg.model_params
    for k in ("number_layer", "hidden_size", "dropout", "nhead", "block_num"):
        row(f"    {k}", getattr(mp, k))

    # ── train-only fields ──────────────────────────────────────────────────
    if not is_test:
        print(f"  {'train_config':{W}}")
        tc = cfg.train_config
        row("    epochs",       tc.epochs)
        row("    batch_size",   tc.batch_size)
        row("    val_interval", tc.val_interval)
        row("    num_workers",  tc.num_workers)

        print(f"  {'optimizer':{W}}")
        oc, lrc = cfg.optimizer_config, cfg.lr_config
        row("    type",         oc.optimizer)
        row("    lr",           lrc.lr)
        row("    (beta1,beta2)",f"({oc.momentum}, {oc.beta2})")
        row("    weight_decay", oc.weight_decay)
        row("    amsgrad",      oc.amsgrad)

        print(f"  {'lr_schedule':{W}}")
        row("    decay_at_epochs",  lrc.lr_decay)
        row("    decay_factor",     lrc.lr_decay_factor)
        row("    decay_duration",   lrc.lr_decay_epochs)
        row("    warmup_start",     lrc.lr_warm_up_start_epoch)
        row("    warmup_epochs",    lrc.lr_warm_up_epochs)
        row("    warmup_factor",    lrc.lr_warm_up_factor)

    # ── loss weights ───────────────────────────────────────────────────────
    print(f"  {'loss':{W}}")
    lc = cfg.loss
    row("    type", lc.loss_type)
    weights = {
        "root_orientation": lc.root_orientation_weight,
        "local_pose":       lc.local_pose_weight,
        "global_pose":      lc.global_pose_weight,
        "joint_position":   lc.joint_position_weight,
        "acceleration":     lc.smooth_loss_weight,
        "shape":            lc.shape_loss_weight,
    }
    max_w = max(weights.values())
    for name, w in weights.items():
        marker = "  ← highest" if w == max_w else ""
        row(f"    {name}", w, marker)

    return cfg, ok


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: synthetic dry run
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(path, cfg):
    """
    1-iteration dry run for a train config.
    Uses synthetic data — no AMASS loading needed.
    Returns True if all checks pass.
    """
    print(f"\n  {SEP2}")
    print(f"  DRY RUN: {os.path.basename(path)}")
    print(f"  {SEP2}")

    device = torch.device("cpu")
    B     = 4                          # small batch for speed
    T     = cfg.input_motion_length    # 40
    D     = cfg.sparse_dim             # 135
    n_mod = len(cfg.compatible_inputs) # 3 during training
    B_eff = B * n_mod

    print(f"  batch: {B} samples × {n_mod} modalities = {B_eff} effective | T={T} | D={D}")

    # ── model ──────────────────────────────────────────────────────────────
    print(f"  [1/4] model init ...", end=" ", flush=True)
    try:
        model = HMDIMUModel(cfg, device)
        model.train()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"OK  ({cfg.model_name}, {n_params:,} params)")
    except Exception as exc:
        print(f"FAIL\n       {exc}")
        traceback.print_exc()
        return False

    # ── loss + optimizer ───────────────────────────────────────────────────
    loss_func = PoseJointLoss(
        cfg.loss.loss_type,
        cfg.loss.root_orientation_weight,
        cfg.loss.local_pose_weight,
        cfg.loss.global_pose_weight,
        cfg.loss.joint_position_weight,
        cfg.loss.smooth_loss_weight,
        cfg.loss.shape_loss_weight,
    )
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr           = cfg.lr_config.lr,
        betas        = (cfg.optimizer_config.momentum, cfg.optimizer_config.beta2),
        weight_decay = cfg.optimizer_config.weight_decay,
        eps          = cfg.optimizer_config.adam_eps,
        amsgrad      = cfg.optimizer_config.amsgrad,
    )

    # ── synthetic batch (mirrors training_loop.py after the modality flatten) ──
    torch.manual_seed(0)
    input_feat     = torch.randn(B_eff, T, D)
    gt_local_pose  = torch.randn(B_eff, T, 22 * 6)
    gt_global_pose = torch.randn(B_eff, T, 22 * 6)
    gt_positions   = torch.randn(B_eff, T, 22, 3)
    gt_betas       = torch.randn(B_eff, T, 16)

    # ── forward pass ───────────────────────────────────────────────────────
    print(f"  [2/4] forward pass ...", end=" ", flush=True)
    try:
        optimizer.zero_grad()
        pred_local_pose, pred_betas, rotation_global_r6d, pred_joint_position = model(input_feat)

        # Shape assertions
        assert pred_local_pose.shape     == (B_eff, T, 132),    f"pose: {pred_local_pose.shape}"
        assert pred_betas.shape          == (B_eff, T, 16),     f"betas: {pred_betas.shape}"
        assert rotation_global_r6d.shape == (B_eff, T, 132),   f"global_rot: {rotation_global_r6d.shape}"
        assert pred_joint_position.shape == (B_eff, T, 22, 3), f"joint_pos: {pred_joint_position.shape}"

        # Finite check
        for name, t in [("pred_local_pose", pred_local_pose),
                         ("rotation_global_r6d", rotation_global_r6d),
                         ("pred_joint_position", pred_joint_position)]:
            assert torch.isfinite(t).all(), f"{name} contains non-finite values"

        # Head-center (identical to training_loop.py:38)
        pred_joint_position_hc = (pred_joint_position
                                  - pred_joint_position[:, :, 15:16]
                                  + gt_positions[:, :, 15:16])

        print(f"OK  "
              f"pose={tuple(pred_local_pose.shape)}  "
              f"joint_pos={tuple(pred_joint_position.shape)}")
    except Exception as exc:
        print(f"FAIL\n       {exc}")
        traceback.print_exc()
        return False

    # ── loss ───────────────────────────────────────────────────────────────
    print(f"  [3/4] loss ...", end=" ", flush=True)
    try:
        (root_loss, local_loss, global_loss,
         pos_loss, accel_loss, shape_loss, total_loss) = loss_func(
            pred_local_pose[:, :, :6],   # root orientation 6D
            pred_local_pose[:, :, 6:],   # body joints 6D
            rotation_global_r6d,
            pred_joint_position_hc,
            pred_betas,
            gt_local_pose[:, :, :6],
            gt_local_pose[:, :, 6:],
            gt_global_pose,
            gt_positions,
            gt_betas,
        )

        components = {
            "total":  total_loss,
            "root":   root_loss,
            "local":  local_loss,
            "global": global_loss,
            "pos":    pos_loss,
            "accel":  accel_loss,
            "shape":  shape_loss,
        }
        for name, val in components.items():
            assert torch.isfinite(val), f"{name}_loss is not finite: {val}"

        assert total_loss.item() > 0, f"total_loss is not positive: {total_loss.item()}"

        parts = "  ".join(f"{k}={v.item():.4f}" for k, v in components.items())
        print(f"OK  {parts}")
    except Exception as exc:
        print(f"FAIL\n       {exc}")
        traceback.print_exc()
        return False

    # ── backward + optimizer step ──────────────────────────────────────────
    print(f"  [4/4] backward + optimizer step ...", end=" ", flush=True)
    try:
        total_loss.backward()
        optimizer.step()

        # Collect grad norms from the network (not body model)
        grad_norms = [
            p.grad.norm().item()
            for p in model.netG.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0,    "no gradients found in netG"
        assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms), \
               "non-finite gradients detected"
        assert max(grad_norms) > 0,    "all gradients are exactly zero"

        print(f"OK  "
              f"grad_norms(netG): max={max(grad_norms):.4f}  "
              f"mean={sum(grad_norms)/len(grad_norms):.4f}  "
              f"n={len(grad_norms)}")
    except Exception as exc:
        print(f"FAIL\n       {exc}")
        traceback.print_exc()
        return False

    print(f"  RESULT: PASS")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def discover_configs():
    return sorted(
        glob.glob("options/train_config.yaml")
        + glob.glob("options/test_config.yaml")
        + glob.glob("options/ablation/train_*.yaml")
        + glob.glob("options/ablation/test_*.yaml")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight validation for all HMD-Poser configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--configs", nargs="*",
        help="Specific config YAML files to check (default: all under options/)",
    )
    parser.add_argument(
        "--no-dry-run", action="store_true",
        help="Phase 1 (config audit) only — skip synthetic dry runs",
    )
    args = parser.parse_args()

    all_configs = args.configs if args.configs else discover_configs()
    train_cfgs  = [c for c in all_configs if os.path.basename(c).startswith("train")]
    test_cfgs   = [c for c in all_configs if os.path.basename(c).startswith("test")]

    print(f"\n{'#'*72}")
    print(f"  HMD-Poser Pre-flight Check")
    print(f"  {len(train_cfgs)} train config(s), {len(test_cfgs)} test config(s)")
    print(f"{'#'*72}")

    # ── Phase 1: audit ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  PHASE 1 — CONFIG AUDIT")
    print(SEP)

    audit_results = {}   # path -> (cfg | None, ok: bool)
    for path in all_configs:
        cfg, ok = audit_config(path)
        audit_results[path] = (cfg, ok)

    # ── Phase 2: dry run ───────────────────────────────────────────────────
    dry_results = {}
    if not args.no_dry_run:
        print(f"\n{SEP}")
        print(f"  PHASE 2 — SYNTHETIC DRY RUNS  (device=cpu, batch=4, T=40)")
        print(SEP)

        for path in train_cfgs:
            cfg, ok = audit_results.get(path, (None, False))
            if cfg is None:
                dry_results[path] = False
                print(f"\n  SKIPPED {path}  (config failed to load)")
                continue
            if not ok:
                dry_results[path] = False
                print(f"\n  SKIPPED {path}  (config has errors — fix phase 1 first)")
                continue
            dry_results[path] = dry_run(path, cfg)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SUMMARY")
    print(SEP)

    all_pass = True

    print(f"\n  Config audit:")
    for path, (cfg, ok) in audit_results.items():
        tag = "OK  " if ok else "FAIL"
        model = cfg.model_name if cfg else "?"
        print(f"    [{tag}]  {path:<55}  ({model})")
        if not ok:
            all_pass = False

    if not args.no_dry_run:
        print(f"\n  Dry runs (train configs):")
        for path, passed in dry_results.items():
            tag = "PASS" if passed else "FAIL"
            print(f"    [{tag}]  {path}")
            if not passed:
                all_pass = False

        skipped = [p for p in train_cfgs if p not in dry_results]
        for path in skipped:
            print(f"    [SKIP]  {path}")

    verdict = "ALL CHECKS PASSED — safe to launch full runs." if all_pass \
              else "SOME CHECKS FAILED — review the output above before running."
    print(f"\n  Overall: {verdict}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
