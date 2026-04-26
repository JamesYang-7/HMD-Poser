#!/usr/bin/env python3
"""
scripts/run_ablation.py — Automated HMD-Poser ablation study workflow.

Orchestrates all steps end-to-end, resuming where it left off on restart:
  0. Preflight dry-run (train configs only)
  1-3. Train each of the 3 ablation variants sequentially
  4. Patch each test config with the discovered checkpoint path
  5. Preflight --no-dry-run (verify all test configs now have valid paths)
  6-8. Sliding-window evaluation of each variant
  9. Collect per-sequence .pt results and write results/ablation/summary.md

State is persisted to scripts/ablation_state.json between runs.
Human action is only requested for:
  - Missing data / dependencies that cannot be fixed automatically
  - Parameter changes that could affect results (e.g., batch-size after OOM)
  - The same error persisting after 3 automated fix attempts

Usage:
    python scripts/run_ablation.py                         # run / resume
    python scripts/run_ablation.py --status                # print state, exit
    python scripts/run_ablation.py --summary-only          # regenerate summary
    python scripts/run_ablation.py --variant spatial_only  # single variant only
    python scripts/run_ablation.py --reset                 # delete state, start fresh
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "scripts" / "ablation_state.json"
LOG_DIR = ROOT / "scripts" / "logs"
PYTHON = sys.executable  # use whichever Python is running this script
EXP_ROOT = ROOT / "experiments"
MAX_AUTO_RETRIES = 3

# Each variant's `task` key must match the `task` field in its train YAML —
# that value becomes the experiment directory prefix (`experiments/<task>_<timestamp>/`).
VARIANTS = [
    {
        "name": "temporal_transformer",
        "task": "hmd_poser_temporal_transformer",
        "label": "CausalTransformer Temporal + Transformer Spatial",
        "train_config": "options/ablation/train_temporal_transformer.yaml",
        "test_config": "options/ablation/test_temporal_transformer.yaml",
        "save_dir": "results/ablation/temporal_transformer",
    },
    {
        "name": "spatial_only",
        "task": "hmd_poser_spatial_only",
        "label": "Spatial-Only (no Temporal LSTM)",
        "train_config": "options/ablation/train_spatial_only.yaml",
        "test_config": "options/ablation/test_spatial_only.yaml",
        "save_dir": "results/ablation/spatial_only",
    },
    {
        "name": "temporal_only",
        "task": "hmd_poser_temporal_only",
        "label": "Temporal-Only LSTM (no Spatial Transformer)",
        "train_config": "options/ablation/train_temporal_only.yaml",
        "test_config": "options/ablation/test_temporal_only.yaml",
        "save_dir": "results/ablation/temporal_only",
    },
    {
        "name": "original",
        "task": "hmd_poser_original",
        "label": "Original (LSTM Temporal + Transformer Spatial)",
        "train_config": "options/ablation/train_original.yaml",
        "test_config": "options/ablation/test_original.yaml",
        "save_dir": "results/ablation/original",
    },
    {
        "name": "no_temporal_mix",
        "task": "hmd_poser_no_temporal_mix",
        "label": "No Temporal Mix (per-frame MLP, capacity-matched) + Transformer",
        "train_config": "options/ablation/train_no_temporal_mix.yaml",
        "test_config": "options/ablation/test_no_temporal_mix.yaml",
        "save_dir": "results/ablation/no_temporal_mix",
    },
    {
        "name": "no_spatial_mix",
        "task": "hmd_poser_no_spatial_mix",
        "label": "LSTM + No Spatial Mix (per-sensor MLP, capacity-matched)",
        "train_config": "options/ablation/train_no_spatial_mix.yaml",
        "test_config": "options/ablation/test_no_spatial_mix.yaml",
        "save_dir": "results/ablation/no_spatial_mix",
    },
]

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

METRICS = ["mpjre", "mpjpe", "mpjve", "handpe", "upperpe", "lowerpe", "rootpe", "pred_jitter", "gt_jitter"]
METRICS_UNITS = {
    "mpjre": "°",     "mpjpe": "cm",    "mpjve": "cm/s",
    "handpe": "cm",   "upperpe": "cm",  "lowerpe": "cm",
    "rootpe": "cm",   "pred_jitter": "m/s³", "gt_jitter": "m/s³",
}

# ── State management ──────────────────────────────────────────────────────────

def _default_state():
    return {
        "schema_version": 2,
        "created_at": _ts(),
        "preflight_done": False,
        "variants": {
            v["name"]: {
                "status": "pending",   # pending|training|trained|evaluating|done|failed
                "train_start_time": None,
                "checkpoint_dir": None,
                "checkpoint_path": None,
                "train_retries": 0,
                "eval_retries": 0,
                "last_error": None,
                "error_history": [],
                "eval_table": None,    # captured PrettyTable string
                "eval_results": None,  # dict of metric → mean value
            }
            for v in VARIANTS
        },
        "summary_generated": False,
    }


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return _default_state()


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Logging ───────────────────────────────────────────────────────────────────

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, level="INFO"):
    print(f"[{_ts()}] [{level:5s}] {msg}", flush=True)


def section(title):
    bar = "=" * 72
    log(bar)
    log(title)
    log(bar)


def user_action_needed(msg):
    """Print a clear message requiring human intervention, then exit non-zero."""
    print("\n" + "=" * 72, flush=True)
    print("  ACTION REQUIRED — automated workflow paused", flush=True)
    print("=" * 72, flush=True)
    print(msg, flush=True)
    print("\nAfter fixing, re-run:  python scripts/run_ablation.py", flush=True)
    print("State is preserved in:", STATE_FILE, flush=True)
    print("=" * 72 + "\n", flush=True)
    sys.exit(1)


# ── Command runner ────────────────────────────────────────────────────────────

def run_command(cmd, log_path):
    """
    Run a subprocess, streaming combined stdout+stderr to both console and a log file.
    Returns (returncode, full_output_str).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"CMD: {' '.join(cmd)}")
    log(f"LOG: {log_path}")

    all_lines = []
    with open(log_path, "w", buffering=1) as lf:
        lf.write(f"Command: {' '.join(cmd)}\nStarted: {_ts()}\n{'='*72}\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            all_lines.append(line)
        proc.wait()
        footer = f"\n\n{'='*72}\nFinished: {_ts()}  rc={proc.returncode}\n"
        lf.write(footer)

    combined = "".join(all_lines)
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    log(f"Result: {status}")
    return proc.returncode, combined


# ── Error classification ──────────────────────────────────────────────────────

# Each entry: (regex_pattern, human_cause, auto_fixable, fix_hint)
_ERROR_PATTERNS = [
    (
        r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError",
        "GPU out of memory — batch_size likely too large",
        True,
        "Reduce batch_size by half. NOTE: this slightly changes training dynamics "
        "(gradient noise). The script will do this automatically.",
    ),
    (
        r"No such file or directory.*datasets_processed|FileNotFoundError.*datasets_processed|"
        r"dataset_path.*not found|cannot open.*datasets_processed",
        "Preprocessed dataset missing (datasets_processed/ not found or wrong path)",
        False,
        "Run: python prepare_data.py  (or check dataset_path in the config YAML).",
    ),
    (
        r"No such file or directory.*body_models|FileNotFoundError.*body_models|"
        r"support_dir.*not found|smplh.*not found|dmpls.*not found",
        "Body-model files missing (body_models/ directory not found)",
        False,
        "Ensure body_models/smplh/ and body_models/dmpls/ exist with the SMPL+H files.",
    ),
    (
        r"ModuleNotFoundError|ImportError|No module named",
        "Missing Python dependency",
        False,
        "Activate the aibase conda environment and install missing packages.",
    ),
    (
        r"loss.*nan|nan.*loss|NaN|is not finite|non-finite",
        "Training diverged — NaN / Inf in loss",
        False,
        "Consider lowering lr in the train config (impacts results — human decision).",
    ),
    (
        r"RuntimeError: CUDA error|CUDA error:|device-side assert",
        "CUDA runtime error (driver / GPU fault)",
        True,
        "Script will retry once. If it persists, check GPU health.",
    ),
    (
        r"resume_model.*not set|ValueError.*resume_model",
        "resume_model not set in test config",
        False,
        "This should be patched automatically after training. "
        "Check that training completed successfully.",
    ),
    (
        r"No such file.*\.pt|epoch_with_best_trainloss.*not found|checkpoint.*not found",
        "Checkpoint .pt file not found",
        False,
        "Verify that training completed and checkpoint_path in state is correct.",
    ),
    (
        r"Killed|OOMKilled|MemoryError",
        "Process killed by OS (likely system OOM or cgroup limit)",
        False,
        "Reduce batch_size (impacts results — human decision needed).",
    ),
]


def classify_error(output):
    """Return list of (cause, probability, auto_fixable, fix_hint), sorted by probability."""
    hits = []
    for pat, cause, auto_fix, hint in _ERROR_PATTERNS:
        matches = re.findall(pat, output, re.IGNORECASE)
        if matches:
            prob = min(0.97, 0.60 + 0.10 * len(matches))
            hits.append((cause, prob, auto_fix, hint))

    if not hits:
        hits.append((
            "Unknown error — check the log file for the full traceback",
            1.0, False, "Open the log file and look for the first Traceback or ERROR line.",
        ))

    hits.sort(key=lambda x: -x[1])
    return hits


def print_error_diagnosis(output, log_path, context=""):
    causes = classify_error(output)
    sep = "─" * 72
    print(f"\n{sep}", flush=True)
    print(f"  ERROR DIAGNOSIS{f' [{context}]' if context else ''}", flush=True)
    print(f"  Log: {log_path}", flush=True)
    print(sep, flush=True)
    for i, (cause, prob, auto_fix, hint) in enumerate(causes, 1):
        tag = "[auto-fixable]" if auto_fix else "[needs human] "
        print(f"  #{i} ({prob:.0%}) {tag}  {cause}", flush=True)
        print(f"       → {hint}", flush=True)
    print(sep + "\n", flush=True)
    return causes


# ── Experiment directory discovery ────────────────────────────────────────────

def list_exp_dirs(task=None):
    """Return the set of experiment subdirectories under experiments/.

    If `task` is given, only directories whose name starts with `<task>_`
    are returned — this limits discovery to a single variant and is how
    find_new_exp_dir() distinguishes a newly-spawned run from unrelated
    dirs created by other variants running in parallel.
    """
    if not EXP_ROOT.exists():
        return set()
    prefix = f"{task}_" if task else None
    return {
        str(p) for p in EXP_ROOT.iterdir()
        if p.is_dir() and (prefix is None or p.name.startswith(prefix))
    }


def find_new_exp_dir(before_dirs, task=None):
    after_dirs = list_exp_dirs(task=task)
    new_dirs = after_dirs - before_dirs
    if not new_dirs:
        return None
    if len(new_dirs) == 1:
        return new_dirs.pop()
    # Multiple new dirs — pick newest by creation time
    return max(new_dirs, key=lambda d: os.path.getctime(d))


# ── YAML patching ─────────────────────────────────────────────────────────────

def _read_yaml(config):
    return (ROOT / config).read_text()


def _write_yaml(config, content):
    (ROOT / config).write_text(content)


def patch_yaml_field(config, field, value):
    """Replace `field: <anything>` with `field: "<value>"` in a YAML file."""
    content = _read_yaml(config)
    new_content = re.sub(
        rf'^({re.escape(field)}:\s*).*$',
        f'{field}: "{value}"',
        content,
        flags=re.MULTILINE,
    )
    if new_content == content:
        log(f"WARNING: could not patch field '{field}' in {config}", "WARN")
        return False
    _write_yaml(config, new_content)
    log(f"Patched {config}: {field} = {value}")
    return True


def clear_yaml_field(config, field):
    """Reset `field: <anything>` to `field: null` in a YAML file."""
    content = _read_yaml(config)
    new_content = re.sub(
        rf'^({re.escape(field)}:\s*).*$',
        f'{field}: null',
        content,
        flags=re.MULTILINE,
    )
    if new_content == content:
        log(f"WARNING: could not clear field '{field}' in {config}", "WARN")
        return False
    _write_yaml(config, new_content)
    log(f"Cleared {config}: {field} = null")
    return True


def read_yaml_field(config, field):
    """Extract a field value from a YAML file using PyYAML (searches all levels)."""
    with open(ROOT / config) as f:
        data = yaml.safe_load(f)

    def _search(node):
        if isinstance(node, dict):
            if field in node:
                return node[field]
            for v in node.values():
                result = _search(v)
                if result is not None:
                    return result
        return None

    val = _search(data)
    return str(val) if val is not None else None


def patch_batch_size(config, new_bs):
    content = _read_yaml(config)
    new_content = re.sub(
        r'^(\s*batch_size:\s*)\d+',
        lambda m: m.group(1) + str(new_bs),
        content,
        flags=re.MULTILINE,
    )
    _write_yaml(config, new_content)
    log(f"Patched {config}: batch_size = {new_bs}")


# ── Step 0: Preflight ─────────────────────────────────────────────────────────

def run_preflight(state, dry_run=True, check_configs=None):
    """
    check_configs: optional list of config paths to pass via --configs.
    When None, all configs under options/ are checked (default behaviour).
    """
    label = "preflight (dry-run)" if dry_run else "preflight (--no-dry-run)"
    section(f"STEP 0: {label}")

    cmd = [PYTHON, "scripts/preflight.py"]
    if not dry_run:
        cmd.append("--no-dry-run")
    if check_configs:
        cmd += ["--configs"] + check_configs
    log_path = LOG_DIR / "preflight.log"
    rc, output = run_command(cmd, log_path)

    if dry_run:
        # Test configs showing [NOT SET] are expected — training hasn't run yet.
        # Only care that train-config dry runs all PASS.
        dry_fails = re.findall(r'\[FAIL\]\s+\S*train_\S+', output)
        if dry_fails:
            print_error_diagnosis(output, log_path, "preflight dry-run")
            user_action_needed(
                f"Preflight dry runs FAILED for:\n  " +
                "\n  ".join(dry_fails) +
                f"\n\nFix the errors shown above, then re-run.\nLog: {log_path}"
            )
        log("Preflight passed (test-config [NOT SET] is expected at this stage).")
        state["preflight_done"] = True
        save_state(state)
        return True
    else:
        if rc != 0:
            # [NOT SET] on test configs is now a real failure
            not_set = re.findall(r'\[NOT SET\]', output)
            print_error_diagnosis(output, log_path, "preflight --no-dry-run")
            user_action_needed(
                f"Preflight --no-dry-run FAILED "
                f"({len(not_set)} [NOT SET] entries found).\n"
                f"Some test configs may still have resume_model: null.\n"
                f"Log: {log_path}"
            )
        log("Preflight --no-dry-run passed — all test configs are valid.")
        return True


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_epoch(path):
    """Return the 0-indexed epoch stored in a checkpoint, or -1 on failure."""
    try:
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        return int(d["epoch"])
    except Exception:
        return -1


def _target_epochs(variant):
    """Return the configured epoch count for a variant (default 400)."""
    with open(ROOT / variant["train_config"]) as f:
        cfg = yaml.safe_load(f)
    try:
        return int(cfg["train_config"]["epochs"])
    except (KeyError, TypeError, ValueError):
        return 400


def _find_orphan_checkpoint(variant):
    """Scan the filesystem for an existing checkpoint for this variant.

    Looks for directories matching `experiments/<task>_*` that contain a
    `checkpoint_latest.pt`, and returns `(dir, epoch)` for the one whose
    checkpoint has the highest epoch. Returns `None` if nothing is found.

    This runs when `ablation_state.json` has no `checkpoint_dir` for a
    variant — e.g., first run after `--reset`, or a state wipe where a
    previous experiment dir was left on disk. It lets the pipeline pick
    up an interrupted (or already-completed) run without any operator
    action.
    """
    task_prefix = f"{variant['task']}_"
    if not EXP_ROOT.exists():
        return None

    candidates = []
    for p in EXP_ROOT.iterdir():
        if not p.is_dir() or not p.name.startswith(task_prefix):
            continue
        latest_ckpt = p / "checkpoint_latest.pt"
        if not latest_ckpt.exists():
            continue
        ep = _ckpt_epoch(str(latest_ckpt))
        if ep < 0:
            continue
        candidates.append((str(p), ep))

    if not candidates:
        return None
    # Prefer the most-advanced run. Ties broken by directory mtime (newest).
    candidates.sort(key=lambda x: (x[1], os.path.getmtime(x[0])))
    return candidates[-1]


# ── Step N: Train a variant ───────────────────────────────────────────────────
#
# ┌─ Checkpoint pickup order (run_training) ────────────────────────────────┐
# │ 1. state["variants"][vname]["checkpoint_dir"] — authoritative when set  │
# │    (populated during the current session, survives soft restarts).     │
# │ 2. Orphan filesystem scan: newest `experiments/<task>_*` dir with a    │
# │    `checkpoint_latest.pt`. Runs only when state has no checkpoint_dir, │
# │    so a `--reset` + existing exp dir still gets auto-resumed.         │
# │ 3. Neither present → fresh run: train.py creates a new exp dir.       │
# │                                                                        │
# │ Then within the discovered dir:                                        │
# │   - checkpoint epoch == target-1 → mark trained, skip training         │
# │   - checkpoint epoch  < target-1 → patch resume_checkpoint, resume     │
# │                                                                        │
# │ To force a fresh training for a task, either delete the orphan dir    │
# │ (`rm -rf experiments/<task>_*`) or rename it out of the prefix        │
# │ before launching.                                                      │
# └────────────────────────────────────────────────────────────────────────┘

def run_training(variant, state):
    vname = variant["name"]
    vstatus = state["variants"][vname]

    section(f"TRAINING: {variant['label']}  [{vname}]")

    # ── Step 2 of the pickup order: scan the filesystem for an orphan run ──
    # when state has nothing recorded for this variant. See the block comment
    # above run_training() for the full logic.
    ckpt_dir = vstatus.get("checkpoint_dir")
    if not ckpt_dir:
        orphan = _find_orphan_checkpoint(variant)
        if orphan:
            orphan_dir, orphan_epoch = orphan
            log(f"Found orphan checkpoint for {vname}: "
                f"{orphan_dir} (epoch {orphan_epoch + 1}/{_target_epochs(variant)})")
            vstatus["checkpoint_dir"] = orphan_dir
            save_state(state)
            ckpt_dir = orphan_dir

    # ── Detect if training already completed (script restarted mid-run) ────
    if ckpt_dir:
        latest_ckpt = os.path.join(ckpt_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_ckpt) and _ckpt_epoch(latest_ckpt) == _target_epochs(variant) - 1:
            # checkpoint_latest.pt contains the final epoch → training truly finished.
            log(f"Training already complete (epoch {_target_epochs(variant)} confirmed). Skipping re-train.")
            vstatus["status"] = "trained"
            vstatus["checkpoint_path"] = latest_ckpt
            save_state(state)
            clear_yaml_field(variant["train_config"], "resume_checkpoint")
            return True

        # checkpoint_latest exists but training incomplete → interrupted, try to resume
        if os.path.exists(latest_ckpt):
            log(f"Found interrupted training checkpoint: {latest_ckpt}")
            log("Patching train config to resume from checkpoint_latest.pt ...")
            patch_yaml_field(variant["train_config"], "resume_checkpoint", latest_ckpt)

    # ── Launch training ────────────────────────────────────────────────────
    before_dirs = list_exp_dirs(task=variant["task"])
    cmd = [PYTHON, "train.py", "--config", variant["train_config"]]
    log_path = LOG_DIR / f"{vname}_train.log"

    vstatus["status"] = "training"
    vstatus["train_start_time"] = _ts()
    save_state(state)

    rc, output = run_command(cmd, log_path)

    # ── Find created experiment directory ──────────────────────────────────
    # NOTE: train.py reuses the resume_checkpoint's parent dir on resume, so a
    # new timestamped dir is only created on the *first* run for a variant.
    # If no new dir appears, training resumed in-place and the existing
    # checkpoint_dir is still authoritative.
    new_dir = find_new_exp_dir(before_dirs, task=variant["task"])
    if new_dir:
        vstatus["checkpoint_dir"] = new_dir
        log(f"Experiment directory: {new_dir}")

    # Prefer new_dir over previously stored ckpt_dir for checkpoint lookup
    lookup_dir = new_dir or vstatus.get("checkpoint_dir")
    if lookup_dir:
        latest_ckpt = os.path.join(lookup_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_ckpt):
            vstatus["checkpoint_path"] = latest_ckpt

    save_state(state)

    # ── Success ────────────────────────────────────────────────────────────
    if rc == 0:
        latest_ckpt = os.path.join(lookup_dir, "checkpoint_latest.pt") if lookup_dir else None
        if latest_ckpt and os.path.exists(latest_ckpt):
            vstatus["status"] = "trained"
            vstatus["checkpoint_path"] = latest_ckpt
            vstatus["last_error"] = None
            save_state(state)
            # Reset resume_checkpoint so a future fresh run doesn't accidentally
            # warm-start from this checkpoint.
            clear_yaml_field(variant["train_config"], "resume_checkpoint")
            log(f"Training complete. Checkpoint: {latest_ckpt}")
            return True
        # rc=0 but no checkpoint yet? Very unlikely — treat as success anyway.
        vstatus["status"] = "trained"
        save_state(state)
        clear_yaml_field(variant["train_config"], "resume_checkpoint")
        log("Training reported rc=0. No best checkpoint found yet (unexpected).", "WARN")
        return True

    # ── Failure diagnosis ──────────────────────────────────────────────────
    tail = output[-4000:]  # last 4 KB is most informative
    causes = print_error_diagnosis(tail, log_path, f"train {vname}")
    top_cause, _, _, fix_hint = causes[0]

    vstatus["last_error"] = top_cause
    vstatus["error_history"].append({"time": _ts(), "rc": rc, "cause": top_cause})
    vstatus["train_retries"] += 1
    save_state(state)

    if vstatus["train_retries"] > MAX_AUTO_RETRIES:
        user_action_needed(
            f"Training {vname} failed {vstatus['train_retries']} times in a row.\n"
            f"Last error: {top_cause}\n"
            f"Hint: {fix_hint}\n"
            f"Log: {log_path}"
        )

    # ── Automated fixes ────────────────────────────────────────────────────
    if "GPU out of memory" in top_cause:
        old_bs_str = read_yaml_field(variant["train_config"], "batch_size")
        old_bs = int(old_bs_str) if old_bs_str and old_bs_str.isdigit() else 128
        new_bs = old_bs // 2
        if new_bs < 8:
            user_action_needed(
                f"OOM detected for {vname} with batch_size={old_bs}.\n"
                f"Reducing further to {new_bs} would likely hurt training quality.\n"
                f"Please decide on a batch_size and update the config manually."
            )
        log(f"AUTO-FIX: OOM — reducing batch_size {old_bs} → {new_bs}.", "FIX")
        log("NOTE: smaller batch_size changes gradient noise; results may differ slightly.", "WARN")
        patch_batch_size(variant["train_config"], new_bs)
        return run_training(variant, state)

    if "CUDA runtime error" in top_cause:
        log(f"AUTO-FIX: CUDA runtime error — retrying (attempt {vstatus['train_retries']}).", "FIX")
        return run_training(variant, state)

    # Try to resume from checkpoint_latest.pt if one was written
    if lookup_dir:
        latest_ckpt = os.path.join(lookup_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_ckpt):
            log(f"AUTO-FIX: Patching resume_checkpoint → {latest_ckpt} and retrying.", "FIX")
            patch_yaml_field(variant["train_config"], "resume_checkpoint", latest_ckpt)
            return run_training(variant, state)

    # No automated fix available
    user_action_needed(
        f"Training {vname} failed.\n"
        f"Most likely cause: {top_cause}\n"
        f"Hint: {fix_hint}\n"
        f"Log: {log_path}\n\n"
        f"If resuming from a partial run, set resume_checkpoint in {variant['train_config']}."
    )
    return False


# ── Patch test config after training ─────────────────────────────────────────

def patch_test_config_after_training(variant, state):
    vname = variant["name"]
    vstatus = state["variants"][vname]

    ckpt = vstatus.get("checkpoint_path")
    if not ckpt:
        # Try to locate it from checkpoint_dir
        ckpt_dir = vstatus.get("checkpoint_dir")
        if ckpt_dir:
            candidate = os.path.join(ckpt_dir, "checkpoint_latest.pt")
            if os.path.exists(candidate):
                ckpt = candidate
                vstatus["checkpoint_path"] = ckpt
                save_state(state)

    if not ckpt or not os.path.exists(ckpt):
        user_action_needed(
            f"No valid checkpoint found for {vname}.\n"
            f"Stored path: {ckpt}\n\n"
            f"Set checkpoint_path manually in {STATE_FILE}, then re-run."
        )

    ok = patch_yaml_field(variant["test_config"], "resume_model", ckpt)
    if not ok:
        user_action_needed(
            f"Could not patch resume_model in {variant['test_config']}.\n"
            f"Set it manually to: {ckpt}"
        )
    return True


# ── Step N: Evaluate a variant ────────────────────────────────────────────────

def run_evaluation(variant, state):
    vname = variant["name"]
    vstatus = state["variants"][vname]

    section(f"EVALUATING: {variant['label']}  [{vname}]")

    cmd = [PYTHON, "test_sliding_window.py", "--config", variant["test_config"]]
    log_path = LOG_DIR / f"{vname}_eval.log"

    vstatus["status"] = "evaluating"
    save_state(state)

    rc, output = run_command(cmd, log_path)

    if rc == 0:
        vstatus["status"] = "done"
        vstatus["last_error"] = None
        vstatus["eval_table"] = _extract_pretty_table(output)
        save_state(state)
        log(f"Evaluation complete for {vname}.")
        return True

    # Failure diagnosis
    tail = output[-4000:]
    causes = print_error_diagnosis(tail, log_path, f"eval {vname}")
    top_cause, _, _, fix_hint = causes[0]

    vstatus["last_error"] = top_cause
    vstatus["error_history"].append({"time": _ts(), "rc": rc, "cause": top_cause, "phase": "eval"})
    vstatus["eval_retries"] += 1
    save_state(state)

    if vstatus["eval_retries"] > MAX_AUTO_RETRIES:
        user_action_needed(
            f"Evaluation {vname} failed {vstatus['eval_retries']} times.\n"
            f"Last error: {top_cause}\nHint: {fix_hint}\nLog: {log_path}"
        )

    if "GPU out of memory" in top_cause:
        # test_sliding_window.py uses train_config.batch_size for sliding-window batching
        old_bs_str = read_yaml_field(variant["test_config"], "batch_size")
        old_bs = int(old_bs_str) if old_bs_str and old_bs_str.isdigit() else 128
        new_bs = max(1, old_bs // 2)
        log(f"AUTO-FIX: OOM in eval — reducing batch_size {old_bs} → {new_bs}.", "FIX")
        patch_batch_size(variant["test_config"], new_bs)
        return run_evaluation(variant, state)

    if "CUDA runtime error" in top_cause:
        log(f"AUTO-FIX: CUDA runtime error in eval — retrying.", "FIX")
        return run_evaluation(variant, state)

    user_action_needed(
        f"Evaluation {vname} failed.\n"
        f"Most likely cause: {top_cause}\nHint: {fix_hint}\nLog: {log_path}"
    )
    return False


# ── Result collection ─────────────────────────────────────────────────────────

def _extract_pretty_table(output):
    """Extract the PrettyTable block from test_sliding_window.py output."""
    lines = output.splitlines()
    # Find first and last separator line (+----+---+...)
    sep_re = re.compile(r'^\+[-+]+\+\s*$')
    idxs = [i for i, ln in enumerate(lines) if sep_re.match(ln)]
    if len(idxs) >= 2:
        return "\n".join(lines[idxs[0]: idxs[-1] + 1])
    return None


def load_variant_results(save_dir):
    """
    Load per-sequence .pt files, return dict of {metric: mean_value}.
    Values in the .pt files are already multiplied by their coefficient.
    """
    try:
        import torch
    except ImportError:
        log("torch not importable in this process — cannot load .pt results.", "WARN")
        return None

    pt_files = sorted(glob.glob(os.path.join(str(ROOT), save_dir, "*.pt")))
    if not pt_files:
        return None

    sums = {m: 0.0 for m in METRICS}
    count = 0
    for f in pt_files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            for m in METRICS:
                sums[m] += d["metrics"][m]
            count += 1
        except Exception as exc:
            log(f"Could not load {f}: {exc}", "WARN")

    if count == 0:
        return None

    return {m: sums[m] / count for m in METRICS}


# ── Summary generation ────────────────────────────────────────────────────────

def generate_summary(state):
    section("GENERATING SUMMARY")

    results = {}
    for v in VARIANTS:
        vname = v["name"]
        r = load_variant_results(v["save_dir"])
        if r:
            results[vname] = r
            log(f"  {vname}: MPJPE={r['mpjpe']:.2f} cm  MPJRE={r['mpjre']:.2f}°")
        else:
            log(f"  {vname}: no results yet (will appear as '—')", "WARN")

    summary_path = ROOT / "results" / "ablation" / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Table columns
    cols = ["Variant", "MPJPE↓ (cm)", "MPJRE↓ (°)", "HandPE↓ (cm)",
            "UpperPE↓ (cm)", "LowerPE↓ (cm)", "RootPE↓ (cm)", "Jitter↓"]

    def _row(label, r):
        if r is None:
            return "| " + " | ".join([label] + ["—"] * (len(cols) - 1)) + " |"
        vals = [
            f"{r['mpjpe']:.2f}", f"{r['mpjre']:.2f}", f"{r['handpe']:.2f}",
            f"{r['upperpe']:.2f}", f"{r['lowerpe']:.2f}", f"{r['rootpe']:.2f}",
            f"{r['pred_jitter']:.2f}",
        ]
        return "| " + " | ".join([label] + vals) + " |"

    md = [
        "# Ablation Study Results — HMD-Poser",
        "",
        f"Generated: {_ts()}",
        "",
        "**Protocol:** sliding-window inference (stride 1), Protocol 1 AMASS, HMD-only input.",
        "",
        "## Metric Summary",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
        # Baseline published numbers (CLAUDE.md)
        "| Baseline LSTM+Transformer *(published)* | 3.15 | — | — | — | — | — | — |",
    ]

    for v in VARIANTS:
        md.append(_row(v["label"], results.get(v["name"])))

    md += [
        "",
        "## Architecture Variants",
        "",
        "| Model name | Temporal module | Spatial module |",
        "| --- | --- | --- |",
        "| `HMD_imu_HME_Universe` (baseline) | Unidirectional LSTM | Transformer |",
        "| `HMD_imu_HME_TemporalTransformer` | Causal Transformer | Transformer |",
        "| `HMD_imu_HME_SpatialOnly` | None | Transformer |",
        "| `HMD_imu_HME_TemporalOnly` | LSTM | None |",
        "",
        "## Research Questions",
        "",
        "1. **LSTM vs causal Transformer as temporal module** — "
        "compare *Baseline* vs *CausalTransformer Temporal*.",
        "2. **Do both modules contribute?** — "
        "compare *Baseline* vs *Spatial-Only* (removes LSTM) and *Temporal-Only* (removes Transformer).",
        "",
        "## Run Details",
        "",
        f"State file: `{STATE_FILE}`",
        "",
        "| Variant | Status | Checkpoint |",
        "| --- | --- | --- |",
    ]

    for v in VARIANTS:
        vstatus = state["variants"][v["name"]]
        ckpt = vstatus.get("checkpoint_path") or "—"
        ckpt_short = os.path.basename(os.path.dirname(ckpt)) if ckpt != "—" else "—"
        md.append(f"| {v['name']} | {vstatus['status']} | {ckpt_short} |")

    md += [
        "",
        "## Notes",
        "",
        "- `shape=0.0000` in all training logs is expected (shape head zeroed in `hmd_imu_model.py:160`).",
        "- `torch.cross` and `weight_norm` deprecation warnings are pre-existing; safe to ignore.",
        "- All metrics are mean over test sequences. Values already scaled (cm, degrees, etc.).",
    ]

    # Per-variant eval tables if captured
    any_table = any(
        state["variants"][v["name"]].get("eval_table") for v in VARIANTS
    )
    if any_table:
        md += ["", "## Per-Variant Evaluation Output", ""]
        for v in VARIANTS:
            tbl = state["variants"][v["name"]].get("eval_table")
            if tbl:
                md += [f"### {v['label']}", "", "```", tbl, "```", ""]

    summary_path.write_text("\n".join(md))
    log(f"Summary written to {summary_path}")
    state["summary_generated"] = True
    save_state(state)


# ── Status printer ────────────────────────────────────────────────────────────

def print_status(state):
    section("ABLATION STUDY STATUS")
    print(f"  State file: {STATE_FILE}")
    print(f"  Preflight:  {'done' if state['preflight_done'] else 'pending'}")
    print()
    print(f"  {'Variant':<25} {'Status':<14} {'Checkpoint':<60} {'Errors'}")
    print("  " + "─" * 110)
    for v in VARIANTS:
        vs = state["variants"][v["name"]]
        ckpt = vs.get("checkpoint_path") or "—"
        ckpt_s = ckpt if len(ckpt) < 58 else "..." + ckpt[-55:]
        err = vs.get("last_error") or "—"
        err_s = err[:40] if len(err) > 40 else err
        retries = vs.get("train_retries", 0) + vs.get("eval_retries", 0)
        retry_s = f"(retries={retries})" if retries else ""
        print(f"  {v['name']:<25} {vs['status']:<14} {ckpt_s:<60} {err_s} {retry_s}")
    print()
    print(f"  Summary generated: {state['summary_generated']}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="HMD-Poser ablation study — automated workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--reset", action="store_true",
                   help="Delete state file and start fresh")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip step 0 preflight dry-run")
    p.add_argument("--summary-only", action="store_true",
                   help="Only (re-)generate the summary markdown")
    p.add_argument("--status", action="store_true",
                   help="Print progress status and exit")
    p.add_argument("--variant", metavar="NAME",
                   help="Run only a single variant "
                        "(temporal_transformer | spatial_only | temporal_only)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            log("State file deleted — starting fresh.")

    state = load_state()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_status(state)
        return

    if args.summary_only:
        generate_summary(state)
        return

    section("HMD-Poser Ablation Study — Automated Workflow")
    log(f"State file: {STATE_FILE}")
    log(f"Log dir:    {LOG_DIR}")

    # Filter to single variant if requested
    variants_to_run = VARIANTS
    if args.variant:
        variants_to_run = [v for v in VARIANTS if v["name"] == args.variant]
        if not variants_to_run:
            log(f"Unknown variant '{args.variant}'. "
                f"Valid: {[v['name'] for v in VARIANTS]}", "ERROR")
            sys.exit(1)
        log(f"Running single variant: {args.variant}")

    # ── Step 0: Preflight dry-run ──────────────────────────────────────────
    if not state["preflight_done"] and not args.skip_preflight:
        run_preflight(state, dry_run=True)
    else:
        log("Preflight already done (skipping).")

    # ── Steps 1-8: Train each variant, then evaluate immediately ────────────
    # Interleaved flow: each variant is trained → test-config patched →
    # --no-dry-run preflight on that one test config → sliding-window eval.
    # Running eval right after training means results are available as soon
    # as possible and any eval bug surfaces early instead of 30 hours later.
    for variant in variants_to_run:
        vname = variant["name"]
        vstatus = state["variants"][vname]

        if vstatus["status"] in ("pending", "training", "failed"):
            run_training(variant, state)

        # Patch test config with discovered checkpoint (idempotent)
        if state["variants"][vname]["status"] == "trained":
            patch_test_config_after_training(variant, state)
            # Preflight --no-dry-run for *this* test config only
            log(f"Verifying test config for {vname} ...")
            run_preflight(state, dry_run=False,
                          check_configs=[variant["test_config"]])

        # Evaluate immediately after training completes
        if state["variants"][vname]["status"] in ("trained", "evaluating"):
            run_evaluation(variant, state)

    # ── Step 9: Summary ───────────────────────────────────────────────────
    any_done = any(
        state["variants"][v["name"]]["status"] == "done"
        for v in variants_to_run
    )
    if any_done:
        generate_summary(state)

    # ── Final status ──────────────────────────────────────────────────────
    print_status(state)
    all_done = all(
        state["variants"][v["name"]]["status"] == "done"
        for v in variants_to_run
    )
    if all_done:
        log("All ablation variants complete.")
        log(f"Results: results/ablation/summary.md")
    else:
        pending = [
            v["name"] for v in variants_to_run
            if state["variants"][v["name"]]["status"] != "done"
        ]
        log(f"Variants not yet complete: {pending}")
        log("Re-run to continue from where we left off.")


if __name__ == "__main__":
    main()
