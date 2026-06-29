import os
import csv
import math
import time
import argparse

import numpy as np
import torch
import prettytable as pt

from utils.utils_config import load_config
from dataset.dataloader import load_data
from model.hmd_imu_model import HMDIMUModel
from utils import utils_transform
from utils.metrics import compute_per_frame_metrics, get_metric_function

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre", "mpjpe", "mpjve",
    "rootre", "upperre", "lowerre",
    "handpe", "upperpe", "lowerpe", "rootpe",
    "upperve", "lowerve",
    "pred_jitter",
]
gt_metrics = ["gt_jitter"]
all_metrics = pred_metrics + gt_metrics

metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "rootre": RADIANS_TO_DEGREES,
    "upperre": RADIANS_TO_DEGREES,
    "lowerre": RADIANS_TO_DEGREES,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "upperve": METERS_TO_CENTIMETERS,
    "lowerve": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--resume-model", type=str, default=None,
                   help="Override resume_model in config.")
    p.add_argument("--dataset-path", type=str, default=None,
                   help="Override dataset_path in config.")
    p.add_argument("--skip-frames", type=int, default=79,
                   help="Drop first N frames for the secondary skip-N aggregate "
                        "(skip-0 is always reported). Sequences with seq_len <= N+3 "
                        "are excluded from the skip-N aggregate but remain in skip-0. "
                        "Pass 0 to disable.")
    p.add_argument("--save-frames", action="store_true",
                   help="Dump a per-sequence .npz under "
                        "<run>/eval/frames/<input_type>/<seq_idx>.npz with "
                        "predicted body_parms (root_orient, pose_body axis-angle), "
                        "head-aligned predicted joint positions, GT body_parms + "
                        "joint positions + betas, and per-frame metric vectors. "
                        "NaN at frames where a metric is undefined "
                        "(frame 0 for VE, 0..2 for jitter). Off by default.")
    return p.parse_args()


def prepare_input_feat(input_feat, input_type):
    input_feat = input_feat.clone()
    if input_type == "HMD_2IMUs":
        input_feat[..., list(range(30, 36)) + list(range(66, 72)) + list(range(132, 135))] = 0
    elif input_type == "HMD":
        input_feat[..., list(range(18, 36)) + list(range(54, 72)) + list(range(126, 135))] = 0
    return input_feat


def compute_seq_metrics(predicted_position, predicted_angle, predicted_root_angle,
                        gt_position, gt_angle, gt_root_angle, upper_index, lower_index, fps):
    out = {}
    for m in all_metrics:
        out[m] = (
            get_metric_function(m)(
                predicted_position, predicted_angle, predicted_root_angle,
                gt_position, gt_angle, gt_root_angle,
                upper_index, lower_index, fps=fps,
            ).cpu().numpy()
        )
    return out


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _dump_per_sequence_frames(
    out_dir, seq_idx, filepath, input_type, fps,
    predicted_position, predicted_angle, predicted_root_angle, pred_trans,
    gt_position, gt_angle, gt_root_angle, gt_trans, gt_betas,
    upper_index, lower_index,
):
    """Write a per-sequence .npz with predicted/GT body_parms and per-frame
    metric vectors. Flat ``__``-separated keys (np.savez can't nest dicts).

    Body-parm key convention (mirrors VR_Pose_Pred): ``pred__<k>`` and
    ``gt__<k>`` with ``k`` in ``{root_orient, pose_body, trans, betas}``;
    ``betas`` is GT-only. Joint positions are not stored — downstream code
    re-runs SMPL FK from the body_parms.

    ``predicted_position`` / ``gt_position`` (head-aligned, joint 15) feed
    the per-frame metric computation; only the metrics survive in the dump.
    """
    os.makedirs(out_dir, exist_ok=True)
    frame_metrics = compute_per_frame_metrics(
        predicted_position, predicted_angle, predicted_root_angle,
        gt_position, gt_angle, gt_root_angle,
        upper_index, lower_index, fps=fps,
    )
    flat = {
        "filepath": np.array(str(filepath)),
        "seq_idx": np.array(int(seq_idx)),
        "input_type": np.array(str(input_type)),
        "T": np.array(int(predicted_position.shape[0])),
        "fps": np.array(int(fps)),
        "pred__root_orient": _to_numpy(predicted_root_angle).astype(np.float32),
        "pred__pose_body": _to_numpy(predicted_angle).astype(np.float32),
        "pred__trans": _to_numpy(pred_trans).astype(np.float32),
        "gt__root_orient": _to_numpy(gt_root_angle).astype(np.float32),
        "gt__pose_body": _to_numpy(gt_angle).astype(np.float32),
    }
    if gt_trans is not None:
        flat["gt__trans"] = _to_numpy(gt_trans).astype(np.float32)
    if gt_betas is not None:
        flat["gt__betas"] = _to_numpy(gt_betas).astype(np.float32)
    for k, v in frame_metrics.items():
        flat[f"frame_metrics__{k}"] = v
    np.savez(os.path.join(out_dir, f"{seq_idx}.npz"), **flat)


def main():
    args = parse_args()
    configs = load_config(args.config)
    if args.resume_model is not None:
        configs.resume_model = args.resume_model
    if args.dataset_path is not None:
        configs.dataset_path = args.dataset_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading test data from: {configs.dataset_path}")
    test_datas = load_data(
        configs.dataset_path, "test",
        input_motion_length=configs.input_motion_length,
    )

    skip_n = args.skip_frames
    print(f"Skip-N eval: N = {skip_n} (skip-0 always reported)")

    if not configs.resume_model:
        raise ValueError("resume_model is not set (pass --resume-model or set in config).")
    model = HMDIMUModel(configs, device)
    model.load(configs.resume_model)
    print(f"Successfully resumed checkpoint from {configs.resume_model}")
    model.eval()

    # Derive run root from resume_model (mirrors test_sliding_window.py).
    ckpt_dir = os.path.dirname(os.path.abspath(configs.resume_model))
    if os.path.basename(ckpt_dir) in ("ckpts", "checkpoints"):
        run_root = os.path.dirname(ckpt_dir)
    else:
        run_root = ckpt_dir
    eval_dir = os.path.join(run_root, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Eval outputs will be saved to: {eval_dir}")

    upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]

    tb = pt.PrettyTable()
    tb.field_names = ["Input_type"] + pred_metrics + gt_metrics

    for input_type in configs.compatible_inputs:
        print(f"\nFull-sequence eval on {input_type}")

        seq_metrics_list, seq_filepaths, seq_lens = [], [], []
        seq_metrics_skip_list, seq_filepaths_skip, seq_lens_skip = [], [], []

        total = len(test_datas)
        t0 = time.time()
        for motion_idx, motion_data in enumerate(test_datas):
            elapsed = time.time() - t0
            print(f"  {motion_idx + 1}/{total}  elapsed={elapsed:.1f}s", flush=True)

            input_feat = prepare_input_feat(motion_data["input_feat"], input_type)
            input_feat = input_feat.unsqueeze(0).to(device).float()  # (1, T, 135)
            gt_local_pose = motion_data["gt_local_pose"].to(device).float()
            gt_positions = motion_data["gt_positions"].to(device).float()
            gt_betas = motion_data.get("gt_betas", None)
            gt_trans = motion_data.get("gt_trans", None)

            with torch.no_grad():
                pred_local_pose, _, _, pred_joint_position = model(input_feat)
            pred_local_pose = pred_local_pose[0]            # (T, 22, 6)
            pred_joint_position = pred_joint_position[0]    # (T, 22, 3)

            seq_len = pred_local_pose.shape[0]
            pred_local_pose_aa = utils_transform.sixd2aa(
                pred_local_pose.reshape(-1, 6).detach()
            ).reshape(seq_len, 22 * 3)
            gt_local_pose_aa = utils_transform.sixd2aa(
                gt_local_pose.reshape(-1, 6).detach()
            ).reshape(seq_len, 22 * 3)

            pred_joint_position = pred_joint_position.detach()
            gt_positions = gt_positions.detach()
            # Head-alignment offset doubles as the predicted SMPL trans:
            # FK at zero-trans yields `pred_joint_position`; setting trans to
            # `gt_head - pred_head_local` lands the predicted head exactly on
            # the GT head, which is the alignment convention used everywhere
            # downstream. Capture before the in-place add overwrites the
            # local positions.
            pred_trans = gt_positions[:, 15] - pred_joint_position[:, 15]  # (T, 3)
            pred_joint_position = (
                pred_joint_position - pred_joint_position[:, 15:16] + gt_positions[:, 15:16]
            )

            predicted_angle = pred_local_pose_aa[..., 3:66]
            predicted_root_angle = pred_local_pose_aa[..., :3]
            predicted_position = pred_joint_position
            gt_angle = gt_local_pose_aa[..., 3:66]
            gt_root_angle = gt_local_pose_aa[..., :3]
            gt_position = gt_positions

            eval_log = compute_seq_metrics(
                predicted_position, predicted_angle, predicted_root_angle,
                gt_position, gt_angle, gt_root_angle,
                upper_index, lower_index, fps=60,
            )
            seq_metrics_list.append(
                {m: float(eval_log[m]) * metrics_coeffs[m] for m in all_metrics}
            )
            seq_filepaths.append(str(motion_data["filepath"]))
            seq_lens.append(seq_len)

            long_enough = skip_n > 0 and skip_n < seq_len - 3
            if long_enough:
                eval_log_skip = compute_seq_metrics(
                    predicted_position[skip_n:], predicted_angle[skip_n:], predicted_root_angle[skip_n:],
                    gt_position[skip_n:], gt_angle[skip_n:], gt_root_angle[skip_n:],
                    upper_index, lower_index, fps=60,
                )
                seq_metrics_skip_list.append(
                    {m: float(eval_log_skip[m]) * metrics_coeffs[m] for m in all_metrics}
                )
                seq_filepaths_skip.append(str(motion_data["filepath"]))
                seq_lens_skip.append(seq_len)

            if args.save_frames:
                _dump_per_sequence_frames(
                    out_dir=os.path.join(eval_dir, "frames", input_type),
                    seq_idx=motion_idx,
                    filepath=motion_data["filepath"],
                    input_type=input_type,
                    fps=60,
                    predicted_position=predicted_position,
                    predicted_angle=predicted_angle,
                    predicted_root_angle=predicted_root_angle,
                    pred_trans=pred_trans,
                    gt_position=gt_position,
                    gt_angle=gt_angle,
                    gt_root_angle=gt_root_angle,
                    gt_trans=gt_trans,
                    gt_betas=gt_betas,
                    upper_index=upper_index,
                    lower_index=lower_index,
                )

        def _avg(seqs, metric):
            return float(np.mean([d[metric] for d in seqs])) if seqs else 0.0

        tb.add_row(
            [f"{input_type} (skip-0, n={len(seq_metrics_list)})"]
            + ["%.2f" % _avg(seq_metrics_list, m) for m in pred_metrics + gt_metrics]
        )
        if skip_n > 0 and seq_metrics_skip_list:
            tb.add_row(
                [f"{input_type} (skip-{skip_n}, n={len(seq_metrics_skip_list)})"]
                + ["%.2f" % _avg(seq_metrics_skip_list, m) for m in pred_metrics + gt_metrics]
            )

        def _save(metrics_list, filepaths, lens, suffix):
            if not metrics_list:
                return
            npz_data = {
                m: np.array([d[m] for d in metrics_list], dtype=np.float64)
                for m in all_metrics
            }
            npz_data["filepath"] = np.array(filepaths)
            npz_data["seq_len"] = np.array(lens, dtype=np.int64)
            np.savez(
                os.path.join(eval_dir, f"eval_full_{input_type}_seq{suffix}.npz"),
                **npz_data,
            )
            csv_path = os.path.join(eval_dir, f"eval_full_{input_type}_avg{suffix}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for m in all_metrics:
                    writer.writerow([m, f"{float(npz_data[m].mean()):.2f}"])

        _save(seq_metrics_list, seq_filepaths, seq_lens, "")
        if skip_n > 0 and seq_metrics_skip_list:
            _save(
                seq_metrics_skip_list, seq_filepaths_skip, seq_lens_skip,
                f"_skip{skip_n}f",
            )

    print("\n" + "=" * 80)
    print("Full-sequence Eval Results")
    print(f"Skip-N: {skip_n}")
    print("=" * 80)
    print(tb)


if __name__ == "__main__":
    main()
