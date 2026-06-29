# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Metric functions with same inputs

import math

import numpy as np
import torch


RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0


# Each metric accepts `reduce_time=True` (default) for the original scalar
# tensor and `reduce_time=False` for a (T,) per-frame vector. Velocity
# metrics prepend 1 NaN (frame 0 undefined); jitter prepends 3 NaNs (frames
# 0..2 undefined). NaN-aware reductions (np.nanmean) recover the scalar.
def _prepend_nan(per_frame, n):
    nan = torch.full(
        (n,), float("nan"), device=per_frame.device, dtype=per_frame.dtype
    )
    return torch.cat([nan, per_frame], dim=0)


def pred_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    j = (
        (
            predicted_position[3:]
            - 3 * predicted_position[2:-1]
            + 3 * predicted_position[1:-2]
            - predicted_position[:-3]
        )
        * (fps**3)
    ).norm(dim=2)
    if reduce_time:
        return j.mean()
    return _prepend_nan(j.mean(dim=-1), 3)


def gt_jitter(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    j = (
        (
            gt_position[3:]
            - 3 * gt_position[2:-1]
            + 3 * gt_position[1:-2]
            - gt_position[:-3]
        )
        * (fps**3)
    ).norm(dim=2)
    if reduce_time:
        return j.mean()
    return _prepend_nan(j.mean(dim=-1), 3)


def mpjre(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    diff = gt_angle - predicted_angle
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    if reduce_time:
        return torch.mean(torch.absolute(diff))
    return torch.absolute(diff).mean(dim=-1)


def mpjpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    pe = torch.sqrt(
        torch.sum(torch.square(gt_position - predicted_position), axis=-1)
    )
    if reduce_time:
        return torch.mean(pe)
    return pe.mean(dim=-1)


def handpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    pe = torch.sqrt(
        torch.sum(torch.square(gt_position - predicted_position), axis=-1)
    )[..., [20, 21]]
    if reduce_time:
        return torch.mean(pe)
    return pe.mean(dim=-1)


def handpe_l(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    pe = torch.sqrt(
        torch.sum(torch.square(gt_position[:, 20] - predicted_position[:, 20]), dim=-1)
    )
    return pe.mean() if reduce_time else pe


def handpe_r(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    pe = torch.sqrt(
        torch.sum(torch.square(gt_position[:, 21] - predicted_position[:, 21]), dim=-1)
    )
    return pe.mean() if reduce_time else pe


def upperpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    pe = torch.sqrt(
        torch.sum(torch.square(gt_position - predicted_position), axis=-1)
    )[..., upper_index]
    if reduce_time:
        return torch.mean(pe)
    return pe.mean(dim=-1)


def lowerpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    pe = torch.sqrt(
        torch.sum(torch.square(gt_position - predicted_position), axis=-1)
    )[..., lower_index]
    if reduce_time:
        return torch.mean(pe)
    return pe.mean(dim=-1)


def rootpe(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    pe = torch.sqrt(
        torch.sum(torch.square(gt_position - predicted_position), axis=-1)
    )[..., [0]]
    if reduce_time:
        return torch.mean(pe)
    return pe.mean(dim=-1)


def mpjve(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (
        predicted_position[1:, ...] - predicted_position[:-1, ...]
    ) * fps
    ve = torch.sqrt(
        torch.sum(torch.square(gt_velocity - predicted_velocity), axis=-1)
    )
    if reduce_time:
        return torch.mean(ve)
    return _prepend_nan(ve.mean(dim=-1), 1)


def rootre(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    diff = gt_root_angle - predicted_root_angle
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    if reduce_time:
        return torch.mean(torch.absolute(diff))
    return torch.absolute(diff).mean(dim=-1)


def upperre(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    gt = torch.cat((gt_root_angle, gt_angle), dim=-1)
    pred = torch.cat((predicted_root_angle, predicted_angle), dim=-1)
    diff = (gt - pred).reshape(-1, 22, 3)[:, upper_index]
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    if reduce_time:
        return torch.mean(torch.absolute(diff))
    return torch.absolute(diff).mean(dim=(1, 2))


def lowerre(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    gt = torch.cat((gt_root_angle, gt_angle), dim=-1)
    pred = torch.cat((predicted_root_angle, predicted_angle), dim=-1)
    diff = (gt - pred).reshape(-1, 22, 3)[:, lower_index]
    diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
    diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
    if reduce_time:
        return torch.mean(torch.absolute(diff))
    return torch.absolute(diff).mean(dim=(1, 2))


def upperve(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (
        predicted_position[1:, ...] - predicted_position[:-1, ...]
    ) * fps
    ve = torch.sqrt(
        torch.sum(torch.square(gt_velocity - predicted_velocity), axis=-1)
    )[..., upper_index]
    if reduce_time:
        return torch.mean(ve)
    return _prepend_nan(ve.mean(dim=-1), 1)


def lowerve(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    *,
    reduce_time=True,
):
    gt_velocity = (gt_position[1:, ...] - gt_position[:-1, ...]) * fps
    predicted_velocity = (
        predicted_position[1:, ...] - predicted_position[:-1, ...]
    ) * fps
    ve = torch.sqrt(
        torch.sum(torch.square(gt_velocity - predicted_velocity), axis=-1)
    )[..., lower_index]
    if reduce_time:
        return torch.mean(ve)
    return _prepend_nan(ve.mean(dim=-1), 1)


metric_funcs_dict = {
    "mpjre": mpjre,
    "mpjpe": mpjpe,
    "mpjve": mpjve,
    "rootre": rootre,
    "upperre": upperre,
    "lowerre": lowerre,
    "handpe": handpe,
    "upperpe": upperpe,
    "lowerpe": lowerpe,
    "rootpe": rootpe,
    "upperve": upperve,
    "lowerve": lowerve,
    "pred_jitter": pred_jitter,
    "gt_jitter": gt_jitter,
}


# Coefficients applied so per-frame metric values land in display units
# (cm / deg / jitter), matching the aggregate CSV/NPZ columns produced by
# the test scripts.
metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "rootre": RADIANS_TO_DEGREES,
    "upperre": RADIANS_TO_DEGREES,
    "lowerre": RADIANS_TO_DEGREES,
    "handpe": METERS_TO_CENTIMETERS,
    "handpe_l": METERS_TO_CENTIMETERS,
    "handpe_r": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "upperve": METERS_TO_CENTIMETERS,
    "lowerve": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
}


# Per-frame registry: superset of metric_funcs_dict adding the per-side
# Hand-PE split (kept out of the aggregate dict so existing test outputs
# stay unchanged).
per_frame_metric_funcs = {
    **metric_funcs_dict,
    "handpe_l": handpe_l,
    "handpe_r": handpe_r,
}


def get_metric_function(metric):
    return metric_funcs_dict[metric]


def compute_per_frame_metrics(
    predicted_position,
    predicted_angle,
    predicted_root_angle,
    gt_position,
    gt_angle,
    gt_root_angle,
    upper_index,
    lower_index,
    fps,
    metrics=None,
):
    """Per-frame analogue of the scalar metric calls in the test scripts.

    Returns ``{metric_name: (T,) np.ndarray[float32]}`` with NaN at frames
    where the metric is undefined (frame 0 for VE, frames 0..2 for jitter).
    Coefficients from ``metrics_coeffs`` are applied so the per-frame
    values land in the same scale (cm / deg / jitter) as the aggregate
    eval columns. ``np.nanmean`` over the returned vector recovers the
    corresponding scalar metric (modulo float ordering).
    """
    if metrics is None:
        metrics = list(per_frame_metric_funcs.keys())
    out = {}
    for name in metrics:
        fn = per_frame_metric_funcs[name]
        v = fn(
            predicted_position,
            predicted_angle,
            predicted_root_angle,
            gt_position,
            gt_angle,
            gt_root_angle,
            upper_index,
            lower_index,
            fps=fps,
            reduce_time=False,
        )
        coeff = metrics_coeffs.get(name, 1.0)
        out[name] = (v * coeff).detach().cpu().numpy().astype(np.float32)
    return out
