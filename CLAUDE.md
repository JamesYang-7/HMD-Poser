# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HMD-Poser is a CVPR 2024 paper implementation for real-time full-body human pose estimation from sparse HMD (Head-Mounted Display) and optional IMU sensor inputs. It predicts SMPL+H body parameters (joint rotations, shape) from up to 6 sensors: head + left/right hand (HMD) + left foot + right foot + pelvis (IMUs).

## Common Commands

### Data Preparation
```bash
python prepare_data.py          # Process AMASS dataset → datasets_processed/
python prepare_data_male_unshaped.py  # Variant for male subjects
```

### Training
```bash
python train.py --config ./options/train_config.yaml
```
Saves checkpoints to `experiments/<task>_<timestamp>/`, where `<task>` comes from the `task` field in the YAML (e.g., `hmd_poser` for the main config, `hmd_poser_temporal_transformer` for that ablation variant).

### Testing / Evaluation
```bash
python test.py --config ./options/test_config.yaml
python test_sliding_window.py --config ./options/test_config.yaml  # batched sliding window
```
Set `resume_model` in `test_config.yaml` to the full path of the trained checkpoint (`.pt` file). Both scripts raise a `ValueError` if `resume_model` is not set.

## Architecture

### Input: 135-dim feature vector per frame
The model processes 8 sensor groups independently before fusing them:
- Head (HMD): rot_6d + rot_vel_6d + position + velocity = 18 dims
- Left/Right hand: same as head = 18 dims each
- Left/Right foot: rot_6d + rot_vel_6d + accel_3d = 15 dims each
- Pelvis: rot_6d + rot_vel_6d + accel_3d = 15 dims
- Left/Right hand in head space (relative): rot + rot_vel + pos + vel = 18 dims each

Zero-masking entire sensor groups simulates missing sensors (HMD-only, HMD+2IMUs, HMD+3IMUs).

### Network: `HMD_imu_HME_Universe` (`model/network.py`)
1. **TemporalSpatialBackbone** (2 blocks by default):
   - MLP per sensor group → `hidden_size=256` embedding
   - Bidirectional LSTM per sensor group (temporal encoding)
   - Transformer encoder across all 8 sensor groups (spatial fusion)
2. **Output heads**: pose (22 joints × 6D = 132D) + shape (16 SMPL betas)

### Model wrapper: `HMDIMUModel` (`model/hmd_imu_model.py`)
Wraps the network with SMPL+H body model. Converts 6D rotation output → rotation matrices → forward kinematics → 3D joint positions.

### Loss: `PoseJointLoss` (`model/loss.py`)
Six weighted L1 terms: root orientation (×1.0), local pose (×5.0), global pose (×1.0), joint position (×1.0), acceleration smoothness (×0.5), shape (×0.1).

### Data: `TrainDataset` / `TestDataset` (`dataset/dataloader.py`)
Training randomly samples `input_motion_length=40` frame windows and creates 3 augmented copies per sample (different sensor modalities). Test uses full sequences.

## Configuration

All hyperparameters live in `options/train_config.yaml` and `options/test_config.yaml`. Key fields:
- `dataset_path`: path to preprocessed `.npz` files
- `sparse_dim: 135`: input feature dimension
- `input_motion_length: 40`: temporal window size
- `model_params`: `hidden_size`, `number_layer` (Transformer layers), `nhead`, `block_num`, `dropout`
- `lr_config`: warm-up, step decay at epoch 300 (×0.1)

## Rotation Representations

All rotations use **6D representation** (first two columns of 3×3 rotation matrix). Utilities in `utils/utils_transform.py`: `aa2sixd`, `sixd2aa`, `sixd2matrot`, `matrot2sixd`. Gram-Schmidt orthogonalization is used for stability.

## Evaluation Metrics (`utils/metrics.py`)

MPJPE (cm), MPJRE (degrees), MPJVE (cm/s), HandPE, UpperPE, LowerPE, RootPE, and Jitter (3rd-order acceleration). Pretrained model (Protocol 1) achieves: HMD-only MPJPE=3.15 cm, HMD+3IMUs MPJPE=2.01 cm.

## Python Environment

Uses **uv** with a `.venv` virtual environment at the project root (Python 3.13).

```bash
uv venv                          # create venv (first time)
source .venv/bin/activate        # activate
uv pip install -r requirements.txt  # install dependencies
```

All dependencies are listed in `requirements.txt`.

## External Dependencies

Requires: `human_body_prior` (VPoser + SMPL+H body model in `body_models/`), AMASS dataset, PyTorch ≥ 2.0, `easydict`, `prettytable`, `tensorboard`, `tqdm`.
