# HMD-Poser: Model Specification

Source: CVPR 2024 — "HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations"

This document is the authoritative technical reference for model construction, training procedure, and evaluation protocol. It is written for agentic consumption: every claim is either directly verifiable in code or cited from the paper/supplementary.

---

## 1. Task Definition

**Goal:** Given sparse sensor observations `x^t` at each timestep `t`, estimate the orientations and positions of all body joints in real time.

**Output representation:** Full-body pose via SMPL [30] using the first J=22 joints.

**Three scalable input scenarios** (same model, different sensors zeroed):

| Scenario | Sensors active | Notes |
|---|---|---|
| HMD | Head (HMD): position + 6DOF | Most convenient, lowest accuracy |
| HMD+2IMUs | HMD + left foot + right foot | Balances friendliness and accuracy |
| HMD+3IMUs | HMD + left foot + right foot + pelvis | Highest accuracy |

Missing sensors are handled by **zero-padding** the corresponding feature slots — the model architecture is identical across scenarios.

---

## 2. Input Feature Vector

**Total dimension: 135**, constructed per frame. All rotations use the **6D representation** (first two columns of 3×3 rotation matrix; [Zhou et al. 2019]).

The 135D vector is assembled in this exact order (indices are 0-based, verified against `network.py:71–78`):

```
Dims  0–5    : head     global rotation 6D
Dims  6–11   : lhand    global rotation 6D
Dims 12–17   : rhand    global rotation 6D
Dims 18–23   : lfoot    global rotation 6D
Dims 24–29   : rfoot    global rotation 6D
Dims 30–35   : pelvis   global rotation 6D
Dims 36–41   : head     global rotation velocity 6D
Dims 42–47   : lhand    global rotation velocity 6D
Dims 48–53   : rhand    global rotation velocity 6D
Dims 54–59   : lfoot    global rotation velocity 6D
Dims 60–65   : rfoot    global rotation velocity 6D
Dims 66–71   : pelvis   global rotation velocity 6D
Dims 72–74   : head     position 3D (world)
Dims 75–77   : lhand    position 3D (world)
Dims 78–80   : rhand    position 3D (world)
Dims 81–83   : head     velocity 3D
Dims 84–86   : lhand    velocity 3D
Dims 87–89   : rhand    velocity 3D
Dims 90–95   : lhand    rotation 6D  (in head space)
Dims 96–101  : rhand    rotation 6D  (in head space)
Dims 102–107 : lhand    rotation velocity 6D (in head space)
Dims 108–113 : rhand    rotation velocity 6D (in head space)
Dims 114–116 : lhand    position 3D (in head space)
Dims 117–119 : rhand    position 3D (in head space)
Dims 120–122 : lhand    velocity 3D (in head space)
Dims 123–125 : rhand    velocity 3D (in head space)
Dims 126–128 : lfoot    linear acceleration 3D
Dims 129–131 : rfoot    linear acceleration 3D
Dims 132–134 : pelvis   linear acceleration 3D
```

**Key design choices:**
- HMD (head + hands) uses position and velocity because HMD provides reliable positional tracking.
- Feet and pelvis use IMU-style data: rotation + rotation velocity + linear acceleration (no absolute position).
- Hand positions in head-local space (`dims 114–125`) are critical for H-PE (hand position error); ablation Table 3 shows removing them degrades H-PE significantly.
- All IMU rotation and acceleration data are calibrated to a body-centric frame before storage.

### Sensor-Masking for Multi-Scenario Training

`dataset/dataloader.py:81–88` — during both training and testing, scenarios are created by zeroing specific index ranges of the full 135D feature:

| Scenario | Zeroed indices |
|---|---|
| HMD+3IMUs | None (full 135D) |
| HMD+2IMUs | `30–35, 66–71, 132–134` (pelvis rotation, rotation-vel, acceleration) |
| HMD | `18–35, 54–71, 126–134` (both feet + pelvis: all rotation, rotation-vel, acceleration) |

Each training sample is replicated 3× (one per scenario) before batching. The LSTM hidden state starts fresh each forward call — no recurrent state is carried across batches.

---

## 3. Network Architecture

### 3.1 Top-Level Module: `HMD_imu_HME_Universe` (`model/network.py:133`)

```
Input:  x ∈ R^{B × T × 135}
           │
           ▼
    TemporalSpatialBackbone   →   features ∈ R^{B × T × (8 × hidden_size)}
           │
     ┌─────┴─────┐
     ▼           ▼
PoseHead      ShapeHead
(132D out)    (16D out)
```

Output of the full network: `pred_pose ∈ R^{B×T×132}`, `pred_shapes ∈ R^{B×T×16}`.

### 3.2 Feature Embedding (per sensor group)

The 135D input is split into **8 sensor groups**, each embedded independently by group-specific shallow MLPs (`Linear → LeakyReLU`). All sub-features within a group are embedded separately, then concatenated to form a `hidden_size`-dim embedding per group.

| Group | Sub-features fed to MLPs | Embedding split |
|---|---|---|
| head | rot(6), rot_vel(6), pos(3), vel(3) | 4 × hidden/4 = hidden |
| lhand | rot(6), rot_vel(6), pos(3), vel(3) | 4 × hidden/4 = hidden |
| rhand | rot(6), rot_vel(6), pos(3), vel(3) | 4 × hidden/4 = hidden |
| lfoot | rot(6), rot_vel(6), accel(3) | (hidden-16)/2, (hidden-16)/2, 16 |
| rfoot | rot(6), rot_vel(6), accel(3) | (hidden-16)/2, (hidden-16)/2, 16 |
| pelvis | rot(6), rot_vel(6), accel(3) | (hidden-16)/2, (hidden-16)/2, 16 |
| lhand_inhead | rot(6), rot_vel(6), pos(3), vel(3) | 4 × hidden/4 = hidden |
| rhand_inhead | rot(6), rot_vel(6), pos(3), vel(3) | 4 × hidden/4 = hidden |

After per-group embedding, LayerNorm is applied. Output: 8 tensors, each `∈ R^{B×T×hidden_size}`, stacked as `∈ R^{B×T×8×hidden_size}`.

### 3.3 Lightweight Temporal-Spatial Feature Learning (TSFL)

`N = block_num` identical blocks, each with two sub-stages applied in sequence:

**Stage A — Temporal encoding (per sensor):**
- 8 independent **unidirectional single-layer LSTMs** (one per sensor group)
- LSTM hidden size = `hidden_size` (same as embedding dim)
- LSTM weights are orthogonally initialized; weight-norm is applied to `weight_ih` and `weight_hh`
- Processes shape `(B, T, hidden_size)` → outputs same shape, capturing temporal context

**Stage B — Spatial encoding (across sensors):**
- A **3-layer Transformer Encoder** with `nhead` attention heads
- Feed-forward dim = `hidden_size`
- Input is the 8 sensor tokens per timestep: shape reshaped to `(B×T, 8, hidden_size)`
- Models cross-sensor correlations at each timestep

After `N` blocks: output `∈ R^{B×T×8×hidden_size}` is flattened along the sensor dim → `R^{B×T×(8×hidden_size)}`.

**Design rationale (from paper):** Prior Transformer-only methods (AvatarJLM) had quadratic cost in sequence length. TSFL decouples temporal (LSTM, linear in T) and spatial (Transformer over 8 fixed tokens, constant cost in T), achieving both accuracy and real-time performance.

### 3.4 Output Regression Heads

Both heads are 2-layer MLPs: `Linear(8×hidden_size, 256) → LeakyReLU → Linear(256, out_dim)`.

| Head | Output dim | Content |
|---|---|---|
| PoseHead | 22 × 6 = 132 | Local joint rotations in 6D (root at `[:6]`, body joints at `[6:]`) |
| ShapeHead | 16 | SMPL body shape betas (overridden to zero at inference; kept for shape loss during training) |

### 3.5 Forward Kinematics Module (`model/hmd_imu_model.py:70`)

Wraps the network output in the body model pipeline:

1. `pred_pose[..., :6]` → root orientation (6D → axis-angle)
2. `pred_pose[..., 6:]` → body joint rotations (21 joints, 6D → axis-angle)
3. `pred_shapes` → **zeroed** at inference (`torch.zeros_like(pred_shapes)`, line 125) — default body shape
4. SMPL+H body model (`BodyModel`) with `num_betas=16`, `num_dmpls=8` runs forward kinematics
5. Output: `Jtr[:, :22]` — 22 joint positions in world frame (meters)
6. Local rotations → global rotations via recursive kinematic-tree multiplication (`forward_kinematics_R`)

**Head-centering at training and evaluation:**
```python
pred_joint_position_head_centered = pred_joint_position - pred_joint_position[:, :, 15:16] + gt_positions[:, :, 15:16]
```
Joint index 15 = head joint. Predicted positions are shifted so the predicted head aligns with the GT head, eliminating accumulated root translation error.

### 3.6 Body Model

- Model: SMPL+H male, loaded from `body_models/smplh/male/model.npz`
- DMPL: `body_models/dmpls/male/model.npz`, `num_dmpls=8`
- Kinematic tree: `bm.kintree_table[0][:22]` — standard SMPL 22-joint hierarchy
- Used only for FK at inference and FK-based position loss at training; not trained end-to-end

---

## 4. Loss Function

### 4.1 Formula

`model/loss.py` — `PoseJointLoss.forward()`.

All component losses use **L1 loss** (MAE). Total loss:

```
L_total = w_root  * L1(pred_root_6d,     gt_root_6d)
        + w_local * L1(pred_local_6d,    gt_local_6d)        ← highest weight
        + w_grot  * L1(pred_global_6d,   gt_global_6d)
        + w_joint * L1(pred_positions,   gt_positions)        ← head-centered
        + w_acc   * L1(accel_pred,       accel_gt)
        + w_shape * L1(pred_betas,       mean_betas_per_seq)  ← temporal consistency
```

**Shape loss special form** (line 51): target is `pred_shape.mean(dim=1, keepdim=True)` replicated across time — penalizes intra-sequence shape variation, not deviation from any external prior.

**Acceleration loss** (lines 53–55): second-order finite difference of joint positions:
```
accel[t] = pos[t-1] - 2*pos[t] + pos[t+1]
```
Applied over the interior of the window (T-2 frames).

### 4.2 Loss Weights (from `train_config.yaml` and paper)

| Term | Config key | Weight | Justification |
|---|---|---|---|
| Root orientation | `root_orientation_weight` | 1.0 | Global alignment |
| Local pose | `local_pose_weight` | **5.0** | Primary tracking target, highest weight |
| Global pose | `global_pose_weight` | 1.0 | Consistency with FK |
| Joint position | `joint_position_weight` | 1.0 | 3D accuracy |
| Acceleration (smooth) | `smooth_loss_weight` | 0.5 | Temporal smoothness |
| Shape | `shape_loss_weight` | 0.1 | Temporal consistency |

**Ablation findings (paper Table E):**
- Removing `L_grot`: MPJPE 3.25 → 3.19 (marginal), but MPJVE and Jitter increase
- Removing `L_smooth`: MPJVE 17.47 → 17.89, Jitter 6.07 → 6.71 — smooth loss is essential for motion quality

---

## 5. Training Procedure

### 5.1 Data

- **Dataset:** AMASS (preprocessed to `.pt` files in `datasets_processed/`)
- **Protocol 1** (primary): CMU, BioMotionLab_NTroje, MPI_HDM05 — 90% train / 10% test
- **Protocol 2**: 12 subsets for training + HumanEva + Transition [31] for test
- **Framerate:** 60 Hz (120 Hz AMASS data downsampled 2×)
- **Input shapes are set to ground-truth body parameters** (not zeroed) during data preprocessing

### 5.2 DataLoader

`dataset/dataloader.py` — `TrainDataset`:
- Each `__getitem__` samples a **random start index** in `[0, seqlen - T]`, returns window of length `T = input_motion_length = 40`
- Returns `(input_feat_res, gt_local_pose, gt_global_pose, gt_positions, gt_betas)` where dim-0 = 3 (one per modality scenario)
- Training loop flattens this: effective batch size = `batch_size × 3`
- `train_dataset_repeat_times = 10`: each epoch sees 10× more "virtual" samples

`TestDataset`: returns full sequences (no windowing), same 3-modality structure.

### 5.3 Optimizer and LR Schedule

**Optimizer:** Adam — `lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0, amsgrad=False`

**LR schedule** (`utils/utils_config.py`):
- Warm-up: epochs 0–1, lr linearly ramps from `lr × lr_warm_up_factor = 1e-6` to `1e-3`
- Constant: epochs 1–300
- Decay: at epoch 300, lr decays from `1e-3` to `1e-4` over `lr_decay_epochs = 3` epochs
- Scheduler steps **per batch** (not per epoch) via `lr_scheduler.step()` after each optimizer step

### 5.4 Training Loop Summary

`runner/training_loop.py`:

```
for epoch in 0..399:
    model.train()
    for batch in train_loader:
        flatten modality dim into batch dim
        forward → pred_pose, pred_shapes, rotation_global_r6d, pred_joint_position
        head-center joint positions (joint 15 = head)
        compute 6-component loss
        backward + optimizer.step() + lr_scheduler.step()
    
    log epoch-average losses to TensorBoard (train_epoch_loss/*)
    save checkpoint if best train loss so far → epoch_with_best_trainloss.pt

    if epoch % val_interval == 0:
        model.eval(), no_grad
        for batch in test_loader:
            same forward, same loss
            compute MPJPE (cm) and MPJRE (degrees) on-the-fly
        
        log to TensorBoard (val_epoch_loss/*)
        save checkpoint if best test loss → epoch_with_best_testloss.pt
        track best MPJPE and MPJRE separately (logged, not used for checkpoint)
```

**Checkpoints** format: `{'state_dict': ..., 'epoch': epoch}` saved via `model.save()`.

### 5.5 Key Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 400 |
| Batch size | 128 (×3 modalities = 384 effective) |
| Window length T | 40 frames |
| Validation interval | every 10 epochs |
| Workers | 4 |
| hidden_size | 256 |
| block_num N | 2 |
| Transformer layers | 3 |
| Attention heads | 8 |
| LSTM layers | 1 |
| Dropout | 0.05 |

**Ablation on N (block_num), paper Table D:**

| N | MPJRE | MPJPE | MPJVE | Jitter |
|---|---|---|---|---|
| 1 | 2.40 | 3.31 | 22.85 | 11.77 |
| **2** | **2.28** | **3.19** | **17.47** | **6.07** |
| 3 | 2.28 | 3.18 | 16.97 | 5.20 |

N=2 is the chosen default: N=3 gives marginal MPJPE gain but adds computation. The benefit of increasing N diminishes past 2.

---

## 6. Evaluation Protocol

### 6.1 Metrics

All metrics computed on **full sequences** (no windowing), **head-centered** joint positions.

| Metric | Unit | Formula | SMPL joints |
|---|---|---|---|
| MPJRE | degrees | Mean L1 of per-joint axis-angle diff (wrapped to [-π, π]) | all 22 |
| MPJPE | cm | Mean Euclidean distance across all joints and frames | all 22 |
| MPJVE | cm/s | Mean Euclidean distance of per-joint velocity (×fps) | all 22 |
| H-PE | cm | MPJPE restricted to joints [20, 21] (hands) | 20, 21 |
| U-PE | cm | MPJPE restricted to upper body joint indices | upper_index |
| L-PE | cm | MPJPE restricted to lower body joint indices | lower_index |
| R-PE | cm | MPJPE restricted to joint [0] (root/pelvis) | 0 |
| Jitter | 10^m/s³ | Mean norm of 3rd-order finite difference of position × fps³ | all 22 |

MPJRE wraps angle differences: `diff[diff > π] -= 2π`, `diff[diff < -π] += 2π`.

Jitter formula: `|pos[t+3] - 3·pos[t+2] + 3·pos[t+1] - pos[t]| × fps³`.

### 6.2 Reference Results (Protocol 1, AMASS)

| Scenario | MPJRE↓ | MPJPE↓ | MPJVE↓ | Jitter↓ | H-PE↓ | U-PE↓ | L-PE↓ | R-PE↓ |
|---|---|---|---|---|---|---|---|---|
| HMD | 2.28 | 3.19 | 17.47 | 6.07 | 1.65 | 1.67 | 5.40 | 3.02 |
| HMD+2IMUs | 1.83 | 2.27 | 13.28 | 5.96 | 1.39 | 1.51 | 3.35 | 2.74 |
| HMD+3IMUs | **1.73** | **1.89** | **11.03** | **5.35** | **1.27** | **1.46** | **2.46** | **2.37** |

### 6.3 Inference Speed

- GPU (RTX 3080): **205.7 FPS**
- PICO 4 HMD (on-device): **90.0 FPS**
- Target operating frequency: **60 Hz** (aligned with AMASS training framerate)

---

## 7. Real-Captured Data (PICO FreeDancing Dataset)

74 sequences, 8 subjects (3 male, 5 female), each 60–120s, HMD+2IMUs scenario.

**Setup:** PICO 4 HMD + 2 PICO motion trackers (lower legs). Ground truth via OptiTrack at 120 Hz + Mosh++ for SMPL parameters.

**Calibration:** Subject performs 3 prescribed motions (stand still 5s → bend knees → lift legs) to compute IMU-to-body transform matrices automatically.

**Synchronization:** Rigid body on HMD headset; common motion sequences between HMD and OptiTrack used for temporal alignment. All data downsampled to 60 Hz.

**Synthetic-vs-real gap (Table 5):**

| Mode | MPJPE | H-PE | Notes |
|---|---|---|---|
| Online (real sensor) | 6.55 | 8.10 | Real calibration errors present |
| Offline (real sensor) | 6.53 | 8.01 | Same real data, offline processing |
| Offline (synthetic GT) | 5.22 | 2.09 | Upper bound with perfect GT |

Large H-PE gap is attributed to hand controller rigidity (controllers move relative to the hand) causing calibration drift.

---

## 8. Ablation Summary

All ablations on Protocol 1, HMD scenario unless noted.

**Effect of hand-in-head representation (Table 3):**

| Config | MPJRE | MPJPE | H-PE | Jitter |
|---|---|---|---|---|
| w/o `x_lh/h, x_rh/h` | 2.45 | 3.43 | 2.36 | 6.25 |
| with `x_lh/h, x_rh/h` | **2.28** | **3.19** | **1.65** | **6.07** |

H-PE improves by 30% — relative hand pose in head frame is the key input for accurate hand tracking.

**Effect of shape regression head (Table 4):**

| Config | MPJRE | MPJPE | H-PE | Jitter |
|---|---|---|---|---|
| w/o ShapeHead | 2.32 | 5.08 | 4.25 | 6.11 |
| with ShapeHead | **2.28** | **3.19** | **1.65** | **6.07** |

MPJPE drops from 5.08 to 3.19 — shape estimation is essential because joint positions depend on per-subject bone lengths.

**Effect of loss terms (Table E):**

| Config | MPJRE | MPJPE | MPJVE | Jitter |
|---|---|---|---|---|
| w/o `L_grot` | 2.29 | 3.25 | 17.76 | 6.06 |
| w/o `L_smooth` | 2.27 | 3.16 | 17.89 | 6.71 |
| all terms | **2.28** | **3.19** | **17.47** | **6.07** |

`L_smooth` is the primary driver of low Jitter. `L_grot` has marginal impact on MPJPE but helps global consistency.
