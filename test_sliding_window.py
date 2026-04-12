import torch
from tqdm import tqdm
import argparse
from utils.utils_config import load_config
from dataset.dataloader import load_data
from model.hmd_imu_model import HMDIMUModel
from utils import utils_transform
import math
from utils.metrics import get_metric_function
import prettytable as pt
import os

#####################
RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "pred_jitter",
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "handpe": METERS_TO_CENTIMETERS,
    "upperpe": METERS_TO_CENTIMETERS,
    "lowerpe": METERS_TO_CENTIMETERS,
    "rootpe": METERS_TO_CENTIMETERS,
    "pred_jitter": 1.0,
    "gt_jitter": 1.0,
}

#####################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    args = parser.parse_args()
    return args


def prepare_input_feat(input_feat, input_type):
    """
    Prepare input features based on input type (HMD, HMD_2IMUs, HMD_3IMUs)
    """
    input_feat = input_feat.clone()
    
    if input_type == 'HMD_2IMUs':
        # Zero out pelvis IMU features
        input_feat[:, list(range(30, 36)) + list(range(66, 72)) + list(range(132, 135))] = 0
    elif input_type == 'HMD':
        # Zero out all IMU features (feet and pelvis)
        input_feat[:, list(range(18, 36)) + list(range(54, 72)) + list(range(126, 135))] = 0
    # For HMD_3IMUs, keep all features as is
    
    return input_feat


def sliding_window_inference(model, motion_data, input_type, window_length, device, batch_size=32):
    """
    Perform sliding window inference on a single motion sequence with batched processing.
    
    Args:
        model: The trained HMDIMUModel
        motion_data: Dictionary containing motion data
        input_type: Type of input (HMD, HMD_2IMUs, HMD_3IMUs)
        window_length: Length of the sliding window
        device: torch device
        batch_size: Number of windows to process in parallel
        
    Returns:
        pred_local_pose: Predicted local poses for the entire sequence
        pred_joint_position: Predicted joint positions for the entire sequence
        gt_local_pose: Ground truth local poses
        gt_positions: Ground truth positions
    """
    input_feat = motion_data['input_feat']
    gt_local_pose = motion_data['gt_local_pose'].to(device).float()
    gt_positions = motion_data['gt_positions'].to(device).float()
    
    seq_len = input_feat.shape[0]
    
    # Handle sequences shorter than window length
    if seq_len < window_length:
        # Pad the sequence
        pad_length = window_length - seq_len
        input_feat_padded = torch.cat([
            input_feat,
            input_feat[-1:].repeat(pad_length, 1)
        ], dim=0)
        
        # Prepare input
        window_input = prepare_input_feat(input_feat_padded, input_type)
        window_input = window_input.unsqueeze(0).to(device).float()
        
        # Inference
        with torch.no_grad():
            pred_local_pose, _, _, pred_joint_position = model(window_input)
        
        # Take only valid frames (unpad)
        pred_local_poses = pred_local_pose[0, :seq_len]
        pred_joint_positions = pred_joint_position[0, :seq_len]
        
        return pred_local_poses, pred_joint_positions, gt_local_pose, gt_positions
    
    # Prepare all windows for batch processing
    num_windows = seq_len - window_length + 1
    all_windows = []
    
    for start_idx in range(num_windows):
        end_idx = start_idx + window_length
        window_input = input_feat[start_idx:end_idx]
        window_input = prepare_input_feat(window_input, input_type)
        all_windows.append(window_input)
    
    # Stack all windows: [num_windows, window_length, feature_dim]
    all_windows = torch.stack(all_windows, dim=0)
    
    # Process windows in batches
    pred_local_poses_list = []
    pred_joint_positions_list = []
    
    for batch_start in range(0, num_windows, batch_size):
        batch_end = min(batch_start + batch_size, num_windows)
        batch_windows = all_windows[batch_start:batch_end].to(device).float()
        
        # Inference on batch
        with torch.no_grad():
            pred_local_pose, _, _, pred_joint_position = model(batch_windows)
        
        # Store predictions
        # For the first window, take all frames
        if batch_start == 0:
            # First window: take all frames
            pred_local_poses_list.append(pred_local_pose[0])
            pred_joint_positions_list.append(pred_joint_position[0])
            # Rest of the batch: take only last frame
            if batch_end > 1:
                pred_local_poses_list.append(pred_local_pose[1:, -1])
                pred_joint_positions_list.append(pred_joint_position[1:, -1])
        else:
            # All windows: take only last frame
            pred_local_poses_list.append(pred_local_pose[:, -1])
            pred_joint_positions_list.append(pred_joint_position[:, -1])
    
    # Concatenate all predictions
    pred_local_poses = torch.cat(pred_local_poses_list, dim=0)
    pred_joint_positions = torch.cat(pred_joint_positions_list, dim=0)
    
    return pred_local_poses, pred_joint_positions, gt_local_pose, gt_positions


def main():
    args = parse_args()
    configs = load_config(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading test data with window length: {configs.input_motion_length}")
    batch_size = configs.train_config.batch_size
    test_datas = load_data(configs.dataset_path, "test", 
        input_motion_length=configs.input_motion_length)

    model = HMDIMUModel(configs, device)
    # Load trained model
    with open("out_dir.txt", "r") as f:
        configs.resume_model = f.readline().strip() + "/epoch_with_best_trainloss.pt"
    model.load(configs.resume_model)
    print(f"Successfully resumed checkpoint from {configs.resume_model}")
    model.eval()

    # Print the value for all the metrics
    tb = pt.PrettyTable()
    tb.field_names = ['Input_type'] + pred_metrics + gt_metrics

    for input_type in configs.compatible_inputs:
        print(f"\nTesting on {input_type} with sliding window inference")
        print(f"Window length: {configs.input_motion_length}, Stride: 1, Batch size: {batch_size}")
        
        log = {}
        for metric in all_metrics:
            log[metric] = 0
        
        # Process each motion sequence
        for motion_idx, motion_data in tqdm(enumerate(test_datas), total=len(test_datas)):
            # Perform sliding window inference with batched processing
            pred_local_poses, pred_joint_positions, gt_local_pose, gt_positions = \
                sliding_window_inference(
                    model, 
                    motion_data, 
                    input_type, 
                    configs.input_motion_length, 
                    device,
                    batch_size=batch_size
                )
            
            # Convert predictions to axis-angle
            seq_len = pred_local_poses.shape[0]
            pred_local_pose_aa = utils_transform.sixd2aa(
                pred_local_poses.reshape(-1, 6).detach()
            ).reshape(seq_len, 22*3)
            gt_local_pose_aa = utils_transform.sixd2aa(
                gt_local_pose.reshape(-1, 6).detach()
            ).reshape(seq_len, 22*3)
            
            # Align predicted positions with ground truth (root alignment)
            pred_joint_positions = pred_joint_positions.detach()
            gt_positions = gt_positions.detach()
            pred_joint_positions = pred_joint_positions - pred_joint_positions[:, 15:16] + gt_positions[:, 15:16]
            
            # Extract relevant components for metrics
            predicted_angle = pred_local_pose_aa[..., 3:66]
            predicted_root_angle = pred_local_pose_aa[..., :3]
            predicted_position = pred_joint_positions
            
            gt_angle = gt_local_pose_aa[..., 3:66]
            gt_root_angle = gt_local_pose_aa[..., :3]
            gt_position = gt_positions
            
            upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
            
            # Calculate metrics
            eval_log = {}
            for metric in all_metrics:
                eval_log[metric] = (
                    get_metric_function(metric)(
                        predicted_position,
                        predicted_angle,
                        predicted_root_angle,
                        gt_position,
                        gt_angle,
                        gt_root_angle,
                        upper_index,
                        lower_index,
                        fps=60,
                    )
                    .cpu()
                    .numpy()
                )
            
            # Accumulate metrics
            for key in eval_log:
                log[key] += eval_log[key]

            if configs.save_output:
                output_path = os.path.join(configs.save_dir, f"{motion_idx+1}.pt")
                torch.save({
                    "pred_local_pose": pred_local_poses,
                    "predicted_position": predicted_position,
                    'filepath': motion_data['filepath']
                }, output_path)
        
        # Add row to table with averaged metrics
        tb.add_row([input_type] + 
            ['%.2f' % (log[metric] / len(test_datas) * metrics_coeffs[metric]) for metric in pred_metrics] +
            ['%.2f' % (log[metric] / len(test_datas) * metrics_coeffs[metric]) for metric in gt_metrics]
        )
    
    print("\n" + "="*80)
    print("Sliding Window Test Results")
    print(f"Window Length: {configs.input_motion_length}, Stride: 1")
    print("="*80)
    print(tb)


if __name__ == "__main__":
    main()
