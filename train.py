#!/usr/bin/env python3
"""
Training script for PointNet binary classification.
Trains on good vs bad sheet pairs using XYZ point clouds.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    TENSORBOARD_AVAILABLE = False
    print(f"Warning: TensorBoard not available. Logging disabled. Error: {type(e).__name__}: {str(e)[:100]}")
from tqdm import tqdm
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import our modules
from dataloader import get_pointnet_dataloaders

# Import PointNet2 from Pointnet_Pointnet2_pytorch
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_DIR = os.path.join(BASE_DIR, 'Pointnet_Pointnet2_pytorch')
sys.path.append(os.path.join(POINTNET_DIR, 'models'))
from Pointnet_Pointnet2_pytorch.models.pointnet2_cls_ssg import get_model, get_loss


def get_next_numbered_folder(base_dir: str, prefix: str) -> str:
    """
    Get the next available numbered folder name.
    
    Args:
        base_dir: Base directory (e.g., 'checkpoints' or 'logs')
        prefix: Folder prefix (e.g., 'pt_' or 'pointnet_')
    
    Returns:
        Path to the next available numbered folder (e.g., 'checkpoints/pt_1')
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Find existing numbered folders
    existing_numbers = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith(prefix):
                # Extract number from folder name (e.g., 'pt_1' -> 1)
                suffix = item[len(prefix):]
                try:
                    num = int(suffix)
                    existing_numbers.append(num)
                except ValueError:
                    continue
    
    # Find next available number
    if existing_numbers:
        next_num = max(existing_numbers) + 1
    else:
        next_num = 1
    
    folder_name = f"{prefix}{next_num}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path


def calculate_weighted_accuracy(y_true, y_pred):
    """
    Calculate weighted accuracy: (0.5 * Precision_Good) + (0.5 * Precision_Bad)
    
    Where Good is treated as positive class:
    - TP = Predicted Good (0) and actual Good (0) = 700
    - FP = Predicted Good (0) and actual Bad (1) = 341
    - FN = Predicted Bad (1) and actual Good (0) = 0
    - TN = Predicted Bad (1) and actual Bad (1) = 59
    
    Precision_Good = TP/(TP+FP)
    Precision_Bad = TN/(TN+FN)
    
    Args:
        y_true: True labels (0 for Good, 1 for Bad)
        y_pred: Predicted labels (0 for Good, 1 for Bad)
    
    Returns:
        weighted_accuracy: Weighted accuracy score (0.0 to 1.0)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Good (0) is positive class
    TP = np.sum((y_pred == 0) & (y_true == 0))  # Predicted Good, actual Good
    FP = np.sum((y_pred == 0) & (y_true == 1))  # Predicted Good, actual Bad
    TN = np.sum((y_pred == 1) & (y_true == 1))  # Predicted Bad, actual Bad
    FN = np.sum((y_pred == 1) & (y_true == 0))  # Predicted Bad, actual Good
    
    precision_good = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    precision_bad = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    
    print(f"Precision Good: {precision_good}, Precision Bad: {precision_bad}")
    weighted_accuracy = (0.5 * precision_good) + (0.5 * precision_bad)
    
    return float(weighted_accuracy)


def rotate_point_cloud(batch_data):
    """Randomly rotate the point clouds to augment the dataset.
    Rotation is per shape based along up direction.
    
    Args:
        batch_data: (B, N, 3) array, original batch of point clouds
    
    Returns:
        (B, N, 3) array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def train_one_epoch(model, train_loader, criterion, optimizer, device, 
                    grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (points, labels) in enumerate(tqdm(train_loader, desc='Training')):
        # Move to device
        points = points.to(device)  # (B, N, 3)
        labels = labels.to(device)  # (B,)
        
        # Note: Data augmentation is now handled in the dataloader transform
        
        # PointNet2 expects input in (B, D, N) format
        # Transpose from (B, N, 3) to (B, 3, N)
        points = points.transpose(2, 1)
        
        # Forward pass
        optimizer.zero_grad()
        # PointNet2 returns (log_softmax, feature_points)
        pred, feature_points = model(points)
        
        # Calculate loss
        # get_loss expects (pred, target, trans_feat) where pred is log_softmax
        # PointNet2 doesn't use trans_feat for regularization, but we pass feature_points for compatibility
        loss = criterion(pred, labels, feature_points)
        
        loss_value = loss.item()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        # Statistics
        # pred is log_softmax, so we need to get argmax
        pred_choice = pred.data.max(1)[1]
        all_preds.extend(pred_choice.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        total_loss += loss_value
        
        # Debug: Print prediction statistics for first batch to diagnose collapse
        if len(all_preds) <= labels.size(0):  # Only for first batch
            # Convert log_softmax to probabilities
            probs = torch.exp(pred)  # (B, 2)
            pred_probs_0 = probs[:, 0].mean().item()  # Average prob of class 0 (Good)
            pred_probs_1 = probs[:, 1].mean().item()  # Average prob of class 1 (Bad)
            logit_diff = (pred[:, 1] - pred[:, 0]).mean().item()  # Average logit difference
            # Check input data statistics
            points_mean = points.mean().item()
            points_std = points.std().item()
            points_min = points.min().item()
            points_max = points.max().item()
            # Check if points are centered (should be near 0 if preprocessing is correct)
            points_centroid = points.mean(dim=2).mean(dim=0)  # Average over batch and points
            centroid_norm = torch.norm(points_centroid).item()
            print(f"\n[DEBUG] First train batch prediction stats:")
            print(f"  Average prob(Good): {pred_probs_0:.4f}, Average prob(Bad): {pred_probs_1:.4f}")
            print(f"  Average logit diff (Bad - Good): {logit_diff:.4f}")
            print(f"  Input data stats: mean={points_mean:.4f}, std={points_std:.4f}, min={points_min:.4f}, max={points_max:.4f}")
            print(f"  Input centroid norm: {centroid_norm:.4f} (should be ~0 if centered)")
            print(f"  Predictions: Good={torch.sum(pred_choice == 0).item()}, Bad={torch.sum(pred_choice == 1).item()}")
            print(f"  Ground truth: Good={torch.sum(labels == 0).item()}, Bad={torch.sum(labels == 1).item()}")
            if points_std > 10.0:
                print(f"  [WARNING] Large std ({points_std:.2f}) suggests data may not be normalized!")
    
    avg_loss = total_loss / len(train_loader)
    # Convert to numpy arrays for accuracy calculation
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    
    # Calculate confusion matrix components (Good is positive class)
    TP = np.sum((all_preds_np == 0) & (all_labels_np == 0))
    FP = np.sum((all_preds_np == 0) & (all_labels_np == 1))
    TN = np.sum((all_preds_np == 1) & (all_labels_np == 1))
    FN = np.sum((all_preds_np == 1) & (all_labels_np == 0))
    
    # Calculate balanced accuracy
    recall_good = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    recall_bad = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_accuracy = (recall_good + recall_bad) / 2.0
    
    avg_loss = float(avg_loss)
    balanced_accuracy = float(balanced_accuracy)
    return avg_loss, balanced_accuracy, all_preds, all_labels


def evaluate_confusion_matrix(all_labels, all_preds, set_name="Set"):
    """
    Print confusion matrix and metrics for given predictions and labels.
    
    Args:
        all_labels: True labels (numpy array or list)
        all_preds: Predicted labels (numpy array or list)
        set_name: Name of the set (e.g., "Train", "Test")
    
    Returns:
        Dictionary with metrics
    """
    # Convert to numpy arrays if needed
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Ensure they are 1D arrays
    all_labels = all_labels.flatten()
    all_preds = all_preds.flatten()
    
    # Get total number of samples
    total_samples = all_labels.shape[0] if hasattr(all_labels, 'shape') else len(all_labels)
    
    # Calculate confusion matrix components (Good is positive class)
    TP = np.sum((all_preds == 0) & (all_labels == 0))  # Predicted Good, actual Good
    FP = np.sum((all_preds == 0) & (all_labels == 1))  # Predicted Good, actual Bad
    TN = np.sum((all_preds == 1) & (all_labels == 1))  # Predicted Bad, actual Bad
    FN = np.sum((all_preds == 1) & (all_labels == 0))  # Predicted Bad, actual Good
    
    # Print confusion matrix
    print(f"\n  {set_name} Confusion Matrix:")
    print(f"  {'':>15} {'Predicted Good':>20} {'Predicted Bad':>20}")
    print(f"  {'Actual Good':>15} {TP:>20} {FN:>20}")
    print(f"  {'Actual Bad':>15} {FP:>20} {TN:>20}")
    print(f"  {'':>15} {'':>20} {'':>20}")
    print(f"  Total samples: {total_samples} (Good: {TP+FN}, Bad: {FP+TN})")
    print(f"  Predictions: Good={TP+FP}, Bad={FN+TN}")
    
    # Calculate and print precision, recall, F1 for each class (Good is positive)
    precision_good = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_good = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_good = 2 * (precision_good * recall_good) / (precision_good + recall_good) if (precision_good + recall_good) > 0 else 0.0
    
    precision_bad = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_bad = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_bad = 2 * (precision_bad * recall_bad) / (precision_bad + recall_bad) if (precision_bad + recall_bad) > 0 else 0.0
    
    # Calculate standard accuracy and balanced accuracy
    standard_accuracy = (TP + TN) / total_samples if total_samples > 0 else 0.0
    balanced_accuracy = (recall_good + recall_bad) / 2.0  # Average of per-class recall
    weighted_accuracy = (0.5 * precision_good) + (0.5 * precision_bad)
    
    print(f"\n  {set_name} Metrics:")
    print(f"  {'Class':>15} {'Precision':>15} {'Recall':>15} {'F1-Score':>15}")
    print(f"  {'Good (0)':>15} {precision_good:>15.4f} {recall_good:>15.4f} {f1_good:>15.4f}")
    print(f"  {'Bad (1)':>15} {precision_bad:>15.4f} {recall_bad:>15.4f} {f1_bad:>15.4f}")
    print(f"  {'':>15} {'':>15} {'':>15} {'':>15}")
    print(f"  {'Standard Acc':>15} {standard_accuracy:>15.4f} ((TP+TN) / Total)")
    print(f"  {'Balanced Acc':>15} {balanced_accuracy:>15.4f} ((Recall_Good + Recall_Bad) / 2)")
    print(f"  {'Weighted Acc':>15} {weighted_accuracy:>15.4f} (0.5 * Precision_Good + 0.5 * Precision_Bad)")
    
    return {
        'standard_accuracy': standard_accuracy,
        'balanced_accuracy': balanced_accuracy,
        'weighted_accuracy': weighted_accuracy,
        'precision_good': precision_good,
        'precision_bad': precision_bad,
        'recall_good': recall_good,
        'recall_bad': recall_bad
    }


def evaluate(model, test_loader, criterion, device, return_predictions=False):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for points, labels in tqdm(test_loader, desc='Evaluating'):
            # Move to device
            points = points.to(device)  # (B, N, 3)
            labels = labels.to(device)
            
            # PointNet2 expects input in (B, D, N) format
            # Transpose from (B, N, 3) to (B, 3, N)
            points = points.transpose(2, 1)
            
            # Forward pass
            # PointNet2 returns (log_softmax, feature_points)
            pred, feature_points = model(points)
            
            # Calculate loss
            # get_loss expects (pred, target, trans_feat) where pred is log_softmax
            # PointNet2 doesn't use trans_feat for regularization, but we pass feature_points for compatibility
            loss = criterion(pred, labels, feature_points)
            
            # Statistics
            # pred is log_softmax, so we need to get argmax
            pred_choice = pred.data.max(1)[1]
            correct = (pred_choice == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            loss_value = loss.item()
            total_loss += loss_value
            
            all_preds.extend(pred_choice.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            # Debug: Print prediction statistics for first batch to diagnose collapse
            if len(all_preds) <= labels.size(0):  # Only for first batch
                # Convert log_softmax to probabilities
                probs = torch.exp(pred)  # (B, 2)
                pred_probs_0 = probs[:, 0].mean().item()  # Average prob of class 0 (Good)
                pred_probs_1 = probs[:, 1].mean().item()  # Average prob of class 1 (Bad)
                logit_diff = (pred[:, 1] - pred[:, 0]).mean().item()  # Average logit difference
                logit_std = (pred[:, 1] - pred[:, 0]).std().item()  # Std of logit differences
                # Check actual log_softmax values
                log_softmax_0 = pred[:, 0].mean().item()
                log_softmax_1 = pred[:, 1].mean().item()
                # Check input data statistics - CRITICAL for diagnosing distribution mismatch
                points_mean = points.mean().item()
                points_std = points.std().item()
                points_min = points.min().item()
                points_max = points.max().item()
                # Check if points are centered (should be near 0 if preprocessing is correct)
                points_centroid = points.mean(dim=2).mean(dim=0)  # Average over batch and points
                centroid_norm = torch.norm(points_centroid).item()
                print(f"\n[DEBUG] First test batch prediction stats:")
                print(f"  Average prob(Good): {pred_probs_0:.4f}, Average prob(Bad): {pred_probs_1:.4f}")
                print(f"  Average log_softmax: Good={log_softmax_0:.4f}, Bad={log_softmax_1:.4f}")
                print(f"  Average logit diff (Bad - Good): {logit_diff:.4f} (std: {logit_std:.4f})")
                print(f"  Input data stats: mean={points_mean:.4f}, std={points_std:.4f}, min={points_min:.4f}, max={points_max:.4f}")
                print(f"  Input centroid norm: {centroid_norm:.4f} (should be ~0 if centered)")
                print(f"  Predictions: Good={torch.sum(pred_choice == 0).item()}, Bad={torch.sum(pred_choice == 1).item()}")
                print(f"  Ground truth: Good={torch.sum(labels == 0).item()}, Bad={torch.sum(labels == 1).item()}")
                if abs(logit_diff) > 5.0:
                    print(f"  [WARNING] Extreme logit diff ({logit_diff:.2f}) suggests:")
                    print(f"    - Model is overconfident in wrong predictions")
                    print(f"    - Possible data distribution mismatch between train/test")
                    print(f"    - Check if test data preprocessing matches training (centering, normalization, PCA)")
                if centroid_norm > 1.0:
                    print(f"    - Test data centroid norm ({centroid_norm:.2f}) suggests data may not be centered!")
                if points_std > 10.0:
                    print(f"    - Test data std ({points_std:.2f}) suggests data may not be normalized!")
    
    avg_loss = total_loss / len(test_loader)
    
    # Convert to numpy arrays for calculations
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    
    # Calculate balanced accuracy (Good is positive class)
    TP = np.sum((all_preds_np == 0) & (all_labels_np == 0))
    FP = np.sum((all_preds_np == 0) & (all_labels_np == 1))
    TN = np.sum((all_preds_np == 1) & (all_labels_np == 1))
    FN = np.sum((all_preds_np == 1) & (all_labels_np == 0))
    
    recall_good = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    recall_bad = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_accuracy = (recall_good + recall_bad) / 2.0
    accuracy = float(balanced_accuracy)
    
    # Calculate per-class accuracy (for reporting)
    class_acc = []
    for cls in range(2):
        mask = all_labels_np == cls
        if mask.sum() > 0:
            cls_acc = (all_preds_np[mask] == cls).sum() / mask.sum()
            class_acc.append(float(cls_acc))
        else:
            class_acc.append(0.0)
    
    # Calculate standard accuracy and balanced accuracy (Good is positive class)
    TP = np.sum((all_preds_np == 0) & (all_labels_np == 0))  # Predicted Good, actual Good
    FP = np.sum((all_preds_np == 0) & (all_labels_np == 1))  # Predicted Good, actual Bad
    TN = np.sum((all_preds_np == 1) & (all_labels_np == 1))  # Predicted Bad, actual Bad
    FN = np.sum((all_preds_np == 1) & (all_labels_np == 0))  # Predicted Bad, actual Good
    
    precision_good = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_good = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision_bad = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_bad = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    standard_accuracy = (TP + TN) / len(all_labels_np) if len(all_labels_np) > 0 else 0.0
    balanced_accuracy = (recall_good + recall_bad) / 2.0  # Average of per-class recall
    
    # Convert to Python scalars for TensorBoard compatibility
    standard_accuracy = float(standard_accuracy)
    balanced_accuracy = float(balanced_accuracy)
    
    if return_predictions:
        # Return lists (not numpy arrays) for easier handling
        # Ensure all numeric values are Python floats
        return (float(avg_loss), float(accuracy), class_acc, 
                all_preds, all_labels, float(standard_accuracy), float(balanced_accuracy))
    return (float(avg_loss), float(accuracy), class_acc, 
            float(standard_accuracy), float(balanced_accuracy))


def plot_loss_curves(train_losses, test_losses, train_accs, test_accs, save_path):
    """Plot and save loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(test_losses, label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves (balanced accuracy)
    axes[1].plot(train_accs, label='Train Balanced Acc', linewidth=2)
    axes[1].plot(test_accs, label='Test Balanced Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Balanced Accuracy', fontsize=12)
    axes[1].set_title('Training and Test Balanced Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['Good', 'Bad']):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
    
    # Print confusion matrix details
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              {class_names[0]:>6}  {class_names[1]:>6}")
    print(f"True {class_names[0]:>6}  {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"     {class_names[1]:>6}  {cm[1,0]:>6}  {cm[1,1]:>6}")
    print(f"\nNormalized:")
    print(f"                Predicted")
    print(f"              {class_names[0]:>6}  {class_names[1]:>6}")
    print(f"True {class_names[0]:>6}  {cm_normalized[0,0]:>6.2%}  {cm_normalized[0,1]:>6.2%}")
    print(f"     {class_names[1]:>6}  {cm_normalized[1,0]:>6.2%}  {cm_normalized[1,1]:>6.2%}")


def main():
    parser = argparse.ArgumentParser(description='Train PointNet for binary classification')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size [default: 32]')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs [default: 250]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate [default: 0.0001]')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer [default: adam]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD [default: 0.9]')
    parser.add_argument('--decay_step', type=int, default=50000, help='Decay step for LR decay [default: 50000]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for LR decay [default: 0.7]')
    # PointNet2 doesn't use feature transform regularization
    parser.add_argument('--bad_class_boost', type=float, default=3.0,
                       help='Multiplier for Bad class weight [default: 3.0]')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping max norm [default: 1.0]')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory [default: logs]')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint save directory [default: checkpoints]')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable all data augmentation')
    parser.add_argument('--use_rotation', action='store_true', help='Enable random rotation augmentation')
    parser.add_argument('--rotation_range', type=float, default=2.0 * np.pi,
                       help='Maximum rotation angle in radians [default: 2π]')
    parser.add_argument('--use_point_dropout', action='store_true', help='Enable point dropout augmentation')
    parser.add_argument('--point_dropout_prob', type=float, default=0.1,
                       help='Probability of dropping each point during training (0.0 to 1.0) [default: 0.1]')
    parser.add_argument('--min_points', type=int, default=512,
                       help='Minimum number of points to keep after dropout [default: 512]')
    parser.add_argument('--use_pca_alignment', action='store_true',
                       help='Enable PCA alignment for patch pairs (aligns both patches to same orientation)')
    parser.add_argument('--pca_target_plane', type=str, default='xy', choices=['xy', 'xz', 'yz'],
                       help='Target plane for PCA alignment [default: xy]')
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'nll'],
                       help='Loss function type: bce (Binary Cross Entropy) or nll (Negative Log Likelihood) [default: bce]')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto) [default: auto]')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers [default: 4]')
    parser.add_argument('--seed', type=int, default=42, help='Random seed [default: 42]')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint [default: None]')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create numbered directories for checkpoints and logs
    checkpoint_dir = get_next_numbered_folder(args.save_dir, 'pt_')
    run_log_dir = get_next_numbered_folder(args.log_dir, 'pointnet_')
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Log directory: {run_log_dir}")
    
    # TensorBoard writer
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(run_log_dir)
    else:
        writer = None
    
    # Save arguments
    with open(os.path.join(run_log_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
    
    # Load datasets
    print("Loading datasets...")
    # Determine which augmentations to use
    use_rotation = args.use_rotation and not args.no_augmentation
    use_point_dropout = args.use_point_dropout and not args.no_augmentation
    use_pca_alignment = args.use_pca_alignment  # PCA alignment is not affected by --no_augmentation
    
    # Create summary file with augmentations and hyperparameters
    def create_training_summary(checkpoint_dir, run_log_dir, args, use_rotation, use_point_dropout):
        """Create a summary file with augmentations and hyperparameters."""
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("TRAINING CONFIGURATION SUMMARY")
        summary_lines.append("=" * 70)
        summary_lines.append("")
        
        # Preprocessing and Augmentations section
        summary_lines.append("DATA PREPROCESSING & AUGMENTATIONS:")
        summary_lines.append("-" * 70)
        summary_lines.append(f"  PCA Alignment: {'ENABLED' if use_pca_alignment else 'DISABLED'}")
        if use_pca_alignment:
            summary_lines.append(f"    - Target plane: {args.pca_target_plane}")
        
        if args.no_augmentation:
            summary_lines.append("  All augmentations: DISABLED")
        else:
            summary_lines.append(f"  Random Rotation: {'ENABLED' if use_rotation else 'DISABLED'}")
            if use_rotation:
                summary_lines.append(f"    - Rotation range: {args.rotation_range:.4f} radians ({np.degrees(args.rotation_range):.2f} degrees)")
            
            summary_lines.append(f"  Point Dropout: {'ENABLED' if use_point_dropout else 'DISABLED'}")
            if use_point_dropout:
                summary_lines.append(f"    - Dropout probability: {args.point_dropout_prob:.2%}")
                summary_lines.append(f"    - Minimum points to keep: {args.min_points}")
        
        if not args.no_augmentation and not use_rotation and not use_point_dropout and not use_pca_alignment:
            summary_lines.append("  No preprocessing or augmentations enabled")
        summary_lines.append("")
        
        # Hyperparameters section
        summary_lines.append("HYPERPARAMETERS:")
        summary_lines.append("-" * 70)
        summary_lines.append(f"  Model Architecture:")
        summary_lines.append(f"    - Model: PointNet2")
        summary_lines.append(f"    - Number of classes: 2 (Good vs Bad)")
        summary_lines.append(f"    - Normal channels: False (XYZ only)")
        summary_lines.append("")
        
        summary_lines.append(f"  Training Parameters:")
        summary_lines.append(f"    - Batch size: {args.batch_size}")
        summary_lines.append(f"    - Number of epochs: {args.epochs}")
        summary_lines.append(f"    - Learning rate: {args.learning_rate}")
        summary_lines.append(f"    - Optimizer: {args.optimizer.upper()}")
        if args.optimizer == 'sgd':
            summary_lines.append(f"    - Momentum: {args.momentum}")
        summary_lines.append(f"    - Learning rate decay step: {args.decay_step} batches")
        summary_lines.append(f"    - Learning rate decay rate: {args.decay_rate}")
        summary_lines.append(f"    - Gradient clipping max norm: {args.grad_clip}")
        summary_lines.append("")
        
        summary_lines.append(f"  Loss Function:")
        loss_type_name = "Binary Cross Entropy (BCE)" if args.loss_type.lower() == 'bce' else "Negative Log Likelihood (NLL)"
        summary_lines.append(f"    - Loss type: {loss_type_name}")
        summary_lines.append(f"    - Weighted: No (unweighted)")
        summary_lines.append("")
        
        summary_lines.append(f"  Data Loading:")
        summary_lines.append(f"    - Train/test split ratio: 80/20")
        summary_lines.append(f"    - Number of workers: {args.num_workers}")
        summary_lines.append(f"    - Random seed: {args.seed}")
        summary_lines.append("")
        
        summary_lines.append(f"  System:")
        summary_lines.append(f"    - Device: {device}")
        summary_lines.append(f"    - Checkpoint directory: {checkpoint_dir}")
        summary_lines.append(f"    - Log directory: {run_log_dir}")
        if args.resume:
            summary_lines.append(f"    - Resumed from: {args.resume}")
        summary_lines.append("")
        
        summary_lines.append("=" * 70)
        summary_lines.append("")
        summary_lines.append(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        summary_text = "\n".join(summary_lines)
        
        # Save to both directories
        summary_path_checkpoint = os.path.join(checkpoint_dir, 'training_summary.txt')
        summary_path_log = os.path.join(run_log_dir, 'training_summary.txt')
        
        with open(summary_path_checkpoint, 'w') as f:
            f.write(summary_text)
        
        with open(summary_path_log, 'w') as f:
            f.write(summary_text)
        
        print("\n" + summary_text)
        print(f"\nTraining summary saved to:")
        print(f"  - {summary_path_checkpoint}")
        print(f"  - {summary_path_log}")
    
    # Create summary (will be called after model is created to get device info)
    
    loaders = get_pointnet_dataloaders(
        batch_size=args.batch_size,
        train_ratio=0.8,
        seed=args.seed,
        num_workers=args.num_workers,
        concatenate=True,
        use_rotation=use_rotation,
        rotation_range=args.rotation_range,
        use_point_dropout=use_point_dropout,
        point_dropout_prob=args.point_dropout_prob,
        min_points=args.min_points,
        use_pca_alignment=use_pca_alignment,
        pca_target_plane=args.pca_target_plane,
        verbose=True
    )
    
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    stats = loaders['stats']
    
    print(f"\nTraining samples: {stats['train_total']}")
    print(f"Test samples: {stats['test_total']}")
    
    # Create PointNet2 model using Pointnet_Pointnet2_pytorch implementation
    # num_class=2 for binary classification (Good vs Bad)
    # normal_channel=False since we only have XYZ coordinates (no normals)
    model = get_model(num_class=2, normal_channel=False)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss function selection: BCE or NLL
    # PointNet2 returns log_softmax outputs, which works for both loss types
    # PointNet2 doesn't use feature transform regularization
    
    # Get class counts for information only (not for weighting)
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.cpu().numpy())
    train_labels = np.array(train_labels)
    
    # Count samples per class
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    
    print(f"\nClass distribution in training set:")
    print(f"  Good (0): {class_counts[0]} samples ({class_counts[0]/total_samples*100:.1f}%)")
    print(f"  Bad (1): {class_counts[1]} samples ({class_counts[1]/total_samples*100:.1f}%)")
    
    # Create loss function based on user selection
    if args.loss_type.lower() == 'bce':
        print(f"Using Binary Cross Entropy (BCE) loss")
        
        class BCELoss(nn.Module):
            def __init__(self):
                super(BCELoss, self).__init__()
                
            def forward(self, pred, target, trans_feat):
                # pred is log_softmax with shape (B, 2)
                # Convert log_softmax to logit for class 1 (Bad)
                # logit = log_softmax[:, 1] - log_softmax[:, 0]
                logit = pred[:, 1] - pred[:, 0]
                
                # Convert target to float for BCE loss (0 for Good, 1 for Bad)
                target_float = target.float()
                
                # BCE loss with logits (unweighted)
                # PointNet2 doesn't use trans_feat for regularization, but we accept it for compatibility
                loss = F.binary_cross_entropy_with_logits(logit, target_float)
                return loss
        
        criterion = BCELoss()
        
    elif args.loss_type.lower() == 'nll':
        print(f"Using Negative Log Likelihood (NLL) loss")
        
        class NLLLoss(nn.Module):
            def __init__(self):
                super(NLLLoss, self).__init__()
                
            def forward(self, pred, target, trans_feat):
                # pred is log_softmax with shape (B, 2)
                # NLL loss expects log probabilities and class indices
                # PointNet2 doesn't use trans_feat for regularization, but we accept it for compatibility
                loss = F.nll_loss(pred, target)
                return loss
        
        criterion = NLLLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}. Choose 'bce' or 'nll'")
    
    criterion = criterion.to(device)
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    # Learning rate scheduler
    # Decay based on number of batches (steps), not epochs
    # Convert decay_step (batch steps) to decay_epochs
    steps_per_epoch = len(train_loader)
    decay_epochs = max(1, args.decay_step // steps_per_epoch)  # At least 1 epoch
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_epochs, gamma=args.decay_rate)
    
    print(f"Learning rate decay: every {decay_epochs} epochs, rate={args.decay_rate}")
    
    # Create training summary file (after all setup is complete)
    create_training_summary(checkpoint_dir, run_log_dir, args, use_rotation, use_point_dropout)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_test_acc = 0.0  # Best balanced accuracy
    
    # Track metrics for plotting
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_test_acc = checkpoint.get('best_test_acc', 0.0)
        # Try to load previous metrics if available
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        test_accs = checkpoint.get('test_accs', [])
        print(f"Resumed from epoch {start_epoch}, best test acc: {best_test_acc:.4f}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Train
        train_loss, train_balanced_acc, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=args.grad_clip
        )
        
        # Evaluate
        # Return order: (avg_loss, accuracy, class_acc, all_preds, all_labels, standard_accuracy, balanced_accuracy)
        test_loss, test_balanced_acc, test_class_acc, test_preds, test_labels, test_standard_acc, _ = evaluate(
            model, test_loader, criterion, device, return_predictions=True
        )
        
        # Track metrics (using balanced accuracy for plotting)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_balanced_acc)
        test_accs.append(test_balanced_acc)
        
        # Save losses to text files after each epoch
        train_loss_file = os.path.join(run_log_dir, 'train_loss.txt')
        test_loss_file = os.path.join(run_log_dir, 'test_loss.txt')
        
        with open(train_loss_file, 'w') as f:
            f.write("Epoch\tTrain_Loss\n")
            for epoch_idx, loss in enumerate(train_losses):
                f.write(f"{epoch_idx+1}\t{loss:.6f}\n")
        
        with open(test_loss_file, 'w') as f:
            f.write("Epoch\tTest_Loss\n")
            for epoch_idx, loss in enumerate(test_losses):
                f.write(f"{epoch_idx+1}\t{loss:.6f}\n")
        
        # Print confusion matrices for both train and test
        print("\n" + "="*70)
        print("TRAINING SET CONFUSION MATRIX")
        print("="*70)
        # Ensure train_labels and train_preds are lists
        if isinstance(train_labels, np.ndarray):
            train_labels = train_labels.tolist()
        if isinstance(train_preds, np.ndarray):
            train_preds = train_preds.tolist()
        _ = evaluate_confusion_matrix(train_labels, train_preds, "Train")
        
        print("\n" + "="*70)
        print("TEST SET CONFUSION MATRIX")
        print("="*70)
        # Ensure test_labels and test_preds are lists
        if isinstance(test_labels, np.ndarray):
            test_labels = test_labels.tolist()
        if isinstance(test_preds, np.ndarray):
            test_preds = test_preds.tolist()
        _ = evaluate_confusion_matrix(test_labels, test_preds, "Test")
        
        # Log to TensorBoard
        if writer is not None:
            # Ensure all values are Python scalars (not numpy scalars or lists)
            # Check if values are already scalars before converting
            def to_float(val):
                if isinstance(val, (list, np.ndarray)):
                    if len(val) == 1:
                        return float(val[0])
                    raise ValueError(f"Cannot convert {type(val)} with length {len(val)} to float")
                return float(val)
            
            writer.add_scalar('Loss/Train', to_float(train_loss), epoch)
            writer.add_scalar('Loss/Test', to_float(test_loss), epoch)
            writer.add_scalar('Accuracy/Train_Balanced', to_float(train_balanced_acc), epoch)
            writer.add_scalar('Accuracy/Test_Balanced', to_float(test_balanced_acc), epoch)
            writer.add_scalar('Accuracy/Test_Good', to_float(test_class_acc[0]), epoch)
            writer.add_scalar('Accuracy/Test_Bad', to_float(test_class_acc[1]), epoch)
            writer.add_scalar('Accuracy/Test_Standard', to_float(test_standard_acc), epoch)
            writer.add_scalar('Learning_Rate', float(optimizer.param_groups[0]['lr']), epoch)
        
        # Calculate precision for summary (Good is positive class)
        test_labels_np = np.array(test_labels)
        test_preds_np = np.array(test_preds)
        TP = np.sum((test_preds_np == 0) & (test_labels_np == 0))  # Predicted Good, actual Good
        FP = np.sum((test_preds_np == 0) & (test_labels_np == 1))  # Predicted Good, actual Bad
        TN = np.sum((test_preds_np == 1) & (test_labels_np == 1))  # Predicted Bad, actual Bad
        FN = np.sum((test_preds_np == 1) & (test_labels_np == 0))  # Predicted Bad, actual Good
        
        test_precision_good = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        test_precision_bad = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        
        # Print results
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Train Loss: {train_loss:.4f}, Train Balanced Acc: {train_balanced_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Balanced Acc: {test_balanced_acc:.4f}")
        print(f"Test Standard Acc: {test_standard_acc:.4f}")
        print(f"Test Per-Class Acc - Good: {test_class_acc[0]:.4f}, Bad: {test_class_acc[1]:.4f}")
        print(f"Test Precision - Good: {test_precision_good:.4f}, Bad: {test_precision_bad:.4f}")
        print(f"\nWeighted Accuracy Formula: (0.5 * Precision_Good) + (0.5 * Precision_Bad)")
        print(f"  Where Precision_Good = TP/(TP+FP), Precision_Bad = TN/(TN+FN)")
        print(f"  TP=True Positives (Good), FP=False Positives, TN=True Negatives (Bad), FN=False Negatives")
        
        # Save loss curves periodically
        if (epoch + 1) % 10 == 0:
            loss_curve_path = os.path.join(run_log_dir, 'loss_curves.png')
            plot_loss_curves(train_losses, test_losses, train_accs, test_accs, loss_curve_path)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_balanced_acc': train_balanced_acc,
                'test_loss': test_loss,
                'test_balanced_acc': test_balanced_acc,
                'best_test_acc': best_test_acc,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accs': train_accs,
                'test_accs': test_accs,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model (based on balanced accuracy)
        if test_balanced_acc > best_test_acc:
            best_test_acc = test_balanced_acc
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_balanced_acc': train_balanced_acc,
                'test_loss': test_loss,
                'test_balanced_acc': test_balanced_acc,
                'best_test_acc': best_test_acc,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accs': train_accs,
                'test_accs': test_accs,
            }, best_model_path)
            print(f"New best model saved! Test balanced accuracy: {test_balanced_acc:.4f}")
        
        # Step learning rate scheduler
        scheduler.step()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best test balanced accuracy: {best_test_acc:.4f}")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    
    # Final evaluation with confusion matrix
    print("\n" + "="*70)
    print("GENERATING FINAL EVALUATION")
    print("="*70)
    test_loss, test_acc, test_class_acc, all_preds, all_labels, test_standard_acc, test_balanced_acc = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    # Print final confusion matrices
    print("\n" + "="*70)
    print("FINAL TRAINING SET CONFUSION MATRIX")
    print("="*70)
    # Get training predictions for final evaluation
    model.eval()
    train_preds_final = []
    train_labels_final = []
    with torch.no_grad():
        for points, labels in train_loader:
            points = points.to(device).transpose(2, 1)
            labels = labels.to(device)
            pred, _ = model(points)  # PointNet2 returns (log_softmax, feature_points)
            pred_choice = pred.data.max(1)[1]
            train_preds_final.extend(pred_choice.cpu().numpy())
            train_labels_final.extend(labels.cpu().numpy())
    evaluate_confusion_matrix(train_labels_final, train_preds_final, "Train")
    
    print("\n" + "="*70)
    print("FINAL TEST SET CONFUSION MATRIX")
    print("="*70)
    evaluate_confusion_matrix(all_labels, all_preds, "Test")
    
    # Save final loss curves
    loss_curve_path = os.path.join(run_log_dir, 'loss_curves_final.png')
    plot_loss_curves(train_losses, test_losses, train_accs, test_accs, loss_curve_path)
    
    # Save final losses to text files
    train_loss_file = os.path.join(run_log_dir, 'train_loss.txt')
    test_loss_file = os.path.join(run_log_dir, 'test_loss.txt')
    
    with open(train_loss_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\n")
        for epoch_idx, loss in enumerate(train_losses):
            f.write(f"{epoch_idx+1}\t{loss:.6f}\n")
    
    with open(test_loss_file, 'w') as f:
        f.write("Epoch\tTest_Loss\n")
        for epoch_idx, loss in enumerate(test_losses):
            f.write(f"{epoch_idx+1}\t{loss:.6f}\n")
    
    print(f"Loss files saved:")
    print(f"  - Train loss: {train_loss_file}")
    print(f"  - Test loss: {test_loss_file}")
    
    # Save confusion matrix
    cm_path = os.path.join(run_log_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, cm_path, class_names=['Good', 'Bad'])
    
    # Also save confusion matrix in checkpoint_dir for easy access
    cm_path_save = os.path.join(checkpoint_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, cm_path_save, class_names=['Good', 'Bad'])
    
    if writer is not None:
        print(f"Logs saved to: {run_log_dir}")
        writer.close()
    
    print(f"\nAll visualizations saved to: {run_log_dir}")
    print(f"Loss curves: {loss_curve_path}")
    print(f"Confusion matrix: {cm_path}")


if __name__ == '__main__':
    main()
