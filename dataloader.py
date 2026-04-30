#!/usr/bin/env python3
"""
DataLoader for good and bad XYZ point cloud pairs.
Loads from good_patches/ and bad_patches/ directories.
Splits 80% for training, 20% for testing for both classes.
"""

import os
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader

# Try to import sklearn for PCA
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None


class PointDropout:
    """
    Data augmentation that randomly removes points from point clouds.
    Simulates missing points in real-world scenarios.
    
    CRITICAL: Maintains original size by DUPLICATING existing points (not zeros)
    This ensures compatibility with normalization - no artificial zero vectors at origin.
    """
    
    def __init__(self, dropout_prob: float = 0.1, min_points: int = 512):
        """
        Args:
            dropout_prob: Probability of dropping each point (0.0 to 1.0)
                          e.g., 0.1 means 10% of points will be dropped on average
            min_points: Minimum number of points to keep (ensures we don't drop too many)
        """
        self.dropout_prob = dropout_prob
        self.min_points = min_points
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Randomly drop points from the point cloud.
        
        CRITICAL: Maintains original size by RANDOMLY DUPLICATING kept points
        (instead of padding with zeros). This ensures the point cloud distribution
        remains similar to test data, which is essential after normalization.
        
        Args:
            points: Point cloud tensor of shape (N, 3)
        
        Returns:
            Point cloud with some points randomly sampled, shape (N, 3)
        """
        if self.dropout_prob <= 0.0:
            return points
        
        num_points = points.shape[0]
        
        # Calculate how many points to keep (at least min_points)
        max_drop = max(0, num_points - self.min_points)
        num_to_drop = min(int(num_points * self.dropout_prob), max_drop)
        
        if num_to_drop == 0:
            return points
        
        num_to_keep = num_points - num_to_drop
        
        # Randomly select indices to keep (these are the "surviving" points)
        keep_indices = torch.randperm(num_points, device=points.device)[:num_to_keep]
        kept_points = points[keep_indices]
        
        # CRITICAL: Instead of filling with zeros, randomly duplicate kept points
        # to reach the original size
        num_to_duplicate = num_to_drop
        
        # Randomly sample from kept points to fill the dropped slots
        duplicate_indices = torch.randint(0, num_to_keep, (num_to_duplicate,), device=points.device)
        duplicated_points = kept_points[duplicate_indices]
        
        # Combine kept points and duplicated points
        result = torch.cat([kept_points, duplicated_points], dim=0)  # (N, 3)
        
        # Shuffle to randomize positions
        shuffle_indices = torch.randperm(num_points, device=points.device)
        result = result[shuffle_indices]
        
        return result


class Centering:
    """
    Data preprocessing that centers point clouds at the origin.
    Ensures the centroid of the patch pair is at the origin.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Center the point cloud at the origin by subtracting the centroid.
        
        Args:
            points: Point cloud tensor of shape (N, 3) - concatenated patch pair
        
        Returns:
            Centered point cloud, shape (N, 3)
        """
        # Calculate centroid of the entire patch pair
        centroid = points.mean(dim=0)  # (3,)
        
        # Center by subtracting centroid
        centered_points = points - centroid
        
        return centered_points


class Normalization:
    """
    Data preprocessing that normalizes point clouds to a consistent scale.
    Scales points to fit within a unit sphere (max distance from origin = 1).
    This ensures consistent scale across all samples.
    Should be applied AFTER centering.
    """
    
    def __init__(self, method: str = 'unit_sphere'):
        """
        Args:
            method: Normalization method
                - 'unit_sphere': Scale so max distance from origin is 1
                - 'std': Standardize to have std=1 (less common for point clouds)
        """
        self.method = method
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Normalize the point cloud to a consistent scale.
        
        Args:
            points: Point cloud tensor of shape (N, 3) - should be centered first
        
        Returns:
            Normalized point cloud, shape (N, 3)
        """
        if self.method == 'unit_sphere':
            # Scale to unit sphere: max distance from origin = 1
            # Calculate distance of each point from origin
            distances = torch.norm(points, dim=1)  # (N,)
            max_distance = distances.max()
            
            # Avoid division by zero
            if max_distance > 1e-6:
                normalized_points = points / max_distance
            else:
                normalized_points = points
                
        elif self.method == 'std':
            # Standardize to have std = 1
            std = points.std()
            if std > 1e-6:
                normalized_points = points / std
            else:
                normalized_points = points
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return normalized_points


class RandomRotation:
    """
    Data augmentation that randomly rotates point clouds around the Y-axis (up direction).
    """
    
    def __init__(self, rotation_range: float = 2.0 * np.pi):
        """
        Args:
            rotation_range: Maximum rotation angle in radians (default: 2π, full rotation)
        """
        self.rotation_range = rotation_range
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Randomly rotate the point cloud around the Y-axis.
        
        Args:
            points: Point cloud tensor of shape (N, 3)
        
        Returns:
            Rotated point cloud, shape (N, 3)
        """
        if self.rotation_range <= 0.0:
            return points
        
        # Generate random rotation angle
        rotation_angle = np.random.uniform(0, self.rotation_range)
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        
        # Rotation matrix around Y-axis
        rotation_matrix = torch.tensor([
            [cosval, 0, sinval],
            [0, 1, 0],
            [-sinval, 0, cosval]
        ], dtype=points.dtype, device=points.device)
        
        # Apply rotation: (N, 3) @ (3, 3) -> (N, 3)
        rotated_points = torch.matmul(points, rotation_matrix.T)
        
        return rotated_points


class PCAAlignment:
    """
    Data preprocessing that applies PCA alignment to patch pairs.
    Aligns both patches in a pair using PCA so they have the same orientation.
    This should be applied before other augmentations.
    """
    
    def __init__(self, target_plane: str = 'xy'):
        """
        Args:
            target_plane: Target plane orientation ('xy', 'xz', or 'yz').
                         'xy' means patches will be oriented parallel to xy plane
                         (first PC -> x-axis, second PC -> y-axis, third PC -> z-axis)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA alignment. Install with: pip install scikit-learn")
        self.target_plane = target_plane
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA alignment to a concatenated point cloud.
        Assumes the point cloud is concatenated from two equal-sized patches.
        
        Args:
            points: Point cloud tensor of shape (N*2, 3) where N is points per patch
        
        Returns:
            PCA-aligned point cloud, shape (N*2, 3)
        """
        if not SKLEARN_AVAILABLE:
            return points
        
        # Convert to numpy
        points_np = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
        
        # Assume equal-sized patches (split in half)
        num_points = points_np.shape[0]
        if num_points % 2 != 0:
            # If odd number of points, handle gracefully
            mid_point = num_points // 2
            patch_a = points_np[:mid_point]
            patch_b = points_np[mid_point:]
        else:
            mid_point = num_points // 2
            patch_a = points_np[:mid_point]
            patch_b = points_np[mid_point:]
        
        # Apply PCA alignment
        try:
            patch_a_aligned, patch_b_aligned = self._reorient_patch_pair_with_pca(
                patch_a, patch_b, self.target_plane
            )
            
            # Concatenate back
            aligned_points = np.concatenate([patch_a_aligned, patch_b_aligned], axis=0)
            
            # Convert back to tensor
            if isinstance(points, torch.Tensor):
                return torch.from_numpy(aligned_points).float().to(points.device)
            else:
                return aligned_points
        except Exception as e:
            # If PCA fails, return original points
            print(f"Warning: PCA alignment failed: {e}. Using original points.")
            return points
    
    def _reorient_patch_pair_with_pca(self, patch_a, patch_b, target_plane='xy'):
        """
        Reorient a patch pair using PCA so both patches have the same orientation.
        
        Args:
            patch_a: First patch, shape (N, 3)
            patch_b: Second patch, shape (M, 3)
            target_plane: Target plane orientation ('xy', 'xz', or 'yz')
        
        Returns:
            patch_a_rotated: Rotated first patch (centered at origin)
            patch_b_rotated: Rotated second patch (centered at origin)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA reorientation. Install with: pip install scikit-learn")
        
        # Combine both patches for PCA
        combined_points = np.vstack([patch_a, patch_b])
        
        # Center the combined points at origin
        centroid = np.mean(combined_points, axis=0)
        centered_points = combined_points - centroid
        
        # Compute PCA
        pca = PCA(n_components=3)
        pca.fit(centered_points)
        
        # Get principal components (each row is a component vector)
        # Components are ordered by explained variance (largest first)
        components = pca.components_  # Shape: (3, 3)
        
        # Create rotation matrix to align PCA components with coordinate axes
        # We want: first PC -> x-axis, second PC -> y-axis, third PC -> z-axis
        rotation_matrix = components.copy()
        
        # Ensure right-handed coordinate system (determinant should be +1)
        if np.linalg.det(rotation_matrix) < 0:
            # Flip the third component to make it right-handed
            rotation_matrix[2, :] = -rotation_matrix[2, :]
        
        # Apply rotation to both patches (after centering)
        patch_a_centered = patch_a - centroid
        patch_b_centered = patch_b - centroid
        
        # Rotate: new_points = (rotation_matrix @ old_points.T).T
        patch_a_rotated = (rotation_matrix @ patch_a_centered.T).T
        patch_b_rotated = (rotation_matrix @ patch_b_centered.T).T
        
        return patch_a_rotated, patch_b_rotated


class Compose:
    """
    Composes multiple transforms together.
    """
    
    def __init__(self, transforms: List):
        """
        Args:
            transforms: List of transform callables
        """
        self.transforms = transforms
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply transforms in sequence.
        
        Args:
            points: Point cloud tensor
        
        Returns:
            Transformed point cloud tensor
        """
        for transform in self.transforms:
            if transform is not None:
                points = transform(points)
        return points


class PointNetDataset(Dataset):
    """Dataset for PointNet - loads XYZ point clouds and concatenates sheet1 and sheet2 pairs."""
    
    def __init__(self, file_pairs: List[Tuple[str, str]], labels: List[int], 
                 concatenate: bool = True, transform=None):
        """
        Args:
            file_pairs: List of (sheet1_path, sheet2_path) tuples (XYZ coordinate files)
            labels: List of labels (0 for good, 1 for bad)
            concatenate: If True, concatenate sheet1 and sheet2 point clouds into one (N*2, 3)
                        If False, return tuple of (points_sheet1, points_sheet2) each (N, 3)
            transform: Optional transform to apply to point clouds
        """
        self.file_pairs = file_pairs
        self.labels = labels
        self.concatenate = concatenate
        self.transform = transform
        
        # Verify all files exist
        for file_sheet1, file_sheet2 in self.file_pairs:
            if not os.path.exists(file_sheet1):
                raise FileNotFoundError(f"File not found: {file_sheet1}")
            if not os.path.exists(file_sheet2):
                raise FileNotFoundError(f"File not found: {file_sheet2}")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        file_sheet1, file_sheet2 = self.file_pairs[idx]
        label = self.labels[idx]
        
        # Load XYZ point clouds
        xyz_sheet1 = np.load(file_sheet1)  # Shape: (N, 3)
        xyz_sheet2 = np.load(file_sheet2)  # Shape: (N, 3)
        
        if self.concatenate:
            # Concatenate sheet1 and sheet2 into single point cloud (N*2, 3)
            # PointNet expects (num_points, 3) format
            points = np.concatenate([xyz_sheet1, xyz_sheet2], axis=0)  # (N*2, 3)
            points_tensor = torch.from_numpy(points).float()
        else:
            # Return both point clouds separately
            points_sheet1_tensor = torch.from_numpy(xyz_sheet1).float()
            points_sheet2_tensor = torch.from_numpy(xyz_sheet2).float()
            points_tensor = (points_sheet1_tensor, points_sheet2_tensor)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Apply transform if provided
        # Note: PCAAlignment works on concatenated point clouds by splitting them,
        # aligning both patches, and concatenating back
        if self.transform:
            if self.concatenate:
                points_tensor = self.transform(points_tensor)
            else:
                points_sheet1_tensor, points_sheet2_tensor = points_tensor
                points_sheet1_tensor = self.transform(points_sheet1_tensor)
                points_sheet2_tensor = self.transform(points_sheet2_tensor)
                points_tensor = (points_sheet1_tensor, points_sheet2_tensor)
        
        return points_tensor, label_tensor


def find_xyz_pairs(directory: str, prefix: str) -> List[Tuple[str, str]]:
    """
    Find all XYZ coordinate pair files (sheet1 and sheet2) in a directory.
    
    Args:
        directory: Directory path (e.g., 'good_patches' or 'bad_patches')
        prefix: File prefix (e.g., 'good_sheet' or 'bad_sheet')
    
    Returns:
        List of (sheet1_path, sheet2_path) tuples for XYZ coordinates
    """
    if not os.path.exists(directory):
        return []
    
    # Find all sheet1 files
    all_files = sorted([f for f in os.listdir(directory) if f.endswith('_sheet1_xyz.npy')])
    
    pairs = []
    for sheet1_file in all_files:
        # Get corresponding sheet2 file
        sheet2_file = sheet1_file.replace('_sheet1_xyz.npy', '_sheet2_xyz.npy')
        sheet1_path = os.path.join(directory, sheet1_file)
        sheet2_path = os.path.join(directory, sheet2_file)
        
        # Only include if both files exist and match the prefix
        if sheet1_file.startswith(prefix) and os.path.exists(sheet2_path):
            pairs.append((sheet1_path, sheet2_path))
    
    return pairs


def split_train_test(pairs: List[Tuple[str, str]], train_ratio: float = 0.8, 
                     seed: int = 42) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split pairs into train and test sets with random shuffling.
    
    Args:
        pairs: List of (file_sheet1, file_sheet2) tuples
        train_ratio: Ratio for training set (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        (train_pairs, test_pairs)
    """
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    n_train = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:n_train]
    test_pairs = shuffled[n_train:]
    
    return train_pairs, test_pairs


def get_pointnet_dataloaders(batch_size: int = 32, train_ratio: float = 0.8,
                             seed: int = 42, num_workers: int = 4,
                             shuffle_train: bool = True, concatenate: bool = True,
                             use_point_dropout: bool = False, point_dropout_prob: float = 0.1, min_points: int = 512,
                             use_rotation: bool = False, rotation_range: float = 2.0 * np.pi,
                             use_pca_alignment: bool = False, pca_target_plane: str = 'xy',
                             dataset_dir: str = '/ingenuity_NAS/23tp8_nas/23tp8_mount/DATASET',
                             verbose: bool = True) -> Dict:
    """
    Get PyTorch DataLoaders for PointNet (XYZ point clouds).
    Loads from good_patches/ and bad_patches/ directories.
    Splits 80% for training, 20% for testing for both classes.
    
    Args:
        batch_size: Batch size for DataLoader
        train_ratio: Ratio for training set (default: 0.8)
        seed: Random seed for reproducibility
        num_workers: Number of workers for DataLoader
        shuffle_train: Whether to shuffle training set
        concatenate: If True, concatenate sheet1 and sheet2 point clouds (default: True)
                    If False, return tuple of (points_sheet1, points_sheet2) separately
        use_point_dropout: Whether to apply point dropout augmentation (default: False)
        point_dropout_prob: Probability of dropping each point during training (0.0 to 1.0)
                           e.g., 0.1 means 10% of points will be dropped on average
                           Only applied to training set, not test set
        min_points: Minimum number of points to keep after dropout (default: 512)
        use_rotation: Whether to apply random rotation augmentation (default: False)
        rotation_range: Maximum rotation angle in radians (default: 2π, full rotation)
        use_pca_alignment: Whether to apply PCA alignment to patch pairs (default: False)
                          PCA alignment ensures both patches in a pair have the same orientation
                          This should be applied before other augmentations
        pca_target_plane: Target plane for PCA alignment ('xy', 'xz', or 'yz') (default: 'xy')
        dataset_dir: Base directory containing good_patches/ and bad_patches/ folders
                    (default: '/ingenuity_NAS/23tp8_nas/23tp8_mount/DATASET')
        verbose: Print statistics if True
    
    Returns:
        Dictionary with:
        - 'train_loader': Training DataLoader
        - 'test_loader': Test DataLoader
        - 'train_dataset': Training Dataset
        - 'test_dataset': Test Dataset
        - 'stats': Statistics dictionary with 'train_total' and 'test_total'
        Output shape: (batch_size, num_points, 3) if concatenate=True
                     (batch_size, 2, num_points, 3) if concatenate=False
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Construct full paths to good_patches and bad_patches directories
    good_patches_dir = os.path.join(dataset_dir, 'good_patches')
    bad_patches_dir = os.path.join(dataset_dir, 'bad_patches')
    
    # Find all good pairs
    good_pairs = find_xyz_pairs(good_patches_dir, 'good_sheet')
    
    # Find all bad pairs
    bad_pairs = find_xyz_pairs(bad_patches_dir, 'bad_sheet')
    
    total_good = len(good_pairs)
    total_bad = len(bad_pairs)
    
    if verbose:
        print("="*70)
        print("DATASET STATISTICS (XYZ POINT CLOUDS)")
        print("="*70)
        print(f"\nTotal pairs found:")
        print(f"  Good pairs: {total_good}")
        print(f"  Bad pairs: {total_bad}")
    
    # Split good pairs: 80% train, 20% test
    train_good, test_good = split_train_test(good_pairs, train_ratio, seed)
    
    # Split bad pairs: 80% train, 20% test
    train_bad, test_bad = split_train_test(bad_pairs, train_ratio, seed)
    
    # Combine training pairs and labels
    train_pairs = train_good + train_bad
    train_labels = [0] * len(train_good) + [1] * len(train_bad)
    
    # Shuffle training set randomly
    random.seed(seed)
    combined_train = list(zip(train_pairs, train_labels))
    random.shuffle(combined_train)
    train_pairs, train_labels = zip(*combined_train)
    train_pairs, train_labels = list(train_pairs), list(train_labels)
    
    # Combine test pairs and labels
    test_pairs = test_good + test_bad
    test_labels = [0] * len(test_good) + [1] * len(test_bad)
    
    # Shuffle test set randomly
    random.seed(seed + 1)  # Use different seed for test shuffle
    combined_test = list(zip(test_pairs, test_labels))
    random.shuffle(combined_test)
    test_pairs, test_labels = zip(*combined_test)
    test_pairs, test_labels = list(test_pairs), list(test_labels)
    
    # Create statistics
    stats = {
        'total_good': total_good,
        'total_bad': total_bad,
        'train_good': len(train_good),
        'train_bad': len(train_bad),
        'test_good': len(test_good),
        'test_bad': len(test_bad),
        'train_total': len(train_pairs),
        'test_total': len(test_pairs)
    }
    
    if verbose:
        print(f"\n" + "="*70)
        print("TRAIN/TEST SPLIT")
        print("="*70)
        print(f"\nTraining set ({train_ratio*100:.0f}% of each class):")
        print(f"  Good pairs: {stats['train_good']}")
        print(f"  Bad pairs: {stats['train_bad']}")
        print(f"  Total: {stats['train_total']} pairs")
        print(f"\nTest set ({(1-train_ratio)*100:.0f}% of each class):")
        print(f"  Good pairs: {stats['test_good']}")
        print(f"  Bad pairs: {stats['test_bad']}")
        print(f"  Total: {stats['test_total']} pairs")
        print("="*70)
    
    # Create augmentation transforms for training
    # CRITICAL ORDER: Centering -> Normalization -> PCA alignment -> Augmentations
    transforms = []
    
    # Always apply centering to both train and test
    transforms.append(Centering())
    
    # Always apply normalization after centering (ensures consistent scale)
    transforms.append(Normalization(method='unit_sphere'))
    
    if verbose:
        print(f"\nPreprocessing enabled for training and test:")
        print(f"  1. Centering: Point clouds centered at origin (centroid = 0)")
        print(f"  2. Normalization: Scaled to unit sphere (max distance from origin = 1)")
    
    if use_pca_alignment:
        if not SKLEARN_AVAILABLE:
            if verbose:
                print(f"\n⚠️  Warning: PCA alignment requested but scikit-learn not available.")
                print(f"   Install with: pip install scikit-learn")
                print(f"   Continuing without PCA alignment...")
        else:
            transforms.append(PCAAlignment(target_plane=pca_target_plane))
            if verbose:
                print(f"\nPCA alignment enabled for training:")
                print(f"  Target plane: {pca_target_plane}")
    
    if use_rotation:
        transforms.append(RandomRotation(rotation_range=rotation_range))
        if verbose:
            print(f"\nRandom rotation augmentation enabled for training:")
            print(f"  Rotation range: {rotation_range:.2f} radians ({np.degrees(rotation_range):.1f} degrees)")
    
    if use_point_dropout:
        transforms.append(PointDropout(dropout_prob=point_dropout_prob, min_points=min_points))
        if verbose:
            print(f"\nPoint dropout augmentation enabled for training:")
            print(f"  Dropout probability: {point_dropout_prob:.2%}")
            print(f"  Minimum points to keep: {min_points}")
    
    # Compose transforms if any are enabled
    train_transform = None
    if transforms:
        train_transform = Compose(transforms)
        if verbose:
            print(f"\nTotal augmentations/preprocessing enabled: {len(transforms)}")
    
    # Create Dataset objects
    # Training set gets all transforms (centering + PCA if enabled + augmentations)
    train_dataset = PointNetDataset(train_pairs, train_labels, concatenate=concatenate, transform=train_transform)
    
    # Test set transform: centering + normalization + PCA alignment if enabled (no rotation or dropout)
    test_transforms = [Centering(), Normalization(method='unit_sphere')]  # Always center and normalize test data
    if use_pca_alignment and SKLEARN_AVAILABLE:
        test_transforms.append(PCAAlignment(target_plane=pca_target_plane))
    
    test_transform = None
    if test_transforms:
        test_transform = Compose(test_transforms)
        if verbose:
            print(f"\nTest set preprocessing: {len(test_transforms)} transform(s) enabled")
            print(f"  1. Centering: ENABLED")
            print(f"  2. Normalization: ENABLED (unit sphere)")
            if use_pca_alignment and SKLEARN_AVAILABLE:
                print(f"  3. PCA alignment: ENABLED")
    
    test_dataset = PointNetDataset(test_pairs, test_labels, concatenate=concatenate, transform=test_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'stats': stats
    }


def main():
    """Example usage of the dataloader."""
    print("\n" + "="*70)
    print("TESTING POINTNET DATALOADER")
    print("="*70)
    
    # Test PointNet dataloader
    pointnet_loaders = get_pointnet_dataloaders(
        batch_size=32,
        train_ratio=0.8,
        seed=42,
        num_workers=4,
        concatenate=True,
        verbose=True
    )
    
    train_loader = pointnet_loaders['train_loader']
    test_loader = pointnet_loaders['test_loader']
    stats = pointnet_loaders['stats']
    
    # Get a batch from training set
    for batch_data, batch_labels in train_loader:
        print(f"\nTraining batch (PointNet):")
        print(f"  Batch shape: {batch_data.shape}")  # Should be (batch_size, num_points, 3)
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Good samples: {sum(1 for l in batch_labels if l == 0)}")
        print(f"  Bad samples: {sum(1 for l in batch_labels if l == 1)}")
        break
    
    # Get a batch from test set
    for batch_data, batch_labels in test_loader:
        print(f"\nTest batch (PointNet):")
        print(f"  Batch shape: {batch_data.shape}")
        print(f"  Labels shape: {batch_labels.shape}")
        print(f"  Good samples: {sum(1 for l in batch_labels if l == 0)}")
        print(f"  Bad samples: {sum(1 for l in batch_labels if l == 1)}")
        break
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"\nTraining samples: {stats['train_total']}")
    print(f"Test samples: {stats['test_total']}")
    
    print("\n" + "="*70)
    print("DATALOADERS READY")
    print("="*70)


if __name__ == "__main__":
    main()
