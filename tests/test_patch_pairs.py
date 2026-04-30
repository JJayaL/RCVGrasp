#!/usr/bin/env python3
"""
Test all patch pairs from ape_patches/ directory and visualize the best "good" prediction.

This script:
1. Loads the trained model from best_model.pth
2. Loads all patches from ape_patches/
3. Creates all possible pairs (200 choose 2 = 19,900 pairs)
4. Applies the same preprocessing as training (centering, normalization, optionally PCA)
5. Runs inference on all pairs
6. Finds the pair with the best "good" prediction (highest probability of class 0)
7. Visualizes the best pair
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import colorsys

# Import model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POINTNET_DIR = os.path.join(BASE_DIR, 'Pointnet_Pointnet2_pytorch')
sys.path.append(os.path.join(POINTNET_DIR, 'models'))
from Pointnet_Pointnet2_pytorch.models.pointnet2_cls_ssg import get_model

# Import preprocessing from dataloader
from dataloader import Centering, Normalization, PCAAlignment, Compose

# Import visualization
from visualize_patches import visualize_patches

# Try to import Open3D for pointcloud visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available for pointcloud visualization. Install with: pip install open3d")


def load_patches(patch_dir):
    """
    Load all patches from the directory.
    
    Args:
        patch_dir: Directory containing patch_*.npy files
        
    Returns:
        List of patch arrays, each of shape (N, 3)
    """
    patches = []
    patch_files = sorted([f for f in os.listdir(patch_dir) if f.startswith('patch_') and f.endswith('.npy')])
    
    print(f"Loading {len(patch_files)} patches from {patch_dir}...")
    for patch_file in tqdm(patch_files, desc='Loading patches'):
        patch_path = os.path.join(patch_dir, patch_file)
        patch = np.load(patch_path)  # Shape: (N, 3)
        patches.append(patch)
    
    print(f"Loaded {len(patches)} patches")
    print(f"Patch shape: {patches[0].shape if patches else 'N/A'}")
    
    return patches


def load_patch_metadata(patch_dir):
    """
    Load patch metadata (center locations in original pointcloud).
    
    Args:
        patch_dir: Directory containing patch_metadata.json
        
    Returns:
        Dictionary with metadata, or None if not found
    """
    import json
    metadata_path = os.path.join(patch_dir, 'patch_metadata.json')
    
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found: {metadata_path}")
        print("Run create_patch_ascii.py to generate metadata.")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def load_ply_pointcloud(filepath):
    """
    Load pointcloud from PLY file (ASCII format).
    
    Args:
        filepath: Path to PLY file
        
    Returns:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (optional, may be None)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header_end = 0
    num_vertices = 0
    has_colors = False
    properties = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
        elif line.startswith('property'):
            parts = line.split()
            if len(parts) >= 3:
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))
                if prop_name in ['red', 'r'] or (prop_name == 'diffuse_red'):
                    has_colors = True
        elif line == 'end_header':
            header_end = i + 1
            break
    
    # Read vertex data (skip header)
    data_lines = lines[header_end:header_end + num_vertices]
    
    points = []
    colors = []
    
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])
            
            # Try to extract colors (could be r/g/b or red/green/blue)
            if has_colors and len(parts) >= 6:
                # Check if colors are in 0-255 range or 0-1 range
                r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                # If values are > 1, they're likely 0-255, normalize them
                if r > 1.0 or g > 1.0 or b > 1.0:
                    r, g, b = int(r), int(g), int(b)
                else:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                colors.append([r, g, b])
    
    points = np.array(points)
    colors = np.array(colors) if colors else None
    
    return points, colors


def load_ascii_pointcloud(filepath):
    """
    Load pointcloud from ASCII file.
    
    Format expected:
    //X Y Z R G B
    <num_points>
    <x> <y> <z> <r> <g> <b>
    ...
    
    Args:
        filepath: Path to ASCII file
        
    Returns:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (optional, may be None)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip comment line and number of points line
    data_lines = lines[2:]
    
    points = []
    colors = []
    
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])
            if len(parts) >= 6:
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                colors.append([r, g, b])
    
    points = np.array(points)
    colors = np.array(colors) if colors else None
    
    return points, colors


def load_obj_pointcloud(filepath):
    """
    Load pointcloud from OBJ file.
    
    OBJ format:
    - Lines starting with 'v ' are vertices (points)
    - Format: v x y z [r g b] (colors optional)
    - Other lines (f, #, etc.) are ignored
    
    Args:
        filepath: Path to OBJ file
        
    Returns:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (optional, may be None)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    points = []
    colors = []
    has_colors = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # OBJ vertex line: v x y z [r g b]
        if line.startswith('v '):
            parts = line.split()
            if len(parts) >= 4:  # At least v x y z
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
                
                # Check if colors are present (r g b after x y z)
                if len(parts) >= 7:
                    r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                    # OBJ colors are typically 0-1 range, convert to 0-255
                    if r <= 1.0 and g <= 1.0 and b <= 1.0:
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    else:
                        r, g, b = int(r), int(g), int(b)
                    colors.append([r, g, b])
                    has_colors = True
                elif has_colors:
                    # If some vertices have colors but this one doesn't, add None/placeholder
                    colors.append([128, 128, 128])  # Default gray
    
    points = np.array(points)
    colors = np.array(colors) if has_colors and len(colors) == len(points) else None
    
    return points, colors


def load_original_pointcloud(filepath):
    """
    Load the original pointcloud file (auto-detects PLY, OBJ, or ASCII format).
    
    Args:
        filepath: Path to pointcloud file (.ply, .obj, or ASCII format)
        
    Returns:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (if available)
    """
    if not os.path.exists(filepath):
        print(f"Warning: Original pointcloud file not found: {filepath}")
        return None, None
    
    print(f"Loading original pointcloud from {filepath}...")
    
    # Check file extension or header to determine format
    if filepath.lower().endswith('.ply'):
        # Check if it's a PLY file by reading first line
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if first_line == 'ply':
                points, colors = load_ply_pointcloud(filepath)
                print(f"Loaded {len(points)} points from original pointcloud (PLY format)")
                return points, colors
    elif filepath.lower().endswith('.obj'):
        # OBJ file format
        points, colors = load_obj_pointcloud(filepath)
        print(f"Loaded {len(points)} points from original pointcloud (OBJ format)")
        return points, colors
    
    # Try ASCII format
    try:
        points, colors = load_ascii_pointcloud(filepath)
        print(f"Loaded {len(points)} points from original pointcloud (ASCII format)")
        return points, colors
    except (ValueError, IndexError) as e:
        # If ASCII fails, try other formats
        if not filepath.lower().endswith(('.ply', '.obj')):
            # Check if it might be PLY by reading first line
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                if first_line == 'ply':
                    points, colors = load_ply_pointcloud(filepath)
                    print(f"Loaded {len(points)} points from original pointcloud (PLY format)")
                    return points, colors
                # Check if it might be OBJ (look for 'v ' lines)
                f.seek(0)
                for line in f:
                    if line.strip().startswith('v '):
                        points, colors = load_obj_pointcloud(filepath)
                        print(f"Loaded {len(points)} points from original pointcloud (OBJ format)")
                        return points, colors
        raise ValueError(f"Could not parse pointcloud file {filepath}. Error: {e}")


def apply_nms_to_pairs(patch_pairs, metadata, points=None, iou_threshold=0.3):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping patch pairs.
    
    Args:
        patch_pairs: List of tuples (rank, patch_idx1, patch_idx2, prob_good, color_name, color_rgb)
        metadata: Patch metadata dictionary
        points: Original pointcloud points (optional, for radius estimation)
        iou_threshold: IoU threshold for NMS (0.0-1.0). Lower values = more aggressive filtering.
                      Based on overlap of patch regions (using center distance).
        
    Returns:
        Filtered list of patch pairs
    """
    if len(patch_pairs) == 0:
        return patch_pairs
    
    # Sort by probability (descending) - keep highest probability pairs
    sorted_pairs = sorted(patch_pairs, key=lambda x: x[3], reverse=True)
    
    # Get patch centers and radii for overlap calculation
    patch_info = {}
    for rank, idx1, idx2, prob_good, color_name, color_rgb in sorted_pairs:
        center1 = np.array(metadata['center_coordinates'][idx1])
        center2 = np.array(metadata['center_coordinates'][idx2])
        
        # Estimate patch radius from k_neighbors
        k_neighbors = metadata.get('k_neighbors', 1024)
        if points is not None and len(points) > 0:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(points)
                distances1, _ = tree.query(center1, k=min(k_neighbors, len(points)))
                distances2, _ = tree.query(center2, k=min(k_neighbors, len(points)))
                radius1 = distances1[-1] if len(distances1) > 0 else 0
                radius2 = distances2[-1] if len(distances2) > 0 else 0
            except (ImportError, NameError):
                # Fallback: use distance-based estimate
                distances1 = np.linalg.norm(points - center1, axis=1)
                distances2 = np.linalg.norm(points - center2, axis=1)
                sorted_dist1 = np.sort(distances1)
                sorted_dist2 = np.sort(distances2)
                radius1 = sorted_dist1[min(k_neighbors, len(points)-1)] if len(sorted_dist1) > 0 else 0
                radius2 = sorted_dist2[min(k_neighbors, len(points)-1)] if len(sorted_dist2) > 0 else 0
        else:
            # Fallback: use fixed radius estimate based on typical patch size
            radius1 = radius2 = 0.05  # Default radius estimate
        
        patch_info[(rank, idx1, idx2)] = {
            'center1': center1,
            'center2': center2,
            'radius1': radius1,
            'radius2': radius2,
            'prob_good': prob_good,
            'color_name': color_name,
            'color_rgb': color_rgb
        }
    
    # Apply NMS: keep pairs that don't overlap significantly with higher-ranked pairs
    kept_pairs = []
    kept_indices = set()  # Track which patch indices are already used
    
    for rank, idx1, idx2, prob_good, color_name, color_rgb in sorted_pairs:
        info = patch_info[(rank, idx1, idx2)]
        center1, center2 = info['center1'], info['center2']
        radius1, radius2 = info['radius1'], info['radius2']
        
        # Check overlap with already kept pairs
        overlap = False
        for kept_rank, kept_idx1, kept_idx2, _, _, _ in kept_pairs:
            kept_info = patch_info[(kept_rank, kept_idx1, kept_idx2)]
            kept_center1, kept_center2 = kept_info['center1'], kept_info['center2']
            kept_radius1, kept_radius2 = kept_info['radius1'], kept_info['radius2']
            
            # Check if patches overlap (share a patch or have close centers)
            # Overlap if: same patch index OR centers are very close
            if (idx1 == kept_idx1 or idx1 == kept_idx2 or 
                idx2 == kept_idx1 or idx2 == kept_idx2):
                overlap = True
                break
            
            # Check if patch centers are too close (within combined radius)
            dist1_to_kept1 = np.linalg.norm(center1 - kept_center1)
            dist1_to_kept2 = np.linalg.norm(center1 - kept_center2)
            dist2_to_kept1 = np.linalg.norm(center2 - kept_center1)
            dist2_to_kept2 = np.linalg.norm(center2 - kept_center2)
            
            min_dist = min(dist1_to_kept1, dist1_to_kept2, dist2_to_kept1, dist2_to_kept2)
            combined_radius = max(radius1, radius2) + max(kept_radius1, kept_radius2)
            
            if min_dist < combined_radius * iou_threshold:
                overlap = True
                break
        
        if not overlap:
            kept_pairs.append((rank, idx1, idx2, prob_good, color_name, color_rgb))
            kept_indices.add(idx1)
            kept_indices.add(idx2)
    
    # Re-rank kept pairs
    kept_pairs_sorted = sorted(kept_pairs, key=lambda x: x[3], reverse=True)
    final_pairs = []
    for new_rank, (old_rank, idx1, idx2, prob_good, color_name, color_rgb) in enumerate(kept_pairs_sorted, 1):
        final_pairs.append((new_rank, idx1, idx2, prob_good, color_name, color_rgb))
    
    print(f"\nNMS: Filtered {len(patch_pairs)} pairs -> {len(final_pairs)} pairs (removed {len(patch_pairs) - len(final_pairs)} overlapping)")
    
    return final_pairs


def visualize_patch_locations_in_pointcloud(points, colors, metadata, patch_pairs):
    """
    Visualize where multiple patch pairs are located in the original pointcloud.
    
    Args:
        points: Original pointcloud, shape (N, 3)
        colors: Original pointcloud colors, shape (N, 3) or None
        metadata: Patch metadata dictionary
        patch_pairs: List of tuples (rank, patch_idx1, patch_idx2, color_name, color_rgb)
                    where rank is 1, 2, 3, etc. and color_rgb is [R, G, B] in 0-1 range
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Cannot visualize pointcloud locations.")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    # Color the pointcloud (gray for background)
    if colors is not None:
        colors_normalized = colors / 255.0
        point_colors = colors_normalized * 0.2  # Make background darker
    else:
        point_colors = np.ones((len(points), 3)) * 0.2  # Dark gray background
    
    # Estimate patch radius from k_neighbors
    k_neighbors = metadata.get('k_neighbors', 1024)
    
    # Process each pair and color accordingly
    patch_centers = []
    center_spheres = []  # Store sphere geometries for enlarged center points
    
    # Estimate average point spacing for sphere size
    if len(points) > 1:
        # Calculate average distance to nearest neighbor
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            distances, _ = tree.query(points[:min(100, len(points))], k=2)  # Sample first 100 points
            avg_spacing = np.mean(distances[:, 1]) if len(distances) > 0 else 0.1
        except (ImportError, IndexError):
            # Fallback: estimate from point cloud bounds
            bounds = np.max(points, axis=0) - np.min(points, axis=0)
            avg_spacing = np.mean(bounds) / 100.0  # Rough estimate
    else:
        avg_spacing = 0.1
    
    # Sphere radius for center points (slightly enlarged)
    center_sphere_radius = avg_spacing * 1.5  # Make centers 1.5x larger than average spacing
    
    for rank, patch_idx1, patch_idx2, prob_good, color_name, color_rgb in patch_pairs:
        center1_idx = metadata['center_indices'][patch_idx1]
        center2_idx = metadata['center_indices'][patch_idx2]
        center1_coord = np.array(metadata['center_coordinates'][patch_idx1])
        center2_coord = np.array(metadata['center_coordinates'][patch_idx2])
        
        patch_centers.append(center1_coord)
        patch_centers.append(center2_coord)
        
        # Highlight patch center points (colour per pair)
        point_colors[center1_idx] = color_rgb
        point_colors[center2_idx] = color_rgb
        
        # Create enlarged spheres at center points (colour per pair)
        sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=center_sphere_radius, resolution=20)
        sphere1.translate(center1_coord)
        sphere1.paint_uniform_color(color_rgb)
        center_spheres.append(sphere1)
        
        sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=center_sphere_radius, resolution=20)
        sphere2.translate(center2_coord)
        sphere2.paint_uniform_color(color_rgb)
        center_spheres.append(sphere2)
        
        # Estimate patch radius for each center
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            distances1, _ = tree.query(center1_coord, k=min(k_neighbors, len(points)))
            distances2, _ = tree.query(center2_coord, k=min(k_neighbors, len(points)))
            patch_radius1 = distances1[-1] if len(distances1) > 0 else 0
            patch_radius2 = distances2[-1] if len(distances2) > 0 else 0
        except ImportError:
            distances1 = np.linalg.norm(points - center1_coord, axis=1)
            distances2 = np.linalg.norm(points - center2_coord, axis=1)
            sorted_dist1 = np.sort(distances1)
            sorted_dist2 = np.sort(distances2)
            patch_radius1 = sorted_dist1[min(k_neighbors, len(points)-1)] if len(sorted_dist1) > 0 else 0
            patch_radius2 = sorted_dist2[min(k_neighbors, len(points)-1)] if len(sorted_dist2) > 0 else 0
        
        # Highlight points within patch radius (use slightly dimmer version of center color)
        patch_color_dimmed = np.array(color_rgb) * 0.6  # Dimmer version for regions
        for i, point in enumerate(points):
            dist1 = np.linalg.norm(point - center1_coord)
            dist2 = np.linalg.norm(point - center2_coord)
            
            # Only color if not already colored by a higher rank pair
            if dist1 <= patch_radius1 * 1.2:
                if np.allclose(point_colors[i], [0.2, 0.2, 0.2], atol=0.1) or (colors is not None and np.allclose(point_colors[i], colors_normalized[i] * 0.2, atol=0.1)):
                    point_colors[i] = patch_color_dimmed
            if dist2 <= patch_radius2 * 1.2:
                if np.allclose(point_colors[i], [0.2, 0.2, 0.2], atol=0.1) or (colors is not None and np.allclose(point_colors[i], colors_normalized[i] * 0.2, atol=0.1)):
                    point_colors[i] = patch_color_dimmed
    
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # Create lines connecting patch centers for each pair
    line_sets = []
    for rank, patch_idx1, patch_idx2, prob_good, color_name, color_rgb in patch_pairs:
        center1_coord = np.array(metadata['center_coordinates'][patch_idx1])
        center2_coord = np.array(metadata['center_coordinates'][patch_idx2])
        
        # Create line set connecting the two centers
        line_points = np.array([center1_coord, center2_coord])
        line_indices = np.array([[0, 1]])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        
        # Color the line with the same color as the pair (slightly dimmer)
        line_color = np.array(color_rgb) * 0.8
        line_set.colors = o3d.utility.Vector3dVector([line_color])
        
        line_sets.append(line_set)
    
    # Set up visualization
    window_title = f"Top {len(patch_pairs)} Patch Pairs on ApeData1_merged"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1920, height=1080)
    vis.add_geometry(pcd)
    
    # Add line sets for connecting patch centers
    for line_set in line_sets:
        vis.add_geometry(line_set)
    
    # Add enlarged center spheres
    for sphere in center_spheres:
        vis.add_geometry(sphere)
    
    # Configure render options
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background
    
    # Set up view control - look at center of all patch centers
    if patch_centers:
        all_centers = np.array(patch_centers)
        midpoint = all_centers.mean(axis=0)
        view_control = vis.get_view_control()
        view_control.set_lookat(midpoint)
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_up([0.0, -1.0, 0.0])
        view_control.set_zoom(0.5)
    
    print("\n" + "="*70)
    print("Pointcloud Visualization - Top Patch Pairs")
    print("="*70)
    print("Legend:")
    for rank, _, _, prob_good, color_name, _ in patch_pairs:
        print(f"  - {color_name} (Rank {rank}): P(Good)={prob_good:.4f}")
    print("  - Dark gray: Rest of pointcloud")
    print("="*70)
    print("Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Shift + Mouse drag: Pan")
    print("  - Close window to exit")
    print("="*70 + "\n")
    
    vis.run()
    vis.destroy_window()


def create_all_pairs(patches):
    """
    Create all possible pairs from patches.
    
    Args:
        patches: List of patch arrays
        
    Returns:
        List of (patch_idx1, patch_idx2, patch1, patch2) tuples
    """
    pairs = []
    num_patches = len(patches)
    
    print(f"\nCreating all possible pairs from {num_patches} patches...")
    print(f"Total pairs: {num_patches * (num_patches - 1) // 2}")
    
    for i in range(num_patches):
        for j in range(i + 1, num_patches):
            pairs.append((i, j, patches[i], patches[j]))
    
    return pairs


def preprocess_pair(patch1, patch2, use_pca_alignment=False, pca_target_plane='xy'):
    """
    Preprocess a patch pair (same as dataloader preprocessing).
    
    Args:
        patch1: First patch, shape (N, 3)
        patch2: Second patch, shape (M, 3)
        use_pca_alignment: Whether to apply PCA alignment
        pca_target_plane: Target plane for PCA alignment
        
    Returns:
        Preprocessed concatenated pair, shape (N+M, 3) as torch.Tensor
    """
    # Convert to torch tensors
    patch1_tensor = torch.from_numpy(patch1).float()
    patch2_tensor = torch.from_numpy(patch2).float()
    
    # Concatenate patches
    combined = torch.cat([patch1_tensor, patch2_tensor], dim=0)  # (N+M, 3)
    
    # Apply preprocessing transforms (same order as dataloader)
    transforms = []
    
    # Always apply centering
    transforms.append(Centering())
    
    # Always apply normalization
    transforms.append(Normalization(method='unit_sphere'))
    
    # Optionally apply PCA alignment
    if use_pca_alignment:
        try:
            transforms.append(PCAAlignment(target_plane=pca_target_plane))
        except Exception as e:
            print(f"Warning: PCA alignment failed: {e}. Continuing without PCA.")
    
    # Compose and apply transforms
    if transforms:
        transform = Compose(transforms)
        combined = transform(combined)
    
    return combined


def test_pairs(model, pairs, device, use_pca_alignment=False, pca_target_plane='xy', batch_size=32, save_preprocessed=False):
    """
    Test all patch pairs and return predictions.
    
    Args:
        model: Trained PointNet2 model
        pairs: List of (idx1, idx2, patch1, patch2) tuples
        device: Device to run inference on
        use_pca_alignment: Whether to apply PCA alignment
        pca_target_plane: Target plane for PCA alignment
        batch_size: Batch size for inference
        save_preprocessed: If True, also return preprocessed patches for "Good" predictions
        
    Returns:
        If save_preprocessed=False: List of (idx1, idx2, prob_good, prob_bad, prediction) tuples
        If save_preprocessed=True: (results, preprocessed_patches_dict) where preprocessed_patches_dict
            maps (idx1, idx2) -> preprocessed_tensor (N+M, 3)
    """
    model.eval()
    results = []
    preprocessed_patches = {}  # Store preprocessed patches for "Good" predictions
    
    print(f"\nTesting {len(pairs)} patch pairs...")
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(pairs), batch_size), desc='Testing pairs'):
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]
        
        # Preprocess batch
        batch_points = []
        batch_indices = []
        batch_preprocessed = []  # Store preprocessed patches before padding
        
        for idx1, idx2, patch1, patch2 in batch_pairs:
            preprocessed = preprocess_pair(patch1, patch2, use_pca_alignment, pca_target_plane)
            batch_points.append(preprocessed)
            batch_indices.append((idx1, idx2))
            if save_preprocessed:
                batch_preprocessed.append(preprocessed.cpu().numpy())  # Store as numpy for saving
        
        # Stack into batch tensor (pad to same length if needed)
        # Find max length in batch
        max_len = max(p.shape[0] for p in batch_points)
        
        # Pad all to same length
        batch_tensor = []
        for p in batch_points:
            if p.shape[0] < max_len:
                # Pad with last point (replicate)
                padding = p[-1:].repeat(max_len - p.shape[0], 1)
                p_padded = torch.cat([p, padding], dim=0)
            else:
                p_padded = p
            batch_tensor.append(p_padded)
        
        batch_tensor = torch.stack(batch_tensor)  # (B, N, 3)
        
        # Move to device
        batch_tensor = batch_tensor.to(device)
        
        # Transpose for PointNet2: (B, N, 3) -> (B, 3, N)
        batch_tensor = batch_tensor.transpose(2, 1)
        
        # Run inference
        with torch.no_grad():
            pred, _ = model(batch_tensor)  # pred is log_softmax, shape (B, 2)
            
            # Convert to probabilities
            probs = torch.exp(pred)  # (B, 2)
            prob_good = probs[:, 0].cpu().numpy()  # Probability of class 0 (Good)
            prob_bad = probs[:, 1].cpu().numpy()   # Probability of class 1 (Bad)
            
            # Get predictions
            pred_choice = pred.data.max(1)[1].cpu().numpy()  # (B,)
            
            # Store results and preprocessed patches for "Good" predictions
            for i, (idx1, idx2) in enumerate(batch_indices):
                results.append((idx1, idx2, float(prob_good[i]), float(prob_bad[i]), int(pred_choice[i])))
                
                # Save preprocessed patch if predicted as "Good"
                if save_preprocessed and pred_choice[i] == 0:  # Class 0 = Good
                    preprocessed_patches[(idx1, idx2)] = batch_preprocessed[i]
    
    if save_preprocessed:
        return results, preprocessed_patches
    else:
        return results


def save_prediction_data(results, patches, output_file, object_name='unknown_object', patch_dir=None):
    """
    Save prediction data to JSON file for histogram analysis.
    
    Args:
        results: List of (idx1, idx2, prob_good, prob_bad, prediction) tuples
        patches: List of patch arrays (not used, but kept for API consistency)
        output_file: Path to output JSON file
        object_name: Name/identifier of the object
        patch_dir: Directory containing patches (optional metadata)
        
    Returns:
        Path to saved file
    """
    # Extract data from results
    prob_good_list = [prob_good for _, _, prob_good, _, _ in results]
    prob_bad_list = [prob_bad for _, _, _, prob_bad, _ in results]
    predictions = [pred for _, _, _, _, pred in results]
    
    # Calculate statistics
    total_pairs = len(results)
    good_predictions = sum(1 for p in predictions if p == 0)
    bad_predictions = sum(1 for p in predictions if p == 1)
    avg_prob_good = np.mean(prob_good_list) if prob_good_list else 0.0
    
    # Create statistics dictionary
    stats = {
        'object_id': object_name,
        'patch_dir': patch_dir,
        'total_pairs': total_pairs,
        'good_predictions': good_predictions,
        'bad_predictions': bad_predictions,
        'avg_prob_good': float(avg_prob_good),
        'avg_prob_bad': float(np.mean(prob_bad_list)) if prob_bad_list else 0.0,
        'prob_good_list': prob_good_list,
        'prob_bad_list': prob_bad_list,
        'predictions': predictions,
        # Also save detailed data for each pair
        'pairs': [
            {
                'patch_idx1': int(idx1),
                'patch_idx2': int(idx2),
                'prob_good': float(prob_good),
                'prob_bad': float(prob_bad),
                'prediction': int(pred)
            }
            for idx1, idx2, prob_good, prob_bad, pred in results
        ]
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return output_file


def save_preprocessed_graspable_pairs(results, preprocessed_patches, output_dir='preprocessed_graspable_results', min_confidence=0.90):
    """
    Save preprocessed patch pairs (after centering and normalization) that were predicted as "Good".
    
    Args:
        results: List of (idx1, idx2, prob_good, prob_bad, prediction) tuples
        preprocessed_patches: Dictionary mapping (idx1, idx2) -> preprocessed array (N+M, 3)
        output_dir: Directory to save preprocessed patch pairs
        min_confidence: Minimum confidence threshold (default: 0.90 for 90% confidence)
        
    Returns:
        Number of pairs saved
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for graspable pairs (predicted as Good, class 0) with confidence >= min_confidence
    if min_confidence >= 1.0:
        epsilon = 0.0001
    else:
        epsilon = 1e-6
    
    all_good_pairs = [(idx1, idx2, prob_good, prob_bad) 
                      for idx1, idx2, prob_good, prob_bad, pred in results 
                      if pred == 0]
    
    graspable_pairs = [(idx1, idx2, prob_good, prob_bad) 
                       for idx1, idx2, prob_good, prob_bad in all_good_pairs
                       if prob_good >= (min_confidence - epsilon) and (idx1, idx2) in preprocessed_patches]
    
    if len(graspable_pairs) == 0:
        print(f"\nNo preprocessed graspable pairs found with confidence >= {min_confidence:.1%} to save.")
        if len(all_good_pairs) > 0:
            max_confidence = max(prob_good for _, _, prob_good, _ in all_good_pairs)
            print(f"  Found {len(all_good_pairs)} pairs predicted as Good, but max confidence was {max_confidence:.6f}")
        return 0
    
    print(f"\nSaving {len(graspable_pairs)} preprocessed graspable patch pairs (confidence >= {min_confidence:.1%}) to {output_dir}/...")
    print(f"  Note: These are the exact patches that went into the model (after centering + normalization to unit sphere)")
    
    # Sort by probability of Good (descending) for consistent naming
    graspable_pairs_sorted = sorted(graspable_pairs, key=lambda x: x[2], reverse=True)
    
    saved_count = 0
    for pair_idx, (idx1, idx2, prob_good, prob_bad) in enumerate(graspable_pairs_sorted):
        # Get preprocessed patch (combined pair after preprocessing)
        preprocessed_combined = preprocessed_patches[(idx1, idx2)]
        
        # Create filename with pair index and patch indices
        filename_base = f"pair_{pair_idx:03d}_patch{idx1:03d}_patch{idx2:03d}_preprocessed"
        
        # Save the combined preprocessed patch pair
        output_path = os.path.join(output_dir, f"{filename_base}.npy")
        np.save(output_path, preprocessed_combined)
        
        saved_count += 1
        
        if (pair_idx + 1) % 50 == 0:
            print(f"  Saved {pair_idx + 1}/{len(graspable_pairs)} pairs...")
    
    print(f"Saved {saved_count} preprocessed graspable patch pairs (confidence >= {min_confidence:.1%}) to {output_dir}/")
    print(f"  Each pair saved as: <pair_XXX_patchYYY_patchZZZ>_preprocessed.npy (combined pair after preprocessing)")
    
    return saved_count


def save_graspable_patch_pairs(results, patches, output_dir='graspable_patch_results', min_confidence=0.90):
    """
    Save patch pairs predicted as "Good" (graspable) with confidence >= min_confidence.
    
    Args:
        results: List of (idx1, idx2, prob_good, prob_bad, prediction) tuples
        patches: List of patch arrays (original patches before preprocessing)
        output_dir: Directory to save patch pairs
        min_confidence: Minimum confidence threshold (default: 0.90 for 90% confidence)
        
    Returns:
        Number of pairs saved
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for graspable pairs (predicted as Good, class 0) with confidence >= min_confidence
    # Use epsilon for floating point comparison (values displayed as 1.0000 might be 0.9999 due to precision)
    # For 100% confidence, accept values >= 0.9999 (essentially 100% when rounded)
    if min_confidence >= 1.0:
        epsilon = 0.0001  # Accept values >= 0.9999 as "100%"
    else:
        epsilon = 1e-6  # Small epsilon for other thresholds
    
    all_good_pairs = [(idx1, idx2, prob_good, prob_bad) 
                      for idx1, idx2, prob_good, prob_bad, pred in results 
                      if pred == 0]
    
    graspable_pairs = [(idx1, idx2, prob_good, prob_bad) 
                       for idx1, idx2, prob_good, prob_bad in all_good_pairs
                       if prob_good >= (min_confidence - epsilon)]
    
    if len(graspable_pairs) == 0:
        print(f"\nNo graspable pairs found with confidence >= {min_confidence:.1%} to save.")
        if len(all_good_pairs) > 0:
            max_confidence = max(prob_good for _, _, prob_good, _ in all_good_pairs)
            print(f"  Found {len(all_good_pairs)} pairs predicted as Good, but max confidence was {max_confidence:.6f}")
        return 0
    
    print(f"\nSaving {len(graspable_pairs)} graspable patch pairs (confidence >= {min_confidence:.1%}) to {output_dir}/...")
    
    # Sort by probability of Good (descending) for consistent naming
    graspable_pairs_sorted = sorted(graspable_pairs, key=lambda x: x[2], reverse=True)
    
    saved_count = 0
    for pair_idx, (idx1, idx2, prob_good, prob_bad) in enumerate(graspable_pairs_sorted):
        # Get original patches (before preprocessing)
        patch1 = patches[idx1]
        patch2 = patches[idx2]
        
        # Create filename with pair index and patch indices
        # Format: pair_000_patch025_patch104_A.npy and pair_000_patch025_patch104_B.npy
        filename_base = f"pair_{pair_idx:03d}_patch{idx1:03d}_patch{idx2:03d}"
        
        # Save patch A
        patch_a_path = os.path.join(output_dir, f"{filename_base}_A.npy")
        np.save(patch_a_path, patch1)
        
        # Save patch B
        patch_b_path = os.path.join(output_dir, f"{filename_base}_B.npy")
        np.save(patch_b_path, patch2)
        
        saved_count += 1
        
        if (pair_idx + 1) % 50 == 0:
            print(f"  Saved {pair_idx + 1}/{len(graspable_pairs)} pairs...")
    
    print(f"Saved {saved_count} graspable patch pairs (confidence >= {min_confidence:.1%}) to {output_dir}/")
    print(f"  Each pair saved as: <pair_XXX_patchYYY_patchZZZ>_A.npy and _B.npy")
    
    return saved_count


def find_best_good_prediction(results):
    """
    Find the pair with the best "good" prediction (highest probability of class 0).
    
    Args:
        results: List of (idx1, idx2, prob_good, prob_bad, prediction) tuples
        
    Returns:
        (idx1, idx2, prob_good, prob_bad, prediction) tuple for best pair
    """
    # Sort by probability of "good" (class 0) in descending order
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    
    best_result = sorted_results[0]
    
    print(f"\nBest 'Good' prediction:")
    print(f"  Patch indices: {best_result[0]} and {best_result[1]}")
    print(f"  Probability of Good (class 0): {best_result[2]:.4f}")
    print(f"  Probability of Bad (class 1): {best_result[3]:.4f}")
    print(f"  Prediction: {'Good' if best_result[4] == 0 else 'Bad'}")
    
    # Print top 10
    print(f"\nTop 10 'Good' predictions:")
    for i, (idx1, idx2, prob_good, prob_bad, pred) in enumerate(sorted_results[:10]):
        print(f"  {i+1}. Patches {idx1:03d} & {idx2:03d}: "
              f"P(Good)={prob_good:.4f}, P(Bad)={prob_bad:.4f}, "
              f"Pred={'Good' if pred == 0 else 'Bad'}")
    
    return best_result


def main():
    parser = argparse.ArgumentParser(description='Test all patch pairs and visualize best prediction')
    parser.add_argument('--model_path', type=str, default='best_model_bce.pth',
                       help='Path to trained model checkpoint [default: best_model.pth]')
    parser.add_argument('--patch_dir', type=str, default='ape_patches',
                       help='Directory containing patch files [default: ape_patches]')
    parser.add_argument('--use_pca_alignment', action='store_true',
                       help='Apply PCA alignment (same as training)')
    parser.add_argument('--pca_target_plane', type=str, default='xy', choices=['xy', 'xz', 'yz'],
                       help='Target plane for PCA alignment [default: xy]')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference [default: 32]')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto) [default: auto]')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to visualize [default: 1]')
    parser.add_argument('--save_graspable', action='store_true',
                       help='Save all graspable patch pairs (predicted as Good) to graspable_patch_results/')
    parser.add_argument('--graspable_output_dir', type=str, default='graspable_patch_results',
                       help='Directory to save graspable patch pairs [default: graspable_patch_results]')
    parser.add_argument('--min_confidence', type=float, default=0.90,
                       help='Minimum confidence threshold for saving graspable pairs (0.0-1.0) [default: 0.90]')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction data to JSON file for histogram analysis (auto-generates filename)')
    parser.add_argument('--object_name', type=str, default=None,
                       help='Object name/identifier for saved prediction data [default: inferred from patch_dir]')
    parser.add_argument('--save_preprocessed', action='store_true',
                       help='Save preprocessed patch pairs (after centering + normalization) that were predicted as Good')
    parser.add_argument('--preprocessed_output_dir', type=str, default='preprocessed_graspable_results',
                       help='Directory to save preprocessed patch pairs [default: preprocessed_graspable_results]')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create model (same as training)
    model = get_model(num_class=2, normal_channel=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    if 'test_balanced_acc' in checkpoint:
        print(f"Model test balanced accuracy: {checkpoint['test_balanced_acc']:.4f}")
    
    # Load patches
    if not os.path.exists(args.patch_dir):
        print(f"Error: Patch directory not found: {args.patch_dir}")
        return
    
    patches = load_patches(args.patch_dir)
    
    if len(patches) == 0:
        print(f"Error: No patches found in {args.patch_dir}")
        return
    
    # Create all pairs
    pairs = create_all_pairs(patches)
    
    # Test all pairs
    if args.save_preprocessed:
        results, preprocessed_patches = test_pairs(model, pairs, device, args.use_pca_alignment, args.pca_target_plane, args.batch_size, save_preprocessed=True)
    else:
        results = test_pairs(model, pairs, device, args.use_pca_alignment, args.pca_target_plane, args.batch_size, save_preprocessed=False)
    
    # Calculate prediction statistics
    total_pairs = len(results)
    good_predictions = sum(1 for _, _, _, _, pred in results if pred == 0)
    bad_predictions = sum(1 for _, _, _, _, pred in results if pred == 1)
    
    print(f"\n" + "="*70)
    print("PREDICTION STATISTICS")
    print("="*70)
    print(f"Total pairs tested: {total_pairs}")
    print(f"Predicted as Good (class 0): {good_predictions} ({good_predictions/total_pairs*100:.2f}%)")
    print(f"Predicted as Bad (class 1): {bad_predictions} ({bad_predictions/total_pairs*100:.2f}%)")
    print("="*70)
    
    # Calculate average probabilities
    avg_prob_good = np.mean([prob_good for _, _, prob_good, _, _ in results])
    avg_prob_bad = np.mean([prob_bad for _, _, _, prob_bad, _ in results])
    print(f"\nAverage probabilities across all pairs:")
    print(f"  Average P(Good): {avg_prob_good:.4f}")
    print(f"  Average P(Bad): {avg_prob_bad:.4f}")
    print("="*70)
    
    # Save prediction data if requested
    if args.save_predictions:
        # Infer object name from patch_dir if not provided
        object_name = args.object_name
        if object_name is None:
            # Try to extract object name from patch_dir (e.g., "obj_000001_patches" -> "obj_000001")
            object_name = os.path.basename(args.patch_dir.rstrip('/'))
            if '_patches' in object_name:
                object_name = object_name.replace('_patches', '')
        
        # Auto-generate filename: predictions_{object_name}.json
        output_file = f"predictions_{object_name}.json"
        
        saved_file = save_prediction_data(results, patches, output_file,   # pyright: ignore[reportUndefinedVariable]
                                         object_name=object_name, patch_dir=args.patch_dir)
        print(f"\n✓ Saved prediction data to: {saved_file}")
        print(f"  Object: {object_name}")
        print(f"  Total pairs: {len(results)}")
    
    # Save graspable patch pairs if requested
    if args.save_graspable:
        saved_count = save_graspable_patch_pairs(results, patches, args.graspable_output_dir, min_confidence=args.min_confidence)
        if saved_count > 0:
            print(f"\n✓ Successfully saved {saved_count} graspable patch pairs")
            print(f"  Output directory: {args.graspable_output_dir}/")
            print(f"  Confidence threshold: >= {args.min_confidence:.1%}")
    
    # Save preprocessed patch pairs if requested
    if args.save_preprocessed:
        if 'preprocessed_patches' in locals():
            saved_count = save_preprocessed_graspable_pairs(results, preprocessed_patches, args.preprocessed_output_dir, min_confidence=args.min_confidence)
            if saved_count > 0:
                print(f"\n✓ Successfully saved {saved_count} preprocessed graspable patch pairs")
                print(f"  Output directory: {args.preprocessed_output_dir}/")
                print(f"  Confidence threshold: >= {args.min_confidence:.1%}")
        else:
            print("\n⚠️  Warning: --save_preprocessed was requested but preprocessed patches were not collected.")
    
    # Filter for pairs predicted as "Good" (pred == 0) and sort by P(Good) probability
    good_predictions = [(idx1, idx2, prob_good, prob_bad, pred) 
                        for idx1, idx2, prob_good, prob_bad, pred in results 
                        if pred == 0]
    
    if len(good_predictions) == 0:
        print("\n⚠️  Warning: No pairs predicted as 'Good' found. Cannot visualize.")
    else:
        print(f"\nFound {len(good_predictions)} pairs predicted as 'Good' (out of {len(results)} total)")
    
    # Sort by P(Good) probability (descending) - highest confidence first
    sorted_good_results = sorted(good_predictions, key=lambda x: x[2], reverse=True)
    
    # Take at least top 3 (or more if requested), but ensure we have at least 3 if available
    min_top_k = max(3, args.top_k)  # At least 3, or user's requested value if higher
    top_k_to_use = min(min_top_k, len(sorted_good_results))  # Don't exceed available good pairs
    
    if len(sorted_good_results) < 3:
        print(f"⚠️  Warning: Only {len(sorted_good_results)} pairs predicted as 'Good' found (requested at least 3)")
        top_k_to_use = len(sorted_good_results)
    
    top_k_results = sorted_good_results[:top_k_to_use]
    
    # Load metadata to show location in original pointcloud
    metadata = load_patch_metadata(args.patch_dir)
    
    if metadata and len(top_k_results) > 0:
        # Show location information for top K
        print(f"\n" + "="*70)
        print(f"LOCATION IN ORIGINAL POINTCLOUD - TOP {top_k_to_use} GOOD PAIRS")
        print("="*70)
        
        # Prepare top pairs with specific colors for ranks 1-5
        top_pairs = []
        
        # Specific colors for ranks 1-5
        rank_colors = {
            1: ("Green", [0.0, 1.0, 0.0]),
            2: ("Blue", [0.0, 0.0, 1.0]),
            3: ("Red", [1.0, 0.0, 0.0]),
            4: ("Pink", [1.0, 0.75, 0.8]),
            5: ("Purple", [0.5, 0.0, 0.5])
        }
        
        # Additional colors for ranks beyond 5
        additional_colors = {
            "Yellow": [1.0, 1.0, 0.0],
            "Orange": [1.0, 0.65, 0.0],
            "Cyan": [0.0, 1.0, 1.0],
            "Magenta": [1.0, 0.0, 1.0],
            "Lime": [0.5, 1.0, 0.0],
            "Teal": [0.0, 0.5, 0.5],
            "Coral": [1.0, 0.5, 0.31],
            "Gold": [1.0, 0.84, 0.0]
        }
        additional_color_names = list(additional_colors.keys())
        
        for rank, (idx1, idx2, prob_good, prob_bad, pred) in enumerate(top_k_results, 1):
            center1_idx = metadata['center_indices'][idx1]
            center2_idx = metadata['center_indices'][idx2]
            center1_coord = np.array(metadata['center_coordinates'][idx1])
            center2_coord = np.array(metadata['center_coordinates'][idx2])
            
            # Get color: specific for ranks 1-5, then cycle through additional colors
            if rank in rank_colors:
                color_name, color_rgb = rank_colors[rank]
            else:
                # For ranks beyond 5, cycle through additional colors
                color_idx = (rank - 6) % len(additional_color_names)
                color_name = additional_color_names[color_idx]
                color_rgb = additional_colors[color_name]
            
            top_pairs.append((rank, idx1, idx2, prob_good, color_name, color_rgb))
            
            print(f"\nRank {rank} ({color_name}) - Patches {idx1:03d} & {idx2:03d}:")
            print(f"  P(Good)={prob_good:.4f}, P(Bad)={prob_bad:.4f}, Pred={'Good' if pred == 0 else 'Bad'}")
            print(f"  Patch {idx1:03d} center:")
            print(f"    Original point index: {center1_idx}")
            print(f"    Coordinates: X={center1_coord[0]:.2f}, Y={center1_coord[1]:.2f}, Z={center1_coord[2]:.2f}")
            print(f"    Line number in {metadata['input_file']}: {center1_idx + 3} (after header lines)")
            print(f"  Patch {idx2:03d} center:")
            print(f"    Original point index: {center2_idx}")
            print(f"    Coordinates: X={center2_coord[0]:.2f}, Y={center2_coord[1]:.2f}, Z={center2_coord[2]:.2f}")
            print(f"    Line number in {metadata['input_file']}: {center2_idx + 3} (after header lines)")
            center_distance = np.linalg.norm(center1_coord - center2_coord)
            print(f"  Distance between patch centers: {center_distance:.2f}")
        
        # Apply NMS to filter overlapping pairs, ensuring we have at least 3 pairs
        print(f"\nApplying Non-Maximum Suppression to filter overlapping pairs...")
        original_file = metadata.get('input_file', 'ape.asc')
        original_points = None
        original_colors = None
        if os.path.exists(original_file):
            original_points, original_colors = load_original_pointcloud(original_file)
        
        # Start with top K pairs and apply NMS
        filtered_top_pairs = apply_nms_to_pairs(top_pairs, metadata, points=original_points, iou_threshold=0.3)
        
        # If we have fewer than 3 pairs after NMS, add more good pairs (P(Good) > 0.5) until we have at least 3
        min_pairs_needed = 3
        if len(filtered_top_pairs) < min_pairs_needed:
            print(f"\nNMS filtered to {len(filtered_top_pairs)} pairs. Adding more good pairs (P(Good) > 0.5) to reach at least {min_pairs_needed}...")
            
            # Get all good pairs with P(Good) > 0.5, sorted by probability
            all_good_pairs_above_threshold = [(idx1, idx2, prob_good, prob_bad, pred) 
                                              for idx1, idx2, prob_good, prob_bad, pred in sorted_good_results 
                                              if prob_good > 0.5]
            
            # Get indices of pairs already kept
            kept_pair_indices = set()
            for rank, idx1, idx2, prob_good, color_name, color_rgb in filtered_top_pairs:
                kept_pair_indices.add((idx1, idx2))
                kept_pair_indices.add((idx2, idx1))  # Add both orderings
            
            # Try to add more pairs incrementally, applying NMS each time
            candidate_pairs = []
            for idx1, idx2, prob_good, prob_bad, pred in all_good_pairs_above_threshold:
                # Skip if already in filtered pairs
                if (idx1, idx2) in kept_pair_indices or (idx2, idx1) in kept_pair_indices:
                    continue
                
                # Calculate rank based on current filtered pairs + candidates
                rank = len(filtered_top_pairs) + len(candidate_pairs) + 1
                
                # Use the same color assignment logic as initial top pairs
                rank_colors = {
                    1: ("Green", [0.0, 1.0, 0.0]),
                    2: ("Blue", [0.0, 0.0, 1.0]),
                    3: ("Red", [1.0, 0.0, 0.0]),
                    4: ("Pink", [1.0, 0.75, 0.8]),
                    5: ("Purple", [0.5, 0.0, 0.5])
                }
                additional_colors = {
                    "Yellow": [1.0, 1.0, 0.0],
                    "Orange": [1.0, 0.65, 0.0],
                    "Cyan": [0.0, 1.0, 1.0],
                    "Magenta": [1.0, 0.0, 1.0],
                    "Lime": [0.5, 1.0, 0.0],
                    "Teal": [0.0, 0.5, 0.5],
                    "Coral": [1.0, 0.5, 0.31],
                    "Gold": [1.0, 0.84, 0.0]
                }
                additional_color_names = list(additional_colors.keys())
                
                if rank in rank_colors:
                    color_name, color_rgb = rank_colors[rank]
                else:
                    color_idx = (rank - 6) % len(additional_color_names)
                    color_name = additional_color_names[color_idx]
                    color_rgb = additional_colors[color_name]
                
                candidate_pairs.append((rank, idx1, idx2, prob_good, color_name, color_rgb))
                
                # If we have enough candidates, try applying NMS
                if len(candidate_pairs) >= 10:  # Try with 10 candidates at a time
                    combined_pairs = filtered_top_pairs + candidate_pairs
                    temp_filtered = apply_nms_to_pairs(combined_pairs, metadata, points=original_points, iou_threshold=0.3)
                    
                    # Update kept indices
                    kept_pair_indices = set()
                    for rank, idx1, idx2, prob_good, color_name, color_rgb in temp_filtered:
                        kept_pair_indices.add((idx1, idx2))
                        kept_pair_indices.add((idx2, idx1))
                    
                    # Update filtered pairs with all non-overlapping ones found so far
                    filtered_top_pairs = temp_filtered
                    
                    # If we now have enough, we're done
                    if len(filtered_top_pairs) >= min_pairs_needed:
                        print(f"   Added pairs to reach {len(filtered_top_pairs)} total pairs after NMS")
                        break
                    else:
                        # Continue looking for more pairs
                        candidate_pairs = []
            
            # If we still don't have enough, try one more time with remaining candidates
            if len(filtered_top_pairs) < min_pairs_needed and len(candidate_pairs) > 0:
                combined_pairs = filtered_top_pairs + candidate_pairs
                temp_filtered = apply_nms_to_pairs(combined_pairs, metadata, points=original_points, iou_threshold=0.3)
                filtered_top_pairs = temp_filtered
                if len(filtered_top_pairs) >= min_pairs_needed:
                    print(f"   Added remaining pairs to reach {len(filtered_top_pairs)} total pairs after NMS")
            
            # Final check
            if len(filtered_top_pairs) < min_pairs_needed:
                print(f"⚠️  Warning: Only {len(filtered_top_pairs)} pairs available after NMS (requested at least {min_pairs_needed})")
                print(f"   Total good pairs with P(Good) > 0.5: {len(all_good_pairs_above_threshold)}")
            else:
                print(f"✓ Successfully selected {len(filtered_top_pairs)} pairs after NMS")
        
        # Re-sort filtered pairs by P(Good) to maintain ranking order
        filtered_top_pairs = sorted(filtered_top_pairs, key=lambda x: x[3], reverse=True)  # x[3] is prob_good
        
        # Print final pairs after NMS (so user sees what was kept)
        print(f"\n--- After NMS (final {len(filtered_top_pairs)} pairs) ---")
        xyz_triplets = []
        for rank, idx1, idx2, prob_good, color_name, color_rgb in filtered_top_pairs:
            print(f"  Rank {rank} ({color_name}): Patches {idx1:03d} & {idx2:03d}, P(Good)={prob_good:.4f}")
            c1 = metadata['center_coordinates'][idx1]
            c2 = metadata['center_coordinates'][idx2]
            xyz_triplets.append(f"{c1[0]:.2f},{c1[1]:.2f},{c1[2]:.2f}")
            xyz_triplets.append(f"{c2[0]:.2f},{c2[1]:.2f},{c2[2]:.2f}")
        if xyz_triplets and original_file:
            asc_for_viz = os.path.basename(original_file)
            xyz_str = " ".join(xyz_triplets)
            print(f"\nTo visualize these {len(filtered_top_pairs)} pairs (with lines) in visualize_points_on_pcd:")
            print(f"  python3 visualize_points_on_pcd.py {asc_for_viz} --xyz \"{xyz_str}\"")
        
        # Renumber ranks and reassign colors to ensure correct rank-specific colors (no duplicates)
        rank_colors = {
            1: ("Green", [0.0, 1.0, 0.0]),
            2: ("Blue", [0.0, 0.0, 1.0]),
            3: ("Red", [1.0, 0.0, 0.0]),
            4: ("Pink", [1.0, 0.75, 0.8]),
            5: ("Purple", [0.5, 0.0, 0.5])
        }
        additional_colors = {
            "Yellow": [1.0, 1.0, 0.0],
            "Orange": [1.0, 0.65, 0.0],
            "Cyan": [0.0, 1.0, 1.0],
            "Magenta": [1.0, 0.0, 1.0],
            "Lime": [0.5, 1.0, 0.0],
            "Teal": [0.0, 0.5, 0.5],
            "Coral": [1.0, 0.5, 0.31],
            "Gold": [1.0, 0.84, 0.0]
        }
        additional_color_names = list(additional_colors.keys())
        
        # Renumber ranks starting from 1 and assign colors based on new rank
        renumbered_pairs = []
        for new_rank, (old_rank, idx1, idx2, prob_good, old_color_name, old_color_rgb) in enumerate(filtered_top_pairs, 1):
            if new_rank in rank_colors:
                color_name, color_rgb = rank_colors[new_rank]
            else:
                color_idx = (new_rank - 6) % len(additional_color_names)
                color_name = additional_color_names[color_idx]
                color_rgb = additional_colors[color_name]
            renumbered_pairs.append((new_rank, idx1, idx2, prob_good, color_name, color_rgb))
        
        filtered_top_pairs = renumbered_pairs
        
        # Visualize top K pairs on original pointcloud
        if os.path.exists(original_file) and original_points is not None:
            if len(filtered_top_pairs) > 0:
                print(f"\nVisualizing top {len(filtered_top_pairs)} good pairs (after NMS) on original pointcloud...")
                visualize_patch_locations_in_pointcloud(
                    original_points, original_colors, metadata, filtered_top_pairs
                )
            else:
                print(f"\n⚠️  No pairs remaining after NMS. Cannot visualize.")
        else:
            print(f"\nOriginal pointcloud file not found: {original_file}")
            print("Skipping pointcloud visualization.")
    
    # Get best result for individual visualization (only if we have good predictions)
    if len(top_k_results) > 0:
        best_result = top_k_results[0]
        best_idx1, best_idx2, best_prob_good, best_prob_bad, best_pred = best_result
        
        # Load best patches for visualization
        best_patch1 = patches[best_idx1]
        best_patch2 = patches[best_idx2]
        
        print(f"\nVisualizing best good pair (patches {best_idx1:03d} & {best_idx2:03d})...")
        print(f"Probability of Good: {best_prob_good:.4f}, Probability of Bad: {best_prob_bad:.4f}")
        
        # Visualize using visualize_patches
        visualize_patches(
            [best_patch1, best_patch2],
            labels=[f'Patch {best_idx1:03d}', f'Patch {best_idx2:03d}'],
            use_open3d=True,
            point_size=1.0,
            show=True,
            use_pca_reorientation=args.use_pca_alignment,
            centralize_centroid=True
        )
        
        # Optionally visualize top K good pairs
        if top_k_to_use > 1:
            print(f"\nVisualizing top {top_k_to_use} good pairs...")
            top_k_patches = []
            top_k_labels = []
            
            for idx1, idx2, prob_good, prob_bad, pred in top_k_results:
                top_k_patches.append(patches[idx1])
                top_k_patches.append(patches[idx2])
                top_k_labels.append(f'P{idx1:03d} (P(Good)={prob_good:.3f})')
                top_k_labels.append(f'P{idx2:03d} (P(Good)={prob_good:.3f})')
            
            visualize_patches(
                top_k_patches,
                labels=top_k_labels,
                use_open3d=True,
                point_size=1.0,
                show=True,
                use_pca_reorientation=args.use_pca_alignment,
                centralize_centroid=True
            )
    else:
        print("\n⚠️  No pairs predicted as 'Good' found. Skipping individual visualization.")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

"""
python3 test_patch_pairs.py --patch_dir cracker_box_patches --save_graspable --min_confidence 0.95 --graspable_output_dir best_cracker_box_pairs --save_predictions --object_name cracker_box
"""
