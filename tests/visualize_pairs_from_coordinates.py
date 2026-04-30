#!/usr/bin/env python3
"""
Visualize patch pairs extracted from PLY file using coordinate file.

Takes a PLY file and a top_3_coordinates text file, extracts patches around
the Red and Blue point coordinates, and visualizes them on the point cloud.
"""

import os
import sys
import argparse
import numpy as np
import re
from typing import List, Tuple, Dict
from collections import namedtuple

# Try to import scipy for KDTree
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

# Try to import Open3D for interactive visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Install with: pip install open3d")

# Try matplotlib as fallback
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Install with: pip install matplotlib")

# Pair information structure
PairInfo = namedtuple('PairInfo', ['rank', 'pair_type', 'probability', 'predicted', 'gt', 
                                   'red_point', 'blue_point', 'gt_quality'])


def parse_ply_header(file_path: str) -> Tuple[int, int, int]:
    """
    Parse PLY file header to get number of vertices and faces.
    
    Args:
        file_path: Path to PLY file
    
    Returns:
        Tuple of (num_vertices, num_faces, header_end_bytes)
    """
    num_vertices = 0
    num_faces = 0
    header_end_bytes = 0
    
    with open(file_path, 'rb') as f:
        line_count = 0
        while True:
            line = f.readline()
            if not line:
                break
            
            # Decode line, handling both ASCII and binary formats
            try:
                line_str = line.decode('ascii').strip()
            except UnicodeDecodeError:
                # If we can't decode, we've reached binary data
                break
            
            line_count += 1
            
            if line_str.startswith('element vertex'):
                num_vertices = int(line_str.split()[-1])
            elif line_str.startswith('element face'):
                num_faces = int(line_str.split()[-1])
            elif line_str == 'end_header':
                header_end_bytes = f.tell()
                break
    
    return num_vertices, num_faces, header_end_bytes


def read_ply_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read PLY file and extract point cloud data.
    Supports both ASCII and binary PLY formats.
    
    Args:
        file_path: Path to PLY file
    
    Returns:
        Tuple of (points, colors, normals)
        - points: (N, 3) array of XYZ coordinates
        - colors: (N, 3) array of RGB colors (0-1 range)
        - normals: (N, 3) array of normals (if available)
    """
    import struct
    
    # Parse header to determine format
    num_vertices = 0
    header_end_bytes = 0
    is_binary = False
    properties = []
    
    with open(file_path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            
            try:
                line_str = line.decode('ascii').strip()
            except UnicodeDecodeError:
                break
            
            if line_str.startswith('format'):
                if 'binary' in line_str:
                    is_binary = True
            elif line_str.startswith('element vertex'):
                num_vertices = int(line_str.split()[-1])
            elif line_str.startswith('property'):
                parts = line_str.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_name, prop_type))
            elif line_str == 'end_header':
                header_end_bytes = f.tell()
                break
    
    # Read vertex data based on format
    data = []
    
    if is_binary:
        # Read binary data
        with open(file_path, 'rb') as f:
            f.seek(header_end_bytes)
            
            # Build struct format string based on properties
            fmt_chars = []
            for prop_name, prop_type in properties:
                if prop_type in ['float', 'float32']:
                    fmt_chars.append('f')
                elif prop_type in ['double', 'float64']:
                    fmt_chars.append('d')
                elif prop_type in ['uchar', 'uint8']:
                    fmt_chars.append('B')
                elif prop_type in ['int', 'int32']:
                    fmt_chars.append('i')
                elif prop_type in ['uint', 'uint32']:
                    fmt_chars.append('I')
                else:
                    fmt_chars.append('f')  # Default to float
            
            fmt = '<' + ''.join(fmt_chars)  # Little-endian
            vertex_size = struct.calcsize(fmt)
            
            for _ in range(num_vertices):
                vertex_bytes = f.read(vertex_size)
                if len(vertex_bytes) < vertex_size:
                    break
                vertex_data = struct.unpack(fmt, vertex_bytes)
                data.append(vertex_data)
    else:
        # Read ASCII data
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Skip header lines
            in_data = False
            count = 0
            for line in lines:
                line = line.strip()
                if line == 'end_header':
                    in_data = True
                    continue
                if in_data and count < num_vertices:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            vertex_data = [float(x) for x in parts]
                            data.append(vertex_data)
                            count += 1
                        except ValueError:
                            continue
    
    if len(data) == 0:
        raise ValueError(f"No valid vertex data found in {file_path}")
    
    data = np.array(data)
    
    # Extract points (always first 3 columns)
    points = data[:, :3]  # x, y, z
    
    # Extract colors if available (columns 6-8 or 3-5 depending on format)
    if data.shape[1] >= 9:
        # Format: x, y, z, nx, ny, nz, r, g, b
        colors = data[:, 6:9] / 255.0  # Normalize to 0-1
        normals = data[:, 3:6] if data.shape[1] >= 6 else None
    elif data.shape[1] >= 6:
        # Format: x, y, z, r, g, b
        colors = data[:, 3:6] / 255.0
        normals = None
    else:
        # No colors, use default gray
        colors = np.ones((len(points), 3)) * 0.7
        normals = None
    
    return points, colors, normals


def parse_coordinate_file(file_path: str) -> List[PairInfo]:
    """
    Parse the top_3_coordinates text file to extract pair information.
    
    Args:
        file_path: Path to coordinate file
    
    Returns:
        List of PairInfo namedtuples
    """
    pairs = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all RANK sections (handles both "RANK 1:" and "RANK 1 (NMS):" formats)
    rank_pattern = r'RANK (\d+)(?:\s*\([^)]*\))?:(.*?)(?=RANK \d+(?:\s*\([^)]*\))?:|$)'
    matches = re.finditer(rank_pattern, content, re.DOTALL)
    
    for match in matches:
        rank = int(match.group(1))
        rank_content = match.group(2)
        
        # Extract pair type
        pair_type_match = re.search(r'Pair Type:\s*(\S+)', rank_content)
        pair_type = pair_type_match.group(1) if pair_type_match else "unknown"
        
        # Extract model probability
        prob_match = re.search(r'Model Probability:\s*([\d.]+)', rank_content)
        probability = float(prob_match.group(1)) if prob_match else 0.0
        
        # Extract predicted and GT
        pred_match = re.search(r'Predicted:\s*(\w+),', rank_content)
        predicted = pred_match.group(1) if pred_match else "UNKNOWN"
        
        gt_match = re.search(r'GT:\s*(\w+)', rank_content)
        gt = gt_match.group(1) if gt_match else "UNKNOWN"
        
        # Extract Red Point coordinates
        red_match = re.search(r'Red Point \(on object\):\s*\[([^\]]+)\]', rank_content)
        if red_match:
            red_coords_str = red_match.group(1)
            red_coords = [float(x.strip()) for x in red_coords_str.split(',')]
            red_point = np.array(red_coords)
        else:
            continue  # Skip if no red point found
        
        # Extract Blue Point coordinates
        blue_match = re.search(r'Blue Point \(on object\):\s*\[([^\]]+)\]', rank_content)
        if blue_match:
            blue_coords_str = blue_match.group(1)
            blue_coords = [float(x.strip()) for x in blue_coords_str.split(',')]
            blue_point = np.array(blue_coords)
        else:
            continue  # Skip if no blue point found
        
        # Extract GT quality score
        quality_match = re.search(r'GT Quality Score:\s*([\d.]+)', rank_content)
        gt_quality = float(quality_match.group(1)) if quality_match else 0.0
        
        pair_info = PairInfo(
            rank=rank,
            pair_type=pair_type,
            probability=probability,
            predicted=predicted,
            gt=gt,
            red_point=red_point,
            blue_point=blue_point,
            gt_quality=gt_quality
        )
        pairs.append(pair_info)
    
    return pairs


def extract_patch_around_point(points: np.ndarray, center: np.ndarray, 
                               radius: float = None, k_neighbors: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a patch of points around a center point.
    
    Args:
        points: Full point cloud (N, 3)
        center: Center point (3,)
        radius: Radius in mm to extract points (if None, uses k_neighbors)
        k_neighbors: Number of nearest neighbors to extract (if radius is None)
    
    Returns:
        Tuple of (patch_points, patch_indices) where:
        - patch_points: (M, 3) array of patch point coordinates
        - patch_indices: (M,) array of indices into original points array
    """
    # Calculate distances from center to all points
    distances = np.linalg.norm(points - center, axis=1)
    
    if radius is not None:
        # Extract points within radius
        mask = distances <= radius
        patch_indices = np.where(mask)[0]
        patch = points[patch_indices]
    else:
        # Extract k nearest neighbors
        k = min(k_neighbors, len(points))
        nearest_indices = np.argpartition(distances, k)[:k]
        patch_indices = nearest_indices
        patch = points[nearest_indices]
    
    return patch, patch_indices


def visualize_with_open3d(points: np.ndarray, colors: np.ndarray, 
                          pairs: List[PairInfo], patch_radius: float = None,
                          k_neighbors: int = 1024, point_size: float = 1.0,
                          centralize_centroid: bool = False):
    """
    Visualize point cloud with patches using Open3D.
    
    Args:
        points: Full point cloud (N, 3)
        colors: Point colors (N, 3)
        pairs: List of PairInfo objects
        patch_radius: Radius for patch extraction (mm)
        k_neighbors: Number of neighbors if radius not specified
        point_size: Size of points in visualization
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for overlap resolution. Install with: pip install scipy")
    
    # Calculate centroid offset if centralization is requested
    centroid_offset = np.zeros(3)
    if centralize_centroid and len(pairs) > 0:
        # Calculate centroid of all pair centers (average of red and blue points)
        all_pair_centers = []
        for pair in pairs:
            pair_center = (pair.red_point + pair.blue_point) / 2.0
            all_pair_centers.append(pair_center)
        centroid_offset = np.mean(all_pair_centers, axis=0)
        print(f"Centralizing pairs: centroid offset = [{centroid_offset[0]:.2f}, {centroid_offset[1]:.2f}, {centroid_offset[2]:.2f}]")
        # Translate points
        points = points - centroid_offset
    
    # Define colors for patches based on rank: Rank 1=Green, Rank 2=Blue, Rank 3=Red
    rank_colors = {
        1: [0.0, 1.0, 0.0],  # Green for rank 1
        2: [0.0, 0.0, 1.0],  # Blue for rank 2
        3: [1.0, 0.0, 0.0],  # Red for rank 3
    }
    
    # Build KDTree for efficient point matching
    tree = cKDTree(points)
    
    # Extract all patches first (store with rank, patch type, and indices)
    all_patch_data = []  # List of (patch_points, patch_indices, rank, patch_name)
    for pair in pairs:
        # Apply centroid offset to pair points if centralization is enabled
        red_point = pair.red_point - centroid_offset if centralize_centroid else pair.red_point
        blue_point = pair.blue_point - centroid_offset if centralize_centroid else pair.blue_point
        
        # Extract red patch
        red_patch, red_indices = extract_patch_around_point(points, red_point, 
                                              radius=patch_radius, k_neighbors=k_neighbors)
        all_patch_data.append((red_patch, red_indices, pair.rank, f"rank_{pair.rank}_red"))
        
        # Extract blue patch
        blue_patch, blue_indices = extract_patch_around_point(points, blue_point,
                                                radius=patch_radius, k_neighbors=k_neighbors)
        all_patch_data.append((blue_patch, blue_indices, pair.rank, f"rank_{pair.rank}_blue"))
        
        print(f"Rank {pair.rank}: Extracted {len(red_patch)} points (red) and {len(blue_patch)} points (blue)")
    
    # Find overlapping points: map each point in original cloud to patches that use it
    point_to_patches = {}  # point_index -> list of (patch_idx, rank, patch_name)
    for patch_idx, (patch_pts, patch_indices, rank, patch_name) in enumerate(all_patch_data):
        for idx in patch_indices:
            if idx not in point_to_patches:
                point_to_patches[idx] = []
            point_to_patches[idx].append((patch_idx, rank, patch_name))
    
    # Create color array for all points - start with black
    point_colors = np.ones((len(points), 3)) * 0.0  # Black by default
    
    # Color points that belong to patches based on rank (higher rank takes precedence)
    for pt_idx in point_to_patches:
        # Get the patch with highest priority (lowest rank number)
        patches = point_to_patches[pt_idx]
        min_rank = min(p[1] for p in patches)
        # Find the patch with this rank (prefer red over blue if same rank)
        best_patch = None
        for p in patches:
            if p[1] == min_rank:
                if best_patch is None or 'red' in p[2]:
                    best_patch = p
        
        if best_patch:
            rank = best_patch[1]
            patch_name = best_patch[2]
            patch_color = rank_colors.get(rank, [0.5, 0.5, 0.5])
            # Use darker color for blue patches
            if 'blue' in patch_name:
                patch_color = [c * 0.7 for c in patch_color]
            point_colors[pt_idx] = patch_color
    
    # Create main point cloud with colored points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(point_colors.astype(np.float64))
    
    
    # Create sphere markers for center points and lines connecting pairs
    sphere_list = []
    line_list = []
    for pair in pairs:
        # Get color for this rank
        patch_color = rank_colors.get(pair.rank, [0.5, 0.5, 0.5])
        
        # Apply centroid offset to pair points if centralization is enabled
        red_point = pair.red_point - centroid_offset if centralize_centroid else pair.red_point
        blue_point = pair.blue_point - centroid_offset if centralize_centroid else pair.blue_point
        
        # Red point marker
        red_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        red_sphere.translate(red_point)
        red_sphere.paint_uniform_color(patch_color)
        sphere_list.append(red_sphere)
        
        # Blue point marker
        blue_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        blue_sphere.translate(blue_point)
        darker_color = [c * 0.7 for c in patch_color]
        blue_sphere.paint_uniform_color(darker_color)
        sphere_list.append(blue_sphere)
        
        # Create line connecting red and blue points
        line_points = np.array([red_point, blue_point])
        line_indices = np.array([[0, 1]])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points.astype(np.float64))
        line_set.lines = o3d.utility.Vector2iVector(line_indices.astype(np.int32))
        # Use a slightly darker version of the patch color for the line
        line_color = [c * 0.5 for c in patch_color]
        line_set.colors = o3d.utility.Vector3dVector([line_color])
        line_list.append(line_set)
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud with Patch Pairs")
    
    # Add main point cloud (with colored patch points)
    vis.add_geometry(pcd)
    
    # Add sphere markers
    for sphere in sphere_list:
        vis.add_geometry(sphere)
    
    # Add lines connecting pairs
    for line_set in line_list:
        vis.add_geometry(line_set)
    
    # Set rendering options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background
    
    print("\nVisualization window opened. Close the window to exit.")
    vis.run()
    vis.destroy_window()


def visualize_with_matplotlib(points: np.ndarray, colors: np.ndarray,
                              pairs: List[PairInfo], patch_radius: float = None,
                              k_neighbors: int = 1024, save_path: str = None,
                              centralize_centroid: bool = False):
    """
    Visualize point cloud with patches using matplotlib.
    
    Args:
        points: Full point cloud (N, 3)
        colors: Point colors (N, 3)
        pairs: List of PairInfo objects
        patch_radius: Radius for patch extraction (mm)
        k_neighbors: Number of neighbors if radius not specified
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required. Install with: pip install matplotlib")
    
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')  # White figure background
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')  # White axes background
    
    # Set 3D pane colors to white
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_alpha(1.0)
    ax.yaxis.pane.set_alpha(1.0)
    ax.zaxis.pane.set_alpha(1.0)
    
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for overlap resolution. Install with: pip install scipy")
    
    # Calculate centroid offset if centralization is requested
    centroid_offset = np.zeros(3)
    if centralize_centroid and len(pairs) > 0:
        # Calculate centroid of all pair centers (average of red and blue points)
        all_pair_centers = []
        for pair in pairs:
            pair_center = (pair.red_point + pair.blue_point) / 2.0
            all_pair_centers.append(pair_center)
        centroid_offset = np.mean(all_pair_centers, axis=0)
        print(f"Centralizing pairs: centroid offset = [{centroid_offset[0]:.2f}, {centroid_offset[1]:.2f}, {centroid_offset[2]:.2f}]")
        # Translate points
        points = points - centroid_offset
    
    # Plot main point cloud in black (subsampled for performance)
    subsample = max(1, len(points) // 10000)  # Show max 10k points
    ax.scatter(points[::subsample, 0], points[::subsample, 1], points[::subsample, 2],
              c='black', s=0.1, alpha=0.3)
    
    # Define colors for patches based on rank: Rank 1=Green, Rank 2=Blue, Rank 3=Red
    rank_colors = {
        1: ([0.0, 1.0, 0.0], [0.0, 0.5, 0.0]),  # Green (bright, dark) for rank 1
        2: ([0.0, 0.0, 1.0], [0.0, 0.0, 0.5]),  # Blue (bright, dark) for rank 2
        3: ([1.0, 0.0, 0.0], [0.5, 0.0, 0.0]),  # Red (bright, dark) for rank 3
    }
    
    # Build KDTree for efficient point matching
    tree = cKDTree(points)
    
    # Extract all patches first (store with rank, patch type, and indices)
    all_patch_data = []  # List of (patch_points, patch_indices, rank, patch_name)
    for pair in pairs:
        # Apply centroid offset to pair points if centralization is enabled
        red_point = pair.red_point - centroid_offset if centralize_centroid else pair.red_point
        blue_point = pair.blue_point - centroid_offset if centralize_centroid else pair.blue_point
        
        # Extract red patch
        red_patch, red_indices = extract_patch_around_point(points, red_point,
                                               radius=patch_radius, k_neighbors=k_neighbors)
        all_patch_data.append((red_patch, red_indices, pair.rank, f"rank_{pair.rank}_red"))
        
        # Extract blue patch
        blue_patch, blue_indices = extract_patch_around_point(points, blue_point,
                                               radius=patch_radius, k_neighbors=k_neighbors)
        all_patch_data.append((blue_patch, blue_indices, pair.rank, f"rank_{pair.rank}_blue"))
        
        print(f"Rank {pair.rank}: Extracted {len(red_patch)} points (red) and {len(blue_patch)} points (blue)")
    
    # Find overlapping points
    point_to_patches = {}  # point_index -> list of (patch_idx, rank)
    for patch_idx, (patch_pts, patch_indices, rank, _) in enumerate(all_patch_data):
        for idx in patch_indices:
            if idx not in point_to_patches:
                point_to_patches[idx] = []
            point_to_patches[idx].append((patch_idx, rank))
    
    # Resolve overlaps and plot
    for patch_idx, (patch_pts, patch_indices, rank, patch_name) in enumerate(all_patch_data):
        resolved_pts = []
        
        # Get rank color for this patch
        bright_color, dark_color = rank_colors.get(rank, ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]))
        # Use appropriate color based on patch type
        patch_color = bright_color if 'red' in patch_name else dark_color
        
        for i, pt_idx in enumerate(patch_indices):
            pt = patch_pts[i]
            
            # Check if this point is shared with other patches
            if pt_idx in point_to_patches and len(point_to_patches[pt_idx]) > 1:
                # This point is shared - check if we're the highest rank
                patches_sharing_point = point_to_patches[pt_idx]
                min_rank = min(p[1] for p in patches_sharing_point)  # Lower number = higher rank
                
                if rank > min_rank:
                    # We're lower rank - replace with nearest neighbor
                    dists, neighbor_indices = tree.query(pt, k=min(10, len(points)))
                    # Find first neighbor that's not the point itself and not already in another patch
                    neighbor_pt = None
                    for n_idx in neighbor_indices[1:]:  # Skip the point itself (index 0)
                        if n_idx not in point_to_patches or len(point_to_patches[n_idx]) == 0:
                            neighbor_pt = points[n_idx]
                            break
                    
                    if neighbor_pt is not None:
                        resolved_pts.append(neighbor_pt)
                    else:
                        # Fallback: use original point if no neighbor found
                        resolved_pts.append(pt)
                else:
                    # We're highest rank (or tied) - keep original point
                    resolved_pts.append(pt)
            else:
                # Not overlapping - use original point
                resolved_pts.append(pt)
        
        if len(resolved_pts) > 0:
            resolved_pts_array = np.array(resolved_pts)
            # Determine label based on patch name
            if 'red' in patch_name:
                label = f'Rank {rank} Red'
            else:
                label = f'Rank {rank} Blue'
            # Use rank color for all points in the patch
            ax.scatter(resolved_pts_array[:, 0], resolved_pts_array[:, 1], resolved_pts_array[:, 2],
                      c=[patch_color], s=5, alpha=0.8, label=label)
    
    # Mark center points and draw lines connecting pairs
    for pair in pairs:
        bright_color, dark_color = rank_colors.get(pair.rank, ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]))
        
        # Apply centroid offset to pair points if centralization is enabled
        red_point = pair.red_point - centroid_offset if centralize_centroid else pair.red_point
        blue_point = pair.blue_point - centroid_offset if centralize_centroid else pair.blue_point
        
        ax.scatter([red_point[0]], [red_point[1]], [red_point[2]],
                  c=[bright_color], s=100, marker='*', edgecolors='black', linewidths=1)
        ax.scatter([blue_point[0]], [blue_point[1]], [blue_point[2]],
                  c=[dark_color], s=100, marker='*', edgecolors='black', linewidths=1)
        
        # Draw line connecting red and blue points
        line_color = [c * 0.5 for c in bright_color]  # Use a darker version for the line
        ax.plot([red_point[0], blue_point[0]],
               [red_point[1], blue_point[1]],
               [red_point[2], blue_point[2]],
               color=line_color, linewidth=2, alpha=0.7, linestyle='--')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Point Cloud with Patch Pairs')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize patch pairs from PLY file using coordinate file'
    )
    parser.add_argument('ply_file', type=str, help='Path to PLY file (e.g., obj_000001.ply)')
    parser.add_argument('coord_file', type=str, 
                      help='Path to coordinate file (e.g., top_3_coordinates_obj_000001_000000_000000.txt)')
    parser.add_argument('--radius', type=float, default=None,
                      help='Radius in mm for patch extraction (default: use k_neighbors)')
    parser.add_argument('--k-neighbors', type=int, default=1024,
                      help='Number of nearest neighbors for patch extraction (default: 1024)')
    parser.add_argument('--point-size', type=float, default=1.0,
                      help='Point size for Open3D visualization (default: 1.0)')
    parser.add_argument('--method', type=str, choices=['open3d', 'matplotlib'], default='open3d',
                      help='Visualization method (default: open3d)')
    parser.add_argument('--save', type=str, default=None,
                      help='Save matplotlib figure to file (only for matplotlib method)')
    parser.add_argument('--rank', type=int, default=None,
                      help='Show only this rank (e.g., --rank 1 for rank 1 only)')
    parser.add_argument('--centralize-centroid', action='store_true',
                      help='Centralize the centroid of all pairs to origin (default: False)')
    
    args = parser.parse_args()
    
    # Validate files
    if not os.path.exists(args.ply_file):
        raise FileNotFoundError(f"PLY file not found: {args.ply_file}")
    if not os.path.exists(args.coord_file):
        raise FileNotFoundError(f"Coordinate file not found: {args.coord_file}")
    
    # Read PLY file
    print(f"Reading PLY file: {args.ply_file}")
    points, colors, normals = read_ply_file(args.ply_file)
    print(f"Loaded {len(points)} points from PLY file")
    
    # Parse coordinate file
    print(f"\nParsing coordinate file: {args.coord_file}")
    pairs = parse_coordinate_file(args.coord_file)
    print(f"Found {len(pairs)} pairs")
    
    # Filter by rank if specified
    if args.rank is not None:
        original_count = len(pairs)
        available_ranks = sorted(set(p.rank for p in pairs))
        pairs = [p for p in pairs if p.rank == args.rank]
        if len(pairs) == 0:
            print(f"\nWarning: No pairs found with rank {args.rank}")
            print(f"Available ranks: {available_ranks}")
            return
        print(f"Filtered to rank {args.rank}: {len(pairs)} pair(s) (from {original_count} total)")
    
    # Calculate centroid offset if centralization is requested (for display)
    centroid_offset_display = np.zeros(3)
    if args.centralize_centroid and len(pairs) > 0:
        all_pair_centers = []
        for pair in pairs:
            pair_center = (pair.red_point + pair.blue_point) / 2.0
            all_pair_centers.append(pair_center)
        centroid_offset_display = np.mean(all_pair_centers, axis=0)
    
    for pair in pairs:
        # Apply offset for display if centralization is enabled
        red_display = pair.red_point - centroid_offset_display if args.centralize_centroid else pair.red_point
        blue_display = pair.blue_point - centroid_offset_display if args.centralize_centroid else pair.blue_point
        print(f"  Rank {pair.rank}: {pair.pair_type}, Predicted={pair.predicted}, GT={pair.gt}")
        print(f"    Red: [{red_display[0]:.2f}, {red_display[1]:.2f}, {red_display[2]:.2f}]")
        print(f"    Blue: [{blue_display[0]:.2f}, {blue_display[1]:.2f}, {blue_display[2]:.2f}]")
    
    # Visualize
    print("\n" + "="*70)
    print("EXTRACTING AND VISUALIZING PATCHES")
    print("="*70)
    
    if args.method == 'open3d':
        if not OPEN3D_AVAILABLE:
            print("Open3D not available, falling back to matplotlib")
            args.method = 'matplotlib'
        else:
            visualize_with_open3d(points, colors, pairs, 
                                  patch_radius=args.radius,
                                  k_neighbors=args.k_neighbors,
                                  point_size=args.point_size,
                                  centralize_centroid=args.centralize_centroid)
            return
    
    if args.method == 'matplotlib':
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required. Install with: pip install matplotlib")
        visualize_with_matplotlib(points, colors, pairs,
                                patch_radius=args.radius,
                                k_neighbors=args.k_neighbors,
                                save_path=args.save,
                                centralize_centroid=args.centralize_centroid)


if __name__ == '__main__':
    main()

