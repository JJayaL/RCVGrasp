#!/usr/bin/env python3
"""
Extract 32x32 patches from a pointcloud file (PLY or ASCII format).

This script:
1. Loads a pointcloud from a PLY or ASCII file (auto-detects format)
2. Selects evenly distributed points across the pointcloud
3. Extracts a 32x32 patch (1024 points) around each selected point
4. Saves each patch as a .npy file in the output folder

Supported formats:
- PLY files (ASCII format): Standard PLY format with vertex data
- ASCII files: Custom format with header (//X Y Z R G B, <num_points>, then data)
"""

import numpy as np
import os
import json


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


def load_pointcloud(filepath):
    """
    Load pointcloud from file (auto-detects PLY, OBJ, or ASCII format).
    
    Args:
        filepath: Path to pointcloud file (.ply, .obj, or ASCII format)
        
    Returns:
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (optional, may be None)
    """
    # Check file extension or header to determine format
    if filepath.lower().endswith('.ply'):
        # Check if it's a PLY file by reading first line
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if first_line == 'ply':
                return load_ply_pointcloud(filepath)
    elif filepath.lower().endswith('.obj'):
        # OBJ file format
        return load_obj_pointcloud(filepath)
    
    # Try ASCII format
    try:
        return load_ascii_pointcloud(filepath)
    except (ValueError, IndexError) as e:
        # If ASCII fails, try other formats
        if not filepath.lower().endswith(('.ply', '.obj')):
            # Check if it might be PLY by reading first line
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                if first_line == 'ply':
                    return load_ply_pointcloud(filepath)
                # Check if it might be OBJ (look for 'v ' lines)
                f.seek(0)
                for line in f:
                    if line.strip().startswith('v '):
                        return load_obj_pointcloud(filepath)
        raise ValueError(f"Could not parse pointcloud file {filepath}. Error: {e}")


def extract_patch_around_point(points, center, k_neighbors=1024):
    """
    Extract a patch of k_neighbors points around a center point.
    
    Args:
        points: Full point cloud (N, 3)
        center: Center point (3,)
        k_neighbors: Number of nearest neighbors to extract (default: 1024 for 32x32)
    
    Returns:
        patch_points: (k_neighbors, 3) array of patch point coordinates
    """
    # Calculate distances from center to all points
    distances = np.linalg.norm(points - center, axis=1)
    
    # Extract k nearest neighbors
    k = min(k_neighbors, len(points))
    nearest_indices = np.argpartition(distances, k)[:k]
    
    # Sort by distance to get the actual k nearest
    nearest_distances = distances[nearest_indices]
    sorted_idx = np.argsort(nearest_distances)
    nearest_indices = nearest_indices[sorted_idx]
    
    patch = points[nearest_indices]
    
    return patch


def select_evenly_distributed_points(points, num_points=100):
    """
    Select evenly distributed points from the pointcloud.
    
    Args:
        points: Point cloud (N, 3)
        num_points: Number of points to select
        
    Returns:
        selected_points: (num_points, 3) array of selected center points
        selected_indices: (num_points,) array of indices into original pointcloud
    """
    n = len(points)
    num_points = min(num_points, n)
    
    # Select evenly spaced indices
    indices = np.linspace(0, n - 1, num_points, dtype=int)
    
    selected_points = points[indices]
    
    return selected_points, indices


def main():
    # Configuration
    input_file = 'ape.asc'
    output_dir = 'ape_patches'
    num_patches = 150
    patch_size = 32  # 32x32 = 1024 points
    k_neighbors = patch_size * patch_size
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading pointcloud from {input_file}...")
    points, colors = load_pointcloud(input_file)
    print(f"Loaded {len(points)} points")
    if colors is not None:
        print(f"  Colors available: {colors.shape}")
    
    print(f"Selecting {num_patches} evenly distributed points...")
    selected_centers, selected_indices = select_evenly_distributed_points(points, num_patches)
    print(f"Selected {len(selected_centers)} center points")
    
    print(f"Extracting {patch_size}x{patch_size} patches (k={k_neighbors} neighbors)...")
    
    # Store metadata about patch centers
    patch_metadata = {
        'input_file': input_file,
        'num_patches': num_patches,
        'patch_size': patch_size,
        'k_neighbors': k_neighbors,
        'center_indices': [],  # Original point indices in the ASCII file
        'center_coordinates': [],  # Center point coordinates (before scaling)
        'total_points': len(points)
    }
    
    for i, (center, center_idx) in enumerate(zip(selected_centers, selected_indices)):
        patch = extract_patch_around_point(points, center, k_neighbors)
        
        # Scale patch by x1000 before saving
        patch_scaled = patch * 750.0
        
        # Save patch as .npy file
        output_path = os.path.join(output_dir, f'patch_{i:03d}.npy')
        np.save(output_path, patch_scaled)
        
        # Store metadata
        patch_metadata['center_indices'].append(int(center_idx))
        patch_metadata['center_coordinates'].append(center.tolist())
        
        if (i + 1) % 10 == 0:
            print(f"  Extracted {i + 1}/{num_patches} patches...")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'patch_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(patch_metadata, f, indent=2)
    
    print(f"\nDone! Saved {num_patches} patches to {output_dir}/")
    print(f"Each patch contains {k_neighbors} points (32x32 grid)")
    print(f"Metadata saved to {metadata_path}")


if __name__ == '__main__':
    main()
