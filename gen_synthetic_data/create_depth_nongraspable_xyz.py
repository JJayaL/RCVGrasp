#!/usr/bin/env python3
"""
Create two depth maps representing parallel planes 3cm apart.
One plane is visible to camera 1, the other to camera 2.
The second plane's depth is transformed to camera 1's coordinate system.

Depth maps are generated using camera intrinsics from camera.json
to ensure proper spatial extent for 32x32 pixel patches.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import random
import json
import os

# Load camera intrinsics
def load_camera_intrinsics(camera_file='camera.json'):
    """Load camera intrinsics from JSON file."""
    if os.path.exists(camera_file):
        with open(camera_file, 'r') as f:
            return json.load(f)
    else:
        # Default intrinsics if file not found
        print(f"Warning: {camera_file} not found, using default intrinsics")
        return {
            'fx': 888.8890923394098,
            'fy': 888.8890923394098,
            'cx': 319.5,
            'cy': 239.5,
            'width': 640,
            'height': 480,
            'depth_scale': 1.0
        }

# Load camera intrinsics globally
CAMERA_INTRINSICS = load_camera_intrinsics()

def calculate_patch_extent(patch_size, depth, fx, fy, cx, cy):
    """
    Calculate the spatial extent (in meters) of a patch at given depth.
    
    Args:
        patch_size: Size of patch in pixels (e.g., 32)
        depth: Depth at which to calculate extent (in meters)
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point in pixels
        
    Returns:
        x_extent: Width of patch in meters at given depth
        y_extent: Height of patch in meters at given depth
    """
    # For a patch_size x patch_size patch centered in the image
    # Calculate the extent at the given depth using pinhole camera model
    # x = (u - cx) * z / fx
    
    # Half patch size
    half_patch = patch_size / 2.0
    
    # Spatial extent at given depth
    x_extent = (half_patch / fx) * depth
    y_extent = (half_patch / fy) * depth
    
    return x_extent, y_extent

def depth_to_xyz(depth_map, X, Y):
    """
    Convert depth map to XYZ point cloud.
    
    Args:
        depth_map: 2D array of depth values (Z coordinates)
        X: 2D meshgrid of X coordinates
        Y: 2D meshgrid of Y coordinates
        
    Returns:
        xyz: Nx3 array of XYZ coordinates (N = total number of points)
    """
    # Flatten all arrays
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = depth_map.flatten()
    
    # Stack into Nx3 array
    xyz = np.stack([x_flat, y_flat, z_flat], axis=1)
    
    return xyz

def pivot_sheets_together(xyz_a, xyz_b, rotation_angles=None):
    """
    Rotate both sheets together as a unit around one or more axes.
    This maintains their relative positions but changes their absolute orientation.
    
    Args:
        xyz_a: Nx3 array of XYZ coordinates for sheet A
        xyz_b: Nx3 array of XYZ coordinates for sheet B
        rotation_angles: Dict with rotation angles in degrees {'x': 0, 'y': 0, 'z': 0}
                        If None, random rotations are applied
    
    Returns:
        xyz_a_rotated: Rotated XYZ coordinates for sheet A
        xyz_b_rotated: Rotated XYZ coordinates for sheet B
    """
    if rotation_angles is None:
        # Random rotation around one or more axes
        axes_to_rotate = random.sample(['x', 'y', 'z'], k=random.randint(1, 3))
        rotation_angles = {'x': 0, 'y': 0, 'z': 0}
        for axis in axes_to_rotate:
            rotation_angles[axis] = random.uniform(0, 360)
    
    # Calculate center point between the two sheets (midpoint of their centroids)
    center_a = np.mean(xyz_a, axis=0)
    center_b = np.mean(xyz_b, axis=0)
    pivot_point = (center_a + center_b) / 2.0
    
    # Create rotation matrix from Euler angles
    rx = np.radians(rotation_angles.get('x', 0))
    ry = np.radians(rotation_angles.get('y', 0))
    rz = np.radians(rotation_angles.get('z', 0))
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Translate both sheets to origin (relative to pivot point)
    xyz_a_centered = xyz_a - pivot_point
    xyz_b_centered = xyz_b - pivot_point
    
    # Apply rotation
    xyz_a_rotated = (R @ xyz_a_centered.T).T
    xyz_b_rotated = (R @ xyz_b_centered.T).T
    
    # Translate back
    xyz_a_rotated = xyz_a_rotated + pivot_point
    xyz_b_rotated = xyz_b_rotated + pivot_point
    
    return xyz_a_rotated, xyz_b_rotated

def check_sheets_intersect(xyz1, xyz2, margin=1000.0):
    """
    Check if two sheets intersect in 3D space by checking bounding box overlap.
    
    Args:
        xyz1: Nx3 array of XYZ coordinates for sheet 1 (in mm)
        xyz2: Mx3 array of XYZ coordinates for sheet 2 (in mm)
        margin: Safety margin in mm to ensure no intersection (default: 1000mm = 1m)
        
    Returns:
        bool: True if sheets intersect (or are too close), False otherwise
    """
    # Calculate bounding boxes for both sheets
    min1 = xyz1.min(axis=0) - margin
    max1 = xyz1.max(axis=0) + margin
    min2 = xyz2.min(axis=0) - margin
    max2 = xyz2.max(axis=0) + margin
    
    # Check if bounding boxes overlap in all three dimensions
    # If they overlap in X, Y, AND Z, then the sheets intersect
    overlap_x = not (max1[0] < min2[0] or max2[0] < min1[0])
    overlap_y = not (max1[1] < min2[1] or max2[1] < min1[1])
    overlap_z = not (max1[2] < min2[2] or max2[2] < min1[2])
    
    # Sheets intersect if bounding boxes overlap in all three dimensions
    return overlap_x and overlap_y and overlap_z

def ensure_sheets_no_intersection(xyz_a, xyz_b, X_a, Y_a, z_a_final, z_b_final,
                                  center_a_x, center_a_y, center_a_z,
                                  center_b_x, center_b_y, center_b_z,
                                  min_separation_mm=5000.0):
    """
    Ensure two sheets do not intersect by adjusting center_b offsets if necessary.
    
    Args:
        xyz_a: Nx3 array of XYZ coordinates for sheet A (in mm)
        xyz_b: Mx3 array of XYZ coordinates for sheet B (may need adjustment, in mm)
        X_a: 2D meshgrid of X coordinates
        Y_a: 2D meshgrid of Y coordinates
        z_a_final: Final Z coordinates for sheet A
        z_b_final: Final Z coordinates for sheet B
        center_a_x, center_a_y, center_a_z: Center offsets for sheet A (in mm)
        center_b_x, center_b_y, center_b_z: Center offsets for sheet B (in mm, may be adjusted)
        min_separation_mm: Minimum separation in mm to ensure no intersection
        
    Returns:
        xyz_a: Adjusted XYZ coordinates for sheet A (usually unchanged)
        xyz_b: Adjusted XYZ coordinates for sheet B (with increased offset if needed)
        center_b_x: Final X offset for sheet B
        center_b_y: Final Y offset for sheet B
        center_b_z: Final Z offset for sheet B (may be adjusted)
    """
    # Calculate patch extent for reference (X_a and Y_a are already in mm)
    x_extent_mm = X_a.max() - X_a.min()
    y_extent_mm = Y_a.max() - Y_a.min()
    max_extent_mm = max(x_extent_mm, y_extent_mm)
    
    # Start with initial offsets
    new_center_b_x = center_b_x
    new_center_b_y = center_b_y
    new_center_b_z = center_b_z
    
    # Check for intersection and increase offset if needed
    max_attempts = 20
    for attempt in range(max_attempts):
        # Recalculate xyz_b with current offsets
        x_b_final = X_a + new_center_b_x
        y_b_final = Y_a + new_center_b_y
        z_b_final_new = z_b_final + new_center_b_z
        xyz_b_new = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final_new.flatten()], axis=1)
        
        # Check if sheets intersect
        if not check_sheets_intersect(xyz_a, xyz_b_new):
            return xyz_a, xyz_b_new, new_center_b_x, new_center_b_y, new_center_b_z
        
        # If they intersect, increase the offset
        # Calculate current distance
        current_distance_xy = np.sqrt((new_center_b_x - center_a_x)**2 + (new_center_b_y - center_a_y)**2)
        current_distance_z = abs(new_center_b_z - center_a_z)
        
        # Increase offset by scaling or adding minimum separation
        if current_distance_xy < min_separation_mm:
            # Need more XY separation
            scale_factor = 1.0 + (attempt + 1) * 0.3
            if current_distance_xy > 0:
                new_center_b_x = center_a_x + (new_center_b_x - center_a_x) * scale_factor
                new_center_b_y = center_a_y + (new_center_b_y - center_a_y) * scale_factor
            else:
                # If no offset, create new offset
                offset_direction = random.uniform(0, 2 * np.pi)
                new_center_b_x = center_a_x + np.cos(offset_direction) * min_separation_mm * (1.0 + attempt * 0.5)
                new_center_b_y = center_a_y + np.sin(offset_direction) * min_separation_mm * (1.0 + attempt * 0.5)
        
        # Also ensure Z separation
        if current_distance_z < min_separation_mm * 0.5:
            if new_center_b_z <= center_a_z:
                new_center_b_z = center_a_z + min_separation_mm * (1.0 + attempt * 0.3)
            else:
                new_center_b_z = center_a_z - min_separation_mm * (1.0 + attempt * 0.3)
    
    # If we still intersect after max attempts, use a very large offset
    # This should rarely happen, but ensures we never return intersecting sheets
    large_offset = 10.0 * max_extent_mm
    offset_direction = random.uniform(0, 2 * np.pi)
    new_center_b_x = center_a_x + np.cos(offset_direction) * large_offset
    new_center_b_y = center_a_y + np.sin(offset_direction) * large_offset
    new_center_b_z = center_a_z + min_separation_mm * 2.0
    
    x_b_final = X_a + new_center_b_x
    y_b_final = Y_a + new_center_b_y
    z_b_final_new = z_b_final + new_center_b_z
    xyz_b_final = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final_new.flatten()], axis=1)
    
    return xyz_a, xyz_b_final, new_center_b_x, new_center_b_y, new_center_b_z

def centralize_pair_centroid(xyz_a, xyz_b):
    """
    Translate both sheets so that the centroid of the combined point cloud is at the origin.
    
    Args:
        xyz_a: Nx3 array of XYZ coordinates for sheet 1
        xyz_b: Mx3 array of XYZ coordinates for sheet 2
        
    Returns:
        xyz_a_centered: Nx3 array with centroid at origin
        xyz_b_centered: Mx3 array with centroid at origin
    """
    # Combine both sheets
    combined = np.vstack([xyz_a, xyz_b])
    
    # Calculate centroid
    centroid = np.mean(combined, axis=0)
    
    # Translate both sheets so centroid is at origin
    xyz_a_centered = xyz_a - centroid
    xyz_b_centered = xyz_b - centroid
    
    return xyz_a_centered, xyz_b_centered

def save_ply(xyz, filename):
    """Save XYZ coordinates as PLY point cloud file."""
    with open(filename, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Write vertex data
        for point in xyz:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

def generate_texture_mm(texture_type, X_norm, Y_norm, patch_size, extreme=False):
    """Generate texture pattern for a single sheet in millimeters.
    
    Args:
        texture_type: Type of texture ("random", "wave", "noise", "grid", "spots", "none")
        X_norm: Normalized X coordinates [-1, 1]
        Y_norm: Normalized Y coordinates [-1, 1]
        patch_size: Size of patch (32)
        extreme: If True, generate extreme texture variations (> 8cm)
    
    Returns:
        Texture array in millimeters
    """
    if texture_type == "none" or texture_type == "no_texture":
        return np.zeros((patch_size, patch_size))
    elif texture_type == "random":
        if extreme:
            # Extreme texture: 40-80mm standard deviation (4-8cm)
            texture_std = random.uniform(40, 80)  # mm
            texture = np.random.normal(0, texture_std, (patch_size, patch_size))
            # Clip to ±2.5 standard deviations
            texture = np.clip(texture, -2.5*texture_std, 2.5*texture_std)
            return texture
        else:
            # Normal texture: 200-500mm variation
            return np.random.normal(0, random.uniform(200, 500), (patch_size, patch_size))
    elif texture_type == "wave":
        freq = random.uniform(3, 6)
        if extreme:
            amplitude = random.uniform(4000, 8000)  # mm (40-80cm)
        else:
            amplitude = random.uniform(300, 800)  # mm
        return amplitude * np.sin(freq * X_norm) * np.cos(freq * Y_norm)
    elif texture_type == "noise":
        if extreme:
            texture_std = random.uniform(40, 80)  # mm
            texture = np.random.normal(0, texture_std, (patch_size, patch_size))
            texture = np.clip(texture, -2.5*texture_std, 2.5*texture_std)
            return texture
        else:
            return np.random.normal(0, random.uniform(200, 500), (patch_size, patch_size))
    elif texture_type == "grid":
        grid_size = random.randint(6, 10)
        if extreme:
            amplitude = random.uniform(4000, 8000)  # mm
        else:
            amplitude = random.uniform(300, 800)  # mm
        return amplitude * (np.sin(grid_size * X_norm) + np.sin(grid_size * Y_norm))
    elif texture_type == "spots":
        texture = np.zeros((patch_size, patch_size))
        num_spots = random.randint(2, 5)
        for _ in range(num_spots):
            cx, cy = random.randint(0, patch_size-1), random.randint(0, patch_size-1)
            radius = random.randint(3, 6)
            if extreme:
                intensity = random.uniform(-8000, 8000)  # mm (80cm)
            else:
                intensity = random.uniform(-1500, 1500)  # mm
            y_ind, x_ind = np.ogrid[:patch_size, :patch_size]
            mask = (x_ind - cx)**2 + (y_ind - cy)**2 <= radius**2
            texture[mask] += intensity
        return texture
    else:
        return np.random.normal(0, random.uniform(200, 500), (patch_size, patch_size))

def generate_curvature_mm(curvature_type, X_norm, Y_norm, patch_size, too_high=False):
    """Generate curvature pattern for a single sheet in millimeters.
    
    Args:
        curvature_type: Type of curvature
        X_norm: Normalized X coordinates [-1, 1]
        Y_norm: Normalized Y coordinates [-1, 1]
        patch_size: Size of patch (32)
        too_high: If True, generate too high curvature
    
    Returns:
        Curvature array in millimeters
    """
    if curvature_type == "none":
        return np.zeros((patch_size, patch_size))
    elif curvature_type == "convex":
        radius_norm = random.uniform(0.5, 2.0)
        if too_high:
            amplitude = random.uniform(6000, 9000)  # mm (too high)
        else:
            amplitude = random.uniform(2000, 5000)  # mm
        return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
    elif curvature_type == "concave":
        radius_norm = random.uniform(0.5, 2.0)
        if too_high:
            amplitude = random.uniform(6000, 9000)  # mm
        else:
            amplitude = random.uniform(2000, 5000)  # mm
        return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
    elif curvature_type == "mixed":
        if random.random() > 0.5:
            radius_norm = random.uniform(0.5, 2.0)
            if too_high:
                amplitude = random.uniform(6000, 9000)
            else:
                amplitude = random.uniform(2000, 5000)
            return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
        else:
            radius_norm = random.uniform(0.5, 2.0)
            if too_high:
                amplitude = random.uniform(6000, 9000)
            else:
                amplitude = random.uniform(2000, 5000)
            return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
    elif curvature_type == "strong_convex":
        radius_norm = random.uniform(0.3, 1.0)
        if too_high:
            amplitude = random.uniform(7000, 10000)  # mm
        else:
            amplitude = random.uniform(4000, 8500)  # mm
        return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
    elif curvature_type == "strong_concave":
        radius_norm = random.uniform(0.3, 1.0)
        if too_high:
            amplitude = random.uniform(7000, 10000)  # mm
        else:
            amplitude = random.uniform(4000, 8500)  # mm
        return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
    elif curvature_type == "opposite_sphere":
        radius_norm = random.uniform(0.5, 1.5)
        if too_high:
            amplitude = random.uniform(6500, 9500)  # mm
        else:
            amplitude = random.uniform(3000, 7000)  # mm
        if random.random() > 0.5:
            return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
        else:
            return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
    else:
        return np.zeros((patch_size, patch_size))

def create_depth_maps(plane_separation=0.02, rotation_angles=None):
    """
    Create two 32x32 depth maps for planes with specified separation and rotation.
    Both planes can be rotated around different axes for more realistic appearance.
    
    Args:
        plane_separation: Distance between planes in meters
        rotation_angles: Dict with rotation angles for each plane {'plane1': {'x': 0, 'y': 0, 'z': 0}, 'plane2': {'x': 0, 'y': 0, 'z': 0}}
    
    Returns:
        depth1: Depth map for plane 1 (camera 1 view)
        depth2: Depth map for plane 2 (camera 2 view, transformed to camera 1 coords)
    """
    # Parameters
    patch_size = 32
    base_depth = 0.85  # Base depth in meters (center of 0.8-0.9 range)
    
    # Get camera intrinsics
    fx = CAMERA_INTRINSICS['fx']
    fy = CAMERA_INTRINSICS['fy']
    cx = CAMERA_INTRINSICS['cx']
    cy = CAMERA_INTRINSICS['cy']
    
    # Calculate the spatial extent of a 32x32 patch at the base depth
    x_extent, y_extent = calculate_patch_extent(patch_size, base_depth, fx, fy, cx, cy)
    
    # Default rotation angles if not provided - both sheets rotate equally around same axes
    if rotation_angles is None:
        # Both sheets rotate equally around the same axes (maintains relative separation)
        shared_rotation = {
            'x': random.uniform(0, 360), 
            'y': random.uniform(0, 360), 
            'z': random.uniform(0, 360)
        }
        rotation_angles = {
            'plane1': shared_rotation,
            'plane2': shared_rotation  # Same rotation for both sheets
        }
    
    # Create coordinate grids for both planes based on camera intrinsics
    # The spatial extent depends on the depth and camera focal length
    x = np.linspace(-x_extent, x_extent, patch_size)
    y = np.linspace(-y_extent, y_extent, patch_size)
    X, Y = np.meshgrid(x, y)
    
    def create_rotation_matrix(angles):
        """Create rotation matrix from Euler angles (in degrees)."""
        rx, ry, rz = np.radians(angles['x']), np.radians(angles['y']), np.radians(angles['z'])
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        return Rz @ Ry @ Rx
    
    def apply_rotation_to_plane(X, Y, base_depth, rotation_matrix):
        """Apply rotation to plane coordinates and return depth map."""
        # Create 3D points on the plane
        points = np.stack([X.flatten(), Y.flatten(), np.full(X.size, base_depth)], axis=1)
        
        # Apply rotation
        rotated_points = (rotation_matrix @ points.T).T
        
        # Extract rotated coordinates
        x_rot = rotated_points[:, 0].reshape(patch_size, patch_size)
        y_rot = rotated_points[:, 1].reshape(patch_size, patch_size)
        z_rot = rotated_points[:, 2].reshape(patch_size, patch_size)
        
        # For no_tex: Allow full rotation effects but keep sheets clean (no texture/curvature)
        # Allow moderate depth variations from rotation to show the rotation clearly
        z_min, z_max = z_rot.min(), z_rot.max()
        if z_max - z_min > 0.15:  # If depth variation is too large (15cm threshold)
            # Scale down the depth variation to keep sheets clean but still show rotation
            z_rot = base_depth + (z_rot - base_depth) * (0.10 / (z_max - z_min))  # Allow 10cm variation
        
        return z_rot
    
    # Create rotation matrices for both planes
    R1 = create_rotation_matrix(rotation_angles['plane1'])
    R2 = create_rotation_matrix(rotation_angles['plane2'])
    
    # Apply rotations to create depth maps with controlled separation
    depth1 = apply_rotation_to_plane(X, Y, base_depth, R1)
    depth2 = apply_rotation_to_plane(X, Y, base_depth + plane_separation, R2)
    
    # STRICT separation control for no_tex sheets - ensure separation stays within 2-8cm range
    min_separation = 0.02  # 2cm minimum
    max_separation = 0.08   # 8cm maximum
    
    depth1_mean = np.mean(depth1)
    depth2_mean = np.mean(depth2)
    current_separation = depth2_mean - depth1_mean
    
    # Always enforce separation within 2-8cm range for no_tex sheets
    if current_separation < min_separation or current_separation > max_separation:
        print(f"Warning: No_tex separation out of range ({current_separation*100:.1f}cm), adjusting to {plane_separation*100:.1f}cm...")
        
        # Calculate target depths
        target_depth1 = base_depth
        target_depth2 = base_depth + plane_separation
        
        # Preserve rotation effects while fixing separation
        depth1_offset = depth1 - depth1_mean
        depth2_offset = depth2 - depth2_mean
        
        # Apply separation fix while preserving 80% of rotation variation
        depth1 = target_depth1 + depth1_offset * 0.8
        depth2 = target_depth2 + depth2_offset * 0.8
    
    # Final enforcement: ensure separation is exactly what we want
    final_separation = np.mean(depth2) - np.mean(depth1)
    if abs(final_separation - plane_separation) > 0.02:  # If off by more than 1cm
        # Force exact separation
        depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + plane_separation
    
    # Ensure depth2 is always deeper than depth1 (maintain minimum plane separation)
    depth2 = np.maximum(depth2, depth1 + plane_separation * 0.5)
    
    # Final safety check: ensure separation is positive and within range
    final_separation = np.mean(depth2) - np.mean(depth1)
    if final_separation < 0:
        print(f"Error: Negative separation detected ({final_separation*100:.1f}cm), fixing...")
        depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + plane_separation
    elif final_separation > max_separation:
        print(f"Error: Separation too large ({final_separation*100:.1f}cm), clamping to {max_separation*100:.1f}cm...")
        depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + max_separation
    
    # Final debug info
    final_separation = np.mean(depth2) - np.mean(depth1)
    print(f"Final separation: {final_separation*100:.2f}cm (target: {plane_separation*100:.2f}cm)")
    print(f"Final depth 1 range: {depth1.min():.4f} to {depth1.max():.4f} meters")
    print(f"Final depth 2 range: {depth2.min():.4f} to {depth2.max():.4f} meters")
    
    return depth1, depth2

def create_textured_depth_maps(plane_separation=0.02, texture_type="random", rotation_angles=None):
    """
    Create two 32x32 depth maps for parallel planes with textures.
    
    Args:
        plane_separation: Distance between planes in meters
        texture_type: Type of texture ("random", "wave", "noise", "grid", "spots")
    
    Returns:
        depth1: Depth map for plane 1 with texture
        depth2: Depth map for plane 2 with texture
    """
    # Parameters
    patch_size = 32
    base_depth = 0.85  # Base depth in meters (center of 0.8-0.9 range)
    
    # Get camera intrinsics
    fx = CAMERA_INTRINSICS['fx']
    fy = CAMERA_INTRINSICS['fy']
    cx = CAMERA_INTRINSICS['cx']
    cy = CAMERA_INTRINSICS['cy']
    
    # Calculate the spatial extent of a 32x32 patch at the base depth
    x_extent, y_extent = calculate_patch_extent(patch_size, base_depth, fx, fy, cx, cy)
    
    # Create coordinate grids for both planes based on camera intrinsics
    x = np.linspace(-x_extent, x_extent, patch_size)
    y = np.linspace(-y_extent, y_extent, patch_size)
    X, Y = np.meshgrid(x, y)
    
    # Normalize X, Y to [-1, 1] range for frequency-based textures
    X_norm = X / x_extent
    Y_norm = Y / y_extent
    
    # Generate texture patterns (consistent visible texture for thin sheets)
    if texture_type == "random":
        # Random noise texture - consistent visible surface variations but thin sheets
        texture1 = np.random.normal(0, 0.0025, (patch_size, patch_size))  # 1.5cm variation
        texture2 = np.random.normal(0, 0.0025, (patch_size, patch_size))
    elif texture_type == "wave":
        # Wave pattern - surface ripples (use normalized coordinates)
        freq = random.uniform(3, 6)
        texture1 = 0.0025 * np.sin(freq * X_norm) * np.cos(freq * Y_norm)  # 2cm amplitude
        texture2 = 0.0025 * np.sin(freq * X_norm + np.pi/4) * np.cos(freq * Y_norm + np.pi/4)
    elif texture_type == "noise":
        # Perlin-like noise - surface roughness
        texture1 = np.random.normal(0, 0.0025, (patch_size, patch_size))  # 1.5cm variation
        texture2 = np.random.normal(0, 0.0025, (patch_size, patch_size))
    elif texture_type == "grid":
        # Grid pattern - surface grooves (use normalized coordinates)
        grid_size = random.randint(6, 10)
        texture1 = 0.0025 * (np.sin(grid_size * X_norm) + np.sin(grid_size * Y_norm))  # 3cm amplitude
        texture2 = 0.0025 * (np.sin(grid_size * X_norm + np.pi/2) + np.sin(grid_size * Y_norm + np.pi/2))
    elif texture_type == "spots":
        # Spot pattern - surface bumps/dents
        texture1 = np.zeros((patch_size, patch_size))
        texture2 = np.zeros((patch_size, patch_size))
        num_spots = random.randint(2, 5)
        for _ in range(num_spots):
            cx, cy = random.randint(0, patch_size-1), random.randint(0, patch_size-1)
            radius = random.randint(3, 8)
            intensity = random.uniform(-0.008, 0.008)  # 0.8cm intensity
            y_ind, x_ind = np.ogrid[:patch_size, :patch_size]
            mask = (x_ind - cx)**2 + (y_ind - cy)**2 <= radius**2
            texture1[mask] += intensity
            texture2[mask] += intensity * random.uniform(0.5, 1.5)
    else:
        # Default to random
        texture1 = np.random.normal(0, 0.015, (patch_size, patch_size))  # 1.5cm variation
        texture2 = np.random.normal(0, 0.015, (patch_size, patch_size))
    
    # Apply textures as surface variations on the parallel planes
    plane2_depth = base_depth + plane_separation
    
    # Create surface textures (more pronounced variations)
    # Keep the base plane depth and add surface variations
    depth1 = base_depth + texture1
    depth2 = plane2_depth + texture2
    
    # Debug: Print texture statistics
    print(f"Texture 1 range: {texture1.min():.4f} to {texture1.max():.4f} meters")
    print(f"Texture 2 range: {texture2.min():.4f} to {texture2.max():.4f} meters")
    print(f"Depth 1 range: {depth1.min():.4f} to {depth1.max():.4f} meters")
    print(f"Depth 2 range: {depth2.min():.4f} to {depth2.max():.4f} meters")
    
    # Apply rotation if provided
    if rotation_angles is not None:
        # Use the same rotation logic as the main function
        def create_rotation_matrix(angles):
            rx, ry, rz = np.radians(angles['x']), np.radians(angles['y']), np.radians(angles['z'])
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            return Rz @ Ry @ Rx
        
        def apply_rotation_to_textured_plane(X, Y, base_depth, texture, rotation_matrix):
            points = np.stack([X.flatten(), Y.flatten(), np.full(X.size, base_depth)], axis=1)
            rotated_points = (rotation_matrix @ points.T).T
            z_rot = rotated_points[:, 2].reshape(patch_size, patch_size)
            
            # Add texture to rotated plane
            z_with_texture = z_rot + texture
            
            # DISABLE depth variation control for textured sheets to test
            # z_min, z_max = z_with_texture.min(), z_with_texture.max()
            # if z_max - z_min > 0.2:  # If depth variation is too large (20cm threshold)
            #     # Scale down the depth variation to keep sheets thin but preserve texture
            #     z_with_texture = base_depth + (z_with_texture - base_depth) * (0.15 / (z_max - z_min))  # Allow 15cm variation
            
            # Preserve texture variations while keeping reasonable depth range
            # Instead of clamping to absolute values, scale the variations to keep them visible
            z_min, z_max = z_with_texture.min(), z_with_texture.max()
            if z_max - z_min > 0.20:  # If depth variation is too large (20cm threshold)
                # Scale down the depth variation but preserve texture
                z_with_texture = base_depth + (z_with_texture - base_depth) * (0.08 / (z_max - z_min))  # Allow 8cm variation
            # Don't clamp depths - keep them as is to preserve textures
            
            return z_with_texture
        
        R1 = create_rotation_matrix(rotation_angles['plane1'])
        R2 = create_rotation_matrix(rotation_angles['plane2'])
        
        depth1 = apply_rotation_to_textured_plane(X, Y, base_depth, texture1, R1)
        depth2 = apply_rotation_to_textured_plane(X, Y, plane2_depth, texture2, R2)
        
    
    # STRICT separation control for textured sheets - ALWAYS enforce 2-8cm range
    min_separation = 0.02  # 2cm minimum
    max_separation = 0.08   # 8cm maximum
    
    depth1_mean = np.mean(depth1)
    depth2_mean = np.mean(depth2)
    current_separation = depth2_mean - depth1_mean
    
    # ALWAYS enforce separation within 2-8cm range for textured sheets
    if current_separation < min_separation or current_separation > max_separation:
        print(f"Warning: Extreme textured separation ({current_separation*100:.1f}cm), adjusting to preserve 2-8cm range...")
        
        # Calculate target depths to achieve proper separation
        target_depth1 = base_depth
        target_depth2 = base_depth + plane_separation
        
        # Preserve texture patterns while fixing separation
        depth1_offset = depth1 - depth1_mean
        depth2_offset = depth2 - depth2_mean
        
        # Apply separation fix while preserving texture variations
        depth1 = target_depth1 + depth1_offset
        depth2 = target_depth2 + depth2_offset
    
    # Lenient final enforcement for textured sheets
    final_separation = np.mean(depth2) - np.mean(depth1)
    if abs(final_separation - plane_separation) > 0.2:  # Only if off by more than 20cm
        # Force reasonable separation while preserving texture variations
        depth2_offset = depth2 - np.mean(depth2)
        depth2 = np.mean(depth1) + plane_separation + depth2_offset
    
    # Ensure depth2 is always deeper than depth1 (maintain minimum plane separation)
    # depth2 = np.maximum(depth2, depth1 + plane_separation * 0.1)
    
    # Lenient final enforcement for textured sheets
    final_separation = np.mean(depth2) - np.mean(depth1)
    if abs(final_separation - plane_separation) > 0.5:  # Only if off by more than 50cm
        # Force reasonable separation while preserving texture variations
        depth2_offset = depth2 - np.mean(depth2)
        depth2 = np.mean(depth1) + plane_separation + depth2_offset
    
    # Ensure depth2 is always deeper than depth1 (maintain minimum plane separation)
    depth2 = np.maximum(depth2, depth1 + plane_separation * 0.05)
    
    # Final safety check: ensure separation is positive and within range
    final_separation = np.mean(depth2) - np.mean(depth1)
    if final_separation < 0:
        print(f"Error: Negative separation detected ({final_separation*100:.1f}cm), fixing...")
        depth2_offset = depth2 - np.mean(depth2)
        depth2 = np.mean(depth1) + plane_separation + depth2_offset
    elif final_separation > 0.50:  # Allow up to 50cm separation for textured sheets
        print(f"Error: Separation too large ({final_separation*100:.1f}cm), clamping to 50cm...")
        depth2_offset = depth2 - np.mean(depth2)
        depth2 = np.mean(depth1) + 0.50 + depth2_offset
    
    return depth1, depth2

def create_curved_depth_maps(plane_separation=0.02, curvature_type="random", rotation_angles=None):
    """
    Create two 32x32 depth maps for parallel planes with slight curvature.
    
    Args:
        plane_separation: Distance between planes in meters
        curvature_type: Type of curvature ("random", "convex", "concave", "wave", "saddle")
    
    Returns:
        depth1: Depth map for plane 1 with curvature
        depth2: Depth map for plane 2 with curvature
    """
    # Parameters
    patch_size = 32
    base_depth = 0.85  # Base depth in meters (center of 0.8-0.9 range)
    
    # Get camera intrinsics
    fx = CAMERA_INTRINSICS['fx']
    fy = CAMERA_INTRINSICS['fy']
    cx = CAMERA_INTRINSICS['cx']
    cy = CAMERA_INTRINSICS['cy']
    
    # Calculate the spatial extent of a 32x32 patch at the base depth
    x_extent, y_extent = calculate_patch_extent(patch_size, base_depth, fx, fy, cx, cy)
    
    # Create coordinate grids for both planes based on camera intrinsics
    x = np.linspace(-x_extent, x_extent, patch_size)
    y = np.linspace(-y_extent, y_extent, patch_size)
    X, Y = np.meshgrid(x, y)
    
    # Generate curvature patterns (maximum curvature range for dramatic sheets)
    if curvature_type == "random":
        # Random curvature variations - visible but controlled
        curvature1 = np.random.normal(0, 0.005, (patch_size, patch_size))  # 3cm variation
        curvature2 = np.random.normal(0, 0.005, (patch_size, patch_size))
    elif curvature_type == "convex":
        # Convex curvature (bulging outward) - dramatic enough to be visible
        radius = random.uniform(0.03, 0.08)
        curvature1 = 0.06 * (X**2 + Y**2) / radius
        curvature2 = 0.06 * (X**2 + Y**2) / radius
    elif curvature_type == "concave":
        # Concave curvature (curving inward) - dramatic enough to be visible
        radius = random.uniform(0.03, 0.08)
        curvature1 = -0.06 * (X**2 + Y**2) / radius
        curvature2 = -0.06 * (X**2 + Y**2) / radius
    elif curvature_type == "wave":
        # Wave-like curvature - very flowy and dramatic
        freq_x = random.uniform(0.8, 3.0)
        freq_y = random.uniform(0.8, 3.0)
        amplitude = random.uniform(0.02, 0.030)
        curvature1 = amplitude * np.sin(freq_x * X) * np.cos(freq_y * Y)
        curvature2 = amplitude * np.sin(freq_x * X + np.pi/3) * np.cos(freq_y * Y + np.pi/3)
    elif curvature_type == "saddle":
        # Saddle curvature (hyperbolic paraboloid) - visible but not too extreme
        curvature1 = 0.15 * (X**2 - Y**2)
        curvature2 = 0.15 * (X**2 - Y**2)
    elif curvature_type == "flowy":
        # Very flowy, organic curvature - like flowing fabric
        freq1 = random.uniform(0.5, 2.0)
        freq2 = random.uniform(0.5, 2.0)
        freq3 = random.uniform(0.3, 1.5)
        amplitude = random.uniform(0.020, 0.050)
        curvature1 = amplitude * (np.sin(freq1 * X) * np.cos(freq2 * Y) + 
                                 0.7 * np.sin(freq3 * (X + Y)) * np.cos(freq3 * (X - Y)))
        curvature2 = amplitude * (np.sin(freq1 * X + np.pi/4) * np.cos(freq2 * Y + np.pi/4) + 
                                  0.7 * np.sin(freq3 * (X + Y) + np.pi/6) * np.cos(freq3 * (X - Y) + np.pi/6))
    elif curvature_type == "fabric":
        # Fabric-like drapery and folds
        freq_x = random.uniform(0.8, 2.5)
        freq_y = random.uniform(0.8, 2.5)
        amplitude = random.uniform(0.02, 0.045)
        # Multiple overlapping waves for fabric-like appearance
        curvature1 = amplitude * (np.sin(freq_x * X) * np.cos(freq_y * Y) + 
                                  0.6 * np.sin(2*freq_x * X) * np.cos(2*freq_y * Y) +
                                  0.4 * np.sin(freq_x * X + freq_y * Y) * np.cos(freq_x * X - freq_y * Y))
        curvature2 = amplitude * (np.sin(freq_x * X + np.pi/3) * np.cos(freq_y * Y + np.pi/3) + 
                                  0.6 * np.sin(2*freq_x * X + np.pi/3) * np.cos(2*freq_y * Y + np.pi/3) +
                                  0.4 * np.sin(freq_x * X + freq_y * Y + np.pi/3) * np.cos(freq_x * X - freq_y * Y + np.pi/3))
    elif curvature_type == "mixed":
        # Mixed curvature: one convex, one concave for interesting contrast
        radius1 = random.uniform(0.03, 0.08)
        radius2 = random.uniform(0.03, 0.08)
        # Plane 1: convex (bulging outward)
        curvature1 = 0.06 * (X**2 + Y**2) / radius1
        # Plane 2: concave (curving inward)
        curvature2 = -0.06 * (X**2 + Y**2) / radius2
    else:
        # Default to random
        curvature1 = np.random.normal(0, 0.03, (patch_size, patch_size))
        curvature2 = np.random.normal(0, 0.03, (patch_size, patch_size))
    
    # Apply curvature to base depths (maintaining thin sheet structure)
    plane2_depth = base_depth + plane_separation
    
    # Create curved surfaces - keep them as thin sheets
    # Apply curvature as surface variations, not as solid blocks
    depth1 = base_depth + curvature1
    depth2 = plane2_depth + curvature2
    
    # Apply rotation if specified (both sheets rotate equally with the same angles)
    if rotation_angles is not None:
        # Helper function to create rotation matrix
        def create_rotation_matrix(angles):
            """Create 3D rotation matrix from euler angles (in degrees)"""
            rx = np.radians(angles['x'])
            ry = np.radians(angles['y'])
            rz = np.radians(angles['z'])
            
            # Rotation matrix around X axis
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(rx), -np.sin(rx)],
                          [0, np.sin(rx), np.cos(rx)]])
            
            # Rotation matrix around Y axis
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                          [0, 1, 0],
                          [-np.sin(ry), 0, np.cos(ry)]])
            
            # Rotation matrix around Z axis
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                          [np.sin(rz), np.cos(rz), 0],
                          [0, 0, 1]])
            
            # Combined rotation
            return Rz @ Ry @ Rx
        
        # Helper function to apply rotation to curved plane
        def apply_rotation_to_curved_plane(X, Y, depth_map, rotation_matrix):
            """Apply rotation to all points in the curved depth map"""
            # Create 3D points from the curved depth map
            points = np.stack([X.flatten(), Y.flatten(), depth_map.flatten()], axis=1)
            
            # Apply rotation to all points
            rotated_points = (rotation_matrix @ points.T).T
            z_rotated = rotated_points[:, 2].reshape(patch_size, patch_size)
            
            # Preserve curvature variations while keeping reasonable depth range
            # Less aggressive scaling to preserve curvature better
            z_min, z_max = z_rotated.min(), z_rotated.max()
            mean_depth = np.mean(depth_map)
            
            if z_max - z_min > 0.12:  # If depth variation is very large (12cm threshold)
                # Scale down but preserve more of the curvature
                z_rotated = mean_depth + (z_rotated - mean_depth) * (0.08 / (z_max - z_min))  # Allow 8cm variation
            
            return z_rotated
        
        # Apply the SAME rotation to both sheets equally
        R = create_rotation_matrix(rotation_angles['plane1'])  # Use same rotation for both
        
        depth1 = apply_rotation_to_curved_plane(X, Y, depth1, R)
        depth2 = apply_rotation_to_curved_plane(X, Y, depth2, R)
        
        # After rotation, ALWAYS enforce separation within 2-8cm range
        depth1_mean = np.mean(depth1)
        depth2_mean = np.mean(depth2)
        current_sep = depth2_mean - depth1_mean
        
        # Preserve the depth variations (curvature) but adjust the mean separation
        depth1_offset = depth1 - depth1_mean
        depth2_offset = depth2 - depth2_mean
        
        # Set new mean depths to achieve target separation
        new_depth1_mean = base_depth
        new_depth2_mean = base_depth + plane_separation
        
        depth1 = new_depth1_mean + depth1_offset
        depth2 = new_depth2_mean + depth2_offset
        
        # Skip additional separation controls since we already handled it
        rotation_applied = True
    else:
        rotation_applied = False
        # Only apply conservative constraints if NOT rotated
    # More conservative constraints to prevent solid block appearance
    depth1 = np.maximum(depth1, base_depth - 0.06)  # Allow curvature but not too extreme
    depth2 = np.maximum(depth2, plane2_depth - 0.06)
    
    # Define separation limits for later use
    min_separation = 0.02  # 2cm minimum
    max_separation = 0.08   # 8cm maximum
    
    # STRICT separation control for curved sheets - only if rotation was NOT already applied
    if not rotation_applied:
        depth1_mean = np.mean(depth1)
        depth2_mean = np.mean(depth2)
        current_separation = depth2_mean - depth1_mean
        
        # Always enforce separation within 2-8cm range for curved sheets
        if current_separation < min_separation or current_separation > max_separation:
            print(f"Warning: Curved separation out of range ({current_separation*100:.1f}cm), adjusting to {plane_separation*100:.1f}cm...")
            
            # Calculate target depths
            target_depth1 = base_depth
            target_depth2 = base_depth + plane_separation
            
            # Preserve curvature effects while fixing separation
            depth1_offset = depth1 - depth1_mean
            depth2_offset = depth2 - depth2_mean
            
            # Apply separation fix while preserving 80% of curvature variation
            depth1 = target_depth1 + depth1_offset * 0.8
            depth2 = target_depth2 + depth2_offset * 0.8
        
        # Final enforcement: ensure separation is exactly what we want
        final_separation = np.mean(depth2) - np.mean(depth1)
        if abs(final_separation - plane_separation) > 0.02:  # If off by more than 1cm
            # Force exact separation
            depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + plane_separation
    
    # Only apply additional safety checks if rotation was NOT applied
    if not rotation_applied:
        # Ensure depth2 is always deeper than depth1 (maintain minimum plane separation)
        depth2 = np.maximum(depth2, depth1 + plane_separation * 0.5)
        
        # Final safety check: ensure separation is positive and within range
        final_separation = np.mean(depth2) - np.mean(depth1)
        if final_separation < 0:
            print(f"Error: Negative separation detected ({final_separation*100:.1f}cm), fixing...")
            depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + plane_separation
        elif final_separation > max_separation:
            print(f"Error: Separation too large ({final_separation*100:.1f}cm), clamping to {max_separation*100:.1f}cm...")
            depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + max_separation
    
    # Texture visibility is preserved by the rotation function above
    # No need to limit sheet thickness here
    
    return depth1, depth2

def create_multiple_depth_pairs():
    """
    Create multiple depth map pairs with varying separations from 2cm to 8cm.
    Saves them in organized folders with clean naming.
    """
    import os
    
    # Create main output directory
    output_dir = "depth_map_pairs"
    no_tex_dir = os.path.join(output_dir, "no_tex")
    os.makedirs(no_tex_dir, exist_ok=True)
    
    # Create PNG and NPY subdirectories
    png_dir = os.path.join(no_tex_dir, "depth_png")
    npy_dir = os.path.join(no_tex_dir, "depth_npy")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    
    # Define separations from 2cm to 8cm with 100 different values
    separations_cm = np.linspace(2, 5, 300)
    
    print(f"Creating 300 clean depth map pairs with separations from 2cm to 8cm")
    print(f"Systematic rotation across all 360 degrees for each axis")
    print(f"Clean rotation-only sheets (no textures, no surface variations)")
    print(f"Output directory: {no_tex_dir}/")
    print(f"PNG files: {png_dir}/")
    print(f"NPY files: {npy_dir}/")
    
    for i, sep_cm in enumerate(separations_cm, 1):
        sep_m = sep_cm / 100.0  # Convert cm to meters
        
        # Create systematic rotation angles for even distribution across 360 degrees
        # Use the separation index to create evenly distributed rotations
        angle_step = 360.0 / len(separations_cm)
        
        # Both sheets rotate equally around the same axes
        shared_rotation = {
            'x': (i * angle_step) % 360,
            'y': (i * angle_step * 1.5) % 360, 
            'z': (i * angle_step * 2.3) % 360
        }
        
        rotation_angles = {
            'plane1': shared_rotation,
            'plane2': shared_rotation  # Same rotation for both sheets
        }
        
        # Create depth maps with systematic rotation (no textures)
        depth1, depth2 = create_depth_maps(plane_separation=sep_m, rotation_angles=rotation_angles)
        
        # Convert to 16-bit PNG (depth in millimeters)
        depth1_mm = (depth1 * 1000).astype(np.uint16)
        depth2_mm = (depth2 * 1000).astype(np.uint16)
        
        # Save PNG files
        cv2.imwrite(os.path.join(png_dir, f"depth{i:03d}_a.png"), depth1_mm)
        cv2.imwrite(os.path.join(png_dir, f"depth{i:03d}_b.png"), depth2_mm)
        
        # Save NPY files
        np.save(os.path.join(npy_dir, f"depth{i:03d}_a.npy"), depth1)
        np.save(os.path.join(npy_dir, f"depth{i:03d}_b.npy"), depth2)
        
        # Print progress every 10 pairs
        if i % 10 == 0 or i == 1:
            print(f"Progress: {i}/100 - Separation {sep_cm:.2f}cm (depth{i:03d})")
            print(f"  Plane A depth: {depth1.min():.3f}m")
            print(f"  Plane B depth: {depth2.min():.3f}m")
            print(f"  Actual separation: {(depth2.min() - depth1.min())*100:.2f}cm")
    
    print(f"\nAll 100 non-textured depth map pairs created successfully!")
    print(f"Total pairs: {len(separations_cm)}")
    print(f"Separation range: 2.00cm - 8.00cm")
    print(f"Files saved in: {png_dir}/ and {npy_dir}/")
    print(f"File naming: depth001_a.png, depth001_b.png, ..., depth100_a.png, depth100_b.png")

def create_textured_depth_pairs():
    """
    Create 300 textured depth map pairs with varying separations and textures.
    Saves them in depth_tex folder.
    """
    import os
    
    # Create main output directory
    output_dir = "depth_map_pairs"
    tex_dir = os.path.join(output_dir, "depth_tex")
    os.makedirs(tex_dir, exist_ok=True)
    
    # Create PNG and NPY subdirectories
    png_dir = os.path.join(tex_dir, "depth_png")
    npy_dir = os.path.join(tex_dir, "depth_npy")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    
    # Define texture types
    texture_types = ["random", "wave", "noise", "grid", "spots"]
    
    # Define separations from 2cm to 8cm
    separations_cm = np.linspace(2, 5, 400)
    
    print(f"Creating 300 textured depth map pairs")
    print(f"Output directory: {tex_dir}/")
    print(f"PNG files: {png_dir}/")
    print(f"NPY files: {npy_dir}/")
    print(f"Texture types: {texture_types}")
    
    pair_count = 0
    for i in range(400):
        # Random separation (2-8cm)
        sep_cm = random.uniform(2, 5)
        sep_m = sep_cm / 100.0
        
        # Random texture type
        texture_type = random.choice(texture_types)
        
        # Create textured depth maps with equal rotation
        # Both sheets rotate equally around the same axes
        shared_rotation = {
            'x': random.uniform(0, 360),
            'y': random.uniform(0, 360), 
            'z': random.uniform(0, 360)
        }
        
        rotation_angles = {
            'plane1': shared_rotation,
            'plane2': shared_rotation
        }
        
        depth1, depth2 = create_textured_depth_maps(plane_separation=sep_m, texture_type=texture_type, rotation_angles=rotation_angles)
        
        # Convert to 16-bit PNG (depth in millimeters)
        depth1_mm = (depth1 * 1000).astype(np.uint16)
        depth2_mm = (depth2 * 1000).astype(np.uint16)
        
        # Save PNG files
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_a.png"), depth1_mm)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_b.png"), depth2_mm)
        
        # Save NPY files
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_a.npy"), depth1)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_b.npy"), depth2)
        
        # Print progress every 50 pairs
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {i+1}/300 - Separation {sep_cm:.2f}cm, Texture: {texture_type}")
            print(f"  Plane A depth: {depth1.min():.3f}m - {depth1.max():.3f}m")
            print(f"  Plane B depth: {depth2.min():.3f}m - {depth2.max():.3f}m")
            print(f"  Actual separation: {(depth2.mean() - depth1.mean())*100:.2f}cm")
    
    print(f"\nAll 300 textured depth map pairs created successfully!")
    print(f"Total pairs: 300")
    print(f"Separation range: 2.00cm - 8.00cm")
    print(f"Texture types used: {texture_types}")
    print(f"Files saved in: {png_dir}/ and {npy_dir}/")
    print(f"File naming: depth001_a.png, depth001_b.png, ..., depth300_a.png, depth300_b.png")

def create_angled_depth_maps(plane_separation=0.02, angle_type="random", texture_type="random", rotation_angles=None):
    """
    Create two 32x32 depth maps for planes at slight angles to each other with texture.
    
    Args:
        plane_separation: Distance between planes in meters
        angle_type: Type of angle ("random", "tilt_x", "tilt_y", "tilt_both", "rotation")
        texture_type: Type of texture ("random", "wave", "noise", "grid", "spots")
    
    Returns:
        depth1: Depth map for plane 1 (facing camera) with texture
        depth2: Depth map for plane 2 (at slight angle) with texture
    """
    # Parameters
    patch_size = 32
    base_depth = 0.85  # Base depth in meters (center of 0.8-0.9 range)
    
    # Get camera intrinsics
    fx = CAMERA_INTRINSICS['fx']
    fy = CAMERA_INTRINSICS['fy']
    cx = CAMERA_INTRINSICS['cx']
    cy = CAMERA_INTRINSICS['cy']
    
    # Calculate the spatial extent of a 32x32 patch at the base depth
    x_extent, y_extent = calculate_patch_extent(patch_size, base_depth, fx, fy, cx, cy)
    
    # Create coordinate grids for both planes based on camera intrinsics
    x = np.linspace(-x_extent, x_extent, patch_size)
    y = np.linspace(-y_extent, y_extent, patch_size)
    X, Y = np.meshgrid(x, y)
    
    # Normalize X, Y to [-1, 1] range for frequency-based textures
    X_norm = X / x_extent
    Y_norm = Y / y_extent
    
    # Generate texture patterns (consistent visible texture for thin sheets)
    if texture_type == "random":
        # Random noise texture - subtle surface variations
        texture1 = np.random.normal(0, 0.0025, (patch_size, patch_size))  # 0.5cm variation
        texture2 = np.random.normal(0, 0.0025, (patch_size, patch_size))
    elif texture_type == "wave":
        # Wave pattern - surface ripples (use normalized coordinates)
        freq = random.uniform(3, 6)
        texture1 = 0.0025 * np.sin(freq * X_norm) * np.cos(freq * Y_norm)  # 2cm amplitude
        texture2 = 0.0025 * np.sin(freq * X_norm + np.pi/4) * np.cos(freq * Y_norm + np.pi/4)
    elif texture_type == "noise":
        # Perlin-like noise - surface roughness
        texture1 = np.random.normal(0, 0.0025, (patch_size, patch_size))  # 0.5cm variation
        texture2 = np.random.normal(0, 0.0025, (patch_size, patch_size))
    elif texture_type == "grid":
        # Grid pattern - surface grooves (use normalized coordinates)
        grid_size = random.randint(6, 10)
        texture1 = 0.0025 * (np.sin(grid_size * X_norm) + np.sin(grid_size * Y_norm))  # 1cm amplitude
        texture2 = 0.0025 * (np.sin(grid_size * X_norm + np.pi/2) + np.sin(grid_size * Y_norm + np.pi/2))
    elif texture_type == "spots":
        # Spot pattern - surface bumps/dents
        texture1 = np.zeros((patch_size, patch_size))
        texture2 = np.zeros((patch_size, patch_size))
        num_spots = random.randint(2, 5)
        for _ in range(num_spots):
            cx, cy = random.randint(0, patch_size-1), random.randint(0, patch_size-1)
            radius = random.randint(3, 6)
            intensity = random.uniform(-0.005, 0.005)  # 0.5cm intensity
            y_ind, x_ind = np.ogrid[:patch_size, :patch_size]
            mask = (x_ind - cx)**2 + (y_ind - cy)**2 <= radius**2
            texture1[mask] += intensity
            texture2[mask] += intensity * random.uniform(0.5, 1.5)
    else:
        # Default to random
        texture1 = np.random.normal(0, 0.005, (patch_size, patch_size))  # 1.5cm variation
        texture2 = np.random.normal(0, 0.005, (patch_size, patch_size))
    
    # Generate angle variations (reduced tilt angles in degrees)
    if angle_type == "random":
        # Random angles - reduced range
        angle_x = random.uniform(-1, 1)  # Tilt around X-axis
        angle_y = 0  # Tilt around Y-axis
        angle_z = 0  # Rotation around Z-axis
    elif angle_type == "tilt_x":
        # Tilt around X-axis only - reduced range
        angle_x = random.uniform(-1, 1)
        angle_y = 0
        angle_z = 0
    elif angle_type == "tilt_y":
        # Tilt around Y-axis only - reduced range
        angle_x = 0
        angle_y = random.uniform(-1, 1)
        angle_z = 0
    elif angle_type == "tilt_both":
        # Tilt around both X and Y axes - reduced range
        angle_x = random.uniform(-1, 1)
        angle_y = random.uniform(-1, 1)
        angle_z = 0
    elif angle_type == "rotation":
        # Rotation around Z-axis only - reduced range
        angle_x = 0
        angle_y = 0
        angle_z = random.uniform(-1, 1)
    else:
        # Default to random - reduced range
        angle_x = random.uniform(-1, 1)
        angle_y = 0
        angle_z = 0
    
    # Convert angles to radians
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)
    
    # Create rotation matrices
    # Rotation around X-axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x_rad), -np.sin(angle_x_rad)],
                   [0, np.sin(angle_x_rad), np.cos(angle_x_rad)]])
    
    # Rotation around Y-axis
    Ry = np.array([[np.cos(angle_y_rad), 0, np.sin(angle_y_rad)],
                   [0, 1, 0],
                   [-np.sin(angle_y_rad), 0, np.cos(angle_y_rad)]])
    
    # Rotation around Z-axis
    Rz = np.array([[np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
                   [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Create tilted planes by varying depth based on position
    # This creates actual geometric tilting rather than just rotation
    
    # Amplify tilt effect for visibility (small patches need larger tilt amplification)
    tilt_amplifier = 2.5  # Amplify the tilt effect to make angles more visible
    
    # Plane 1: Tilted with texture
    # Apply tilt based on angle type
    if angle_type == "tilt_x":
        # Tilt around X-axis: depth varies with Y position
        tilt_factor1 = np.tan(angle_x_rad) * Y * tilt_amplifier
        depth1 = base_depth + texture1 + tilt_factor1
    elif angle_type == "tilt_y":
        # Tilt around Y-axis: depth varies with X position
        tilt_factor1 = np.tan(angle_y_rad) * X * tilt_amplifier
        depth1 = base_depth + texture1 + tilt_factor1
    elif angle_type == "tilt_both":
        # Tilt around both axes
        tilt_factor1 = (np.tan(angle_x_rad) * Y + np.tan(angle_y_rad) * X) * tilt_amplifier
        depth1 = base_depth + texture1 + tilt_factor1
    else:
        # Random or rotation: apply random tilt
        tilt_factor1 = (np.tan(angle_x_rad) * Y + np.tan(angle_y_rad) * X) * tilt_amplifier
        depth1 = base_depth + texture1 + tilt_factor1
    
    # Plane 2: Different angle to create noticeable angle BETWEEN sheets
    plane2_depth = base_depth + plane_separation
    
    # Apply a more different tilt to plane 2 to create noticeable angle between sheets
    if angle_type == "tilt_x":
        # More different tilt around X-axis
        angle2_x_rad = angle_x_rad * 0.5  # 50% of plane1 angle (bigger difference)
        tilt_factor2 = np.tan(angle2_x_rad) * Y * tilt_amplifier
        depth2 = plane2_depth + texture2 + tilt_factor2
    elif angle_type == "tilt_y":
        # More different tilt around Y-axis
        angle2_y_rad = angle_y_rad * 0.5  # 50% of plane1 angle (bigger difference)
        tilt_factor2 = np.tan(angle2_y_rad) * X * tilt_amplifier
        depth2 = plane2_depth + texture2 + tilt_factor2
    elif angle_type == "tilt_both":
        # Different tilt combination with opposite signs for maximum visible angle
        tilt_factor2 = (-np.tan(angle_x_rad * 0.4) * Y + np.tan(angle_y_rad * 0.4) * X) * tilt_amplifier
        depth2 = plane2_depth + texture2 + tilt_factor2
    else:
        # Random: apply more different random tilt
        angle2_x_rad = angle_x_rad * 0.4  # 40% of plane1 angle (bigger difference)
        angle2_y_rad = angle_y_rad * 0.4
        tilt_factor2 = (np.tan(angle2_x_rad) * Y + np.tan(angle2_y_rad) * X) * tilt_amplifier
        depth2 = plane2_depth + texture2 + tilt_factor2
    
    # STRICT separation control for angled sheets - ensure separation stays within 2-8cm range
    min_separation = 0.02  # 2cm minimum
    max_separation = 0.06   # 8cm maximum
    
    depth1_mean = np.mean(depth1)
    depth2_mean = np.mean(depth2)
    current_separation = depth2_mean - depth1_mean
    
    # Always enforce separation within 2-8cm range for angled sheets
    if current_separation < min_separation or current_separation > max_separation:
        print(f"Warning: Angled separation out of range ({current_separation*100:.1f}cm), adjusting to {plane_separation*100:.1f}cm...")
        
        # Calculate target depths
        target_depth1 = base_depth
        target_depth2 = base_depth + plane_separation
        
        # Preserve angle/texture effects while fixing separation
        depth1_offset = depth1 - depth1_mean
        depth2_offset = depth2 - depth2_mean
        
        # Apply separation fix while preserving 80% of angle/texture variation
        depth1 = target_depth1 + depth1_offset * 0.8
        depth2 = target_depth2 + depth2_offset * 0.8
    
    # Final enforcement: ensure separation is exactly what we want
    final_separation = np.mean(depth2) - np.mean(depth1)
    if abs(final_separation - plane_separation) > 0.02:  # If off by more than 1cm
        # Force exact separation
        depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + plane_separation
    
    # Ensure depth2 is always deeper than depth1 (maintain minimum plane separation)
    depth2 = np.maximum(depth2, depth1 + plane_separation * 0.5)
    
    # Final safety check: ensure separation is positive and within range
    final_separation = np.mean(depth2) - np.mean(depth1)
    if final_separation < 0:
        print(f"Error: Negative separation detected ({final_separation*100:.1f}cm), fixing...")
        depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + plane_separation
    elif final_separation > max_separation:
        print(f"Error: Separation too large ({final_separation*100:.1f}cm), clamping to {max_separation*100:.1f}cm...")
        depth2 = depth2 - np.mean(depth2) + np.mean(depth1) + max_separation
    
    return depth1, depth2

def create_curved_depth_pairs():
    """
    Create 300 curved depth map pairs with varying separations and curvatures.
    Saves them in depth_curved folder.
    """
    import os
    
    # Create main output directory
    output_dir = "depth_map_pairs"
    curved_dir = os.path.join(output_dir, "depth_curved")
    os.makedirs(curved_dir, exist_ok=True)
    
    # Create PNG and NPY subdirectories
    png_dir = os.path.join(curved_dir, "depth_png")
    npy_dir = os.path.join(curved_dir, "depth_npy")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    
    # Define curvature types
    curvature_types = ["random", "convex", "concave", "wave", "saddle", "flowy", "fabric", "mixed"]
    
    print(f"Creating 300 curved depth map pairs")
    print(f"Output directory: {curved_dir}/")
    print(f"PNG files: {png_dir}/")
    print(f"NPY files: {npy_dir}/")
    print(f"Curvature types: {curvature_types}")
    
    for i in range(400):
        # Random separation (2-8cm)
        sep_cm = random.uniform(2, 5)
        sep_m = sep_cm / 100.0
        
        # Random curvature type
        curvature_type = random.choice(curvature_types)
        
        # Create curved depth maps with equal rotation
        # Both sheets rotate equally around the same axes
        shared_rotation = {
            'x': random.uniform(0, 360),
            'y': random.uniform(0, 360), 
            'z': random.uniform(0, 360)
        }
        
        rotation_angles = {
            'plane1': shared_rotation,
            'plane2': shared_rotation
        }
        
        depth1, depth2 = create_curved_depth_maps(plane_separation=sep_m, curvature_type=curvature_type, rotation_angles=rotation_angles)
        
        # Convert to 16-bit PNG (depth in millimeters)
        depth1_mm = (depth1 * 1000).astype(np.uint16)
        depth2_mm = (depth2 * 1000).astype(np.uint16)
        
        # Save PNG files
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_a.png"), depth1_mm)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_b.png"), depth2_mm)
        
        # Save NPY files
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_a.npy"), depth1)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_b.npy"), depth2)
        
        # Print progress every 50 pairs
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {i+1}/300 - Separation {sep_cm:.2f}cm, Curvature: {curvature_type}")
            print(f"  Plane A depth: {depth1.min():.3f}m - {depth1.max():.3f}m")
            print(f"  Plane B depth: {depth2.min():.3f}m - {depth2.max():.3f}m")
            print(f"  Actual separation: {(depth2.mean() - depth1.mean())*100:.2f}cm")
    
    print(f"\nAll 300 curved depth map pairs created successfully!")
    print(f"Total pairs: 300")
    print(f"Separation range: 2.00cm - 8.00cm")
    print(f"Curvature types used: {curvature_types}")
    print(f"Files saved in: {png_dir}/ and {npy_dir}/")
    print(f"File naming: depth001_a.png, depth001_b.png, ..., depth300_a.png, depth300_b.png")

def create_angled_depth_pairs():
    """
    Create 400 angled depth map pairs with varying separations and angles.
    Saves them in depth_angled folder.
    """
    import os
    
    # Create main output directory
    output_dir = "depth_map_pairs"
    angled_dir = os.path.join(output_dir, "depth_angled")
    os.makedirs(angled_dir, exist_ok=True)
    
    # Create PNG and NPY subdirectories
    png_dir = os.path.join(angled_dir, "depth_png")
    npy_dir = os.path.join(angled_dir, "depth_npy")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    
    # Define angle types
    angle_types = ["random", "tilt_x", "tilt_y", "tilt_both", "rotation"]
    
    # Define texture types
    texture_types = ["random", "wave", "noise", "grid", "spots"]
    
    print(f"Creating 400 angled depth map pairs with texture")
    print(f"Output directory: {angled_dir}/")
    print(f"PNG files: {png_dir}/")
    print(f"NPY files: {npy_dir}/")
    print(f"Angle types: {angle_types}")
    print(f"Texture types: {texture_types}")
    
    for i in range(1000):
        # Random separation (2-8cm)
        sep_cm = random.uniform(2, 5)
        sep_m = sep_cm / 100.0
        
        # Random angle type
        angle_type = random.choice(angle_types)
        
        # Random texture type
        texture_type = random.choice(texture_types)
        
        # Create angled depth maps with equal rotation
        # Both sheets rotate equally around the same axes
        shared_rotation = {
            'x': random.uniform(0, 360),
            'y': random.uniform(0, 360), 
            'z': random.uniform(0, 360)
        }
        
        rotation_angles = {
            'plane1': shared_rotation,
            'plane2': shared_rotation
        }
        
        depth1, depth2 = create_angled_depth_maps(plane_separation=sep_m, angle_type=angle_type, texture_type=texture_type, rotation_angles=rotation_angles)
        
        # Convert to 16-bit PNG (depth in millimeters)
        depth1_mm = (depth1 * 1000).astype(np.uint16)
        depth2_mm = (depth2 * 1000).astype(np.uint16)
        
        # Save PNG files
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_a.png"), depth1_mm)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_b.png"), depth2_mm)
        
        # Save NPY files
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_a.npy"), depth1)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_b.npy"), depth2)
        
        # Print progress every 50 pairs
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {i+1}/400 - Separation {sep_cm:.2f}cm, Angle: {angle_type}, Texture: {texture_type}")
            print(f"  Plane A depth: {depth1.min():.3f}m - {depth1.max():.3f}m")
            print(f"  Plane B depth: {depth2.min():.3f}m - {depth2.max():.3f}m")
            print(f"  Actual separation: {(depth2.mean() - depth1.mean())*100:.2f}cm")
    
    print(f"\nAll 400 angled depth map pairs with texture created successfully!")
    print(f"Total pairs: 400")
    print(f"Separation range: 2.00cm - 8.00cm")
    print(f"Angle types used: {angle_types}")
    print(f"Texture types used: {texture_types}")
    print(f"Files saved in: {png_dir}/ and {npy_dir}/")
    print(f"File naming: depth001_a.png, depth001_b.png, ..., depth400_a.png, depth400_b.png")

def visualize_depth_maps(depth1, depth2):
    """Visualize the two depth maps."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Depth map 1
    im1 = axes[0, 0].imshow(depth1, cmap='viridis', aspect='equal')
    axes[0, 0].set_title('Depth Map 1 (Plane 1 - Camera 1)')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 0], label='Depth (m)')
    
    # Depth map 2
    im2 = axes[0, 1].imshow(depth2, cmap='viridis', aspect='equal')
    axes[0, 1].set_title('Depth Map 2 (Plane 2 - Camera 1 coords)')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[0, 1], label='Depth (m)')
    
    # Difference map
    diff_map = np.abs(depth2 - depth1)
    im3 = axes[0, 2].imshow(diff_map, cmap='hot', aspect='equal')
    axes[0, 2].set_title('Depth Difference Map')
    axes[0, 2].set_xlabel('X (pixels)')
    axes[0, 2].set_ylabel('Y (pixels)')
    plt.colorbar(im3, ax=axes[0, 2], label='Depth Diff (m)')
    
    # 3D visualization of plane 1
    ax1 = fig.add_subplot(2, 3, 4, projection='3d')
    x = np.linspace(0, 31, 32)
    y = np.linspace(0, 31, 32)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, depth1, alpha=0.7, cmap='viridis')
    ax1.set_title('3D View - Plane 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Depth (m)')
    
    # 3D visualization of plane 2
    ax2 = fig.add_subplot(2, 3, 5, projection='3d')
    ax2.plot_surface(X, Y, depth2, alpha=0.7, cmap='viridis')
    ax2.set_title('3D View - Plane 2')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Depth (m)')
    
    # Combined 3D view
    ax3 = fig.add_subplot(2, 3, 6, projection='3d')
    ax3.plot_surface(X, Y, depth1, alpha=0.5, cmap='viridis', label='Plane 1')
    ax3.plot_surface(X, Y, depth2, alpha=0.5, cmap='plasma', label='Plane 2')
    ax3.set_title('Combined 3D View')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Depth (m)')
    
    plt.tight_layout()
    return fig

def save_depth_maps(depth1, depth2, prefix="depth"):
    """Save depth maps as images and numpy arrays."""
    # Save as 16-bit PNG images (preserving actual depth values in millimeters)
    depth1_mm = (depth1 * 1000).astype(np.uint16)  # Convert meters to millimeters
    depth2_mm = (depth2 * 1000).astype(np.uint16)
    
    cv2.imwrite(f"{prefix}_plane1.png", depth1_mm)
    cv2.imwrite(f"{prefix}_plane2.png", depth2_mm)
    
    # Also save normalized 8-bit versions for visualization
    depth1_range = depth1.max() - depth1.min()
    depth2_range = depth2.max() - depth2.min()
    
    if depth1_range > 0:
        depth1_norm = ((depth1 - depth1.min()) / depth1_range * 255).astype(np.uint8)
    else:
        depth1_norm = np.full_like(depth1, 128, dtype=np.uint8)
    
    if depth2_range > 0:
        depth2_norm = ((depth2 - depth2.min()) / depth2_range * 255).astype(np.uint8)
    else:
        depth2_norm = np.full_like(depth2, 128, dtype=np.uint8)
    
    cv2.imwrite(f"{prefix}_plane1_vis.png", depth1_norm)
    cv2.imwrite(f"{prefix}_plane2_vis.png", depth2_norm)
    
    # Save as numpy arrays
    np.save(f"{prefix}_plane1.npy", depth1)
    np.save(f"{prefix}_plane2.npy", depth2)
    
    print(f"Depth maps saved as:")
    print(f"  - {prefix}_plane1.png (16-bit depth data) and {prefix}_plane1.npy")
    print(f"  - {prefix}_plane2.png (16-bit depth data) and {prefix}_plane2.npy")
    print(f"  - {prefix}_plane1_vis.png and {prefix}_plane2_vis.png (8-bit visualization)")

def print_depth_stats(depth1, depth2):
    """Print statistics about the depth maps."""
    print("\nDepth Map Statistics:")
    print("=" * 50)
    print(f"Plane 1 (Camera 1):")
    print(f"  Min depth: {depth1.min():.4f} m")
    print(f"  Max depth: {depth1.max():.4f} m")
    print(f"  Mean depth: {depth1.mean():.4f} m")
    print(f"  Std depth: {depth1.std():.4f} m")
    
    print(f"\nPlane 2 (Camera 1 coords):")
    print(f"  Min depth: {depth2.min():.4f} m")
    print(f"  Max depth: {depth2.max():.4f} m")
    print(f"  Mean depth: {depth2.mean():.4f} m")
    print(f"  Std depth: {depth2.std():.4f} m")
    
    print(f"\nDepth difference (Plane 2 - Plane 1):")
    diff = depth2 - depth1
    print(f"  Min difference: {diff.min():.4f} m")
    print(f"  Max difference: {diff.max():.4f} m")
    print(f"  Mean difference: {diff.mean():.4f} m")
    print(f"  Expected separation: 0.0200 m")

def create_bad_same_plane_pairs(num_pairs=100):
    """
    Category 1: Both sheets on the EXACT SAME PLANE (coplanar).
    The sheets are at the same Z depth but at DIFFERENT XY positions.
    Like two papers lying side-by-side on a table (horizontally, vertically, or diagonally separated).
    All pairs have spatial separation in X and/or Y directions but zero Z separation.
    
    Uses millimeter scale and saves in same format as good patches.
    """
    import os
    
    # Create output directory (same structure as good patches)
    output_dir = "bad_patches"
    category_dir = os.path.join(output_dir, "same_plane")
    os.makedirs(category_dir, exist_ok=True)
    
    # Parameters - use millimeter scale like good patches
    patch_size = 32
    num_points = patch_size * patch_size  # 1024 points
    
    # Patch A (cam1) ranges in mm (similar to good patches)
    patch_a_x_range = (-4000, 4000)  # mm
    patch_a_y_range = (-4000, 6000)   # mm
    patch_a_z_range = (30000, 65000)  # mm
    
    # Patch extents in mm - ensure no elongation
    base_extent = random.uniform(28000, 30000)  # mm
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    print(f"Creating {num_pairs} BAD pairs: SAME PLANE (coplanar sheets with XY spatial separation)")
    print(f"Patch size: {patch_size}x{patch_size} = {num_points} points")
    print(f"Coordinates in millimeters (mm)")
    print(f"Files will be saved in '{category_dir}' directory")
    print()
    
    for pair_idx in range(1, num_pairs + 1):
        # Initialize variables for tracking bend information
        face_toward = None
        bend_axis = None
        use_bend = False
        bend_sheet_1 = False
        bend_sheet_2 = False
        
        # Generate patch A (cam1) center
        center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
        center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
        center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
        
        # For same plane: patch B has same Z (or very small separation)
        # Small Z separation (0-10mm) to simulate "same plane" but not exactly coplanar
        small_z_sep = random.uniform(0, 10)  # mm
        center_b_z = center_a_z + small_z_sep
        
        # Create coordinate grids for patch A (relative to center, in mm)
        x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
        y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
        X_a, Y_a = np.meshgrid(x_a, y_a)
        
        # Normalize X, Y to [-1, 1] range for frequency-based textures/curvatures
        X_a_norm = X_a / (patch_extent_x / 2)
        Y_a_norm = Y_a / (patch_extent_y / 2)
        
        # Determine separation first (needed for bend logic)
        # Choose whether sheets are close together (interconnected) or reasonably apart
        # 30% chance for close sheets (interconnected), 70% for reasonably apart
        is_close = random.random() < 0.3
        
        # Choose whether sheets are side-by-side horizontally, vertically, or diagonally
        orientation = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        if is_close:
            # Interconnected sheets: edges should touch, not intersect through middle
            # Calculate offset so edges touch (approximately patch_extent distance)
            # Add small gap (0-2% of patch extent) to ensure edges touch but don't overlap
            edge_gap_factor = random.uniform(0.98, 1.02)  # 98-102% of extent (slight gap or slight overlap at edges only)
            
            if orientation == 'horizontal':
                # Edges touch horizontally: offset = patch_extent_x (so right edge of A touches left edge of B)
                xy_offset_x = random.choice([-1, 1]) * patch_extent_x * edge_gap_factor
                xy_offset_y = 0
                center_b_x = center_a_x + xy_offset_x
                center_b_y = center_a_y
            elif orientation == 'vertical':
                # Edges touch vertically: offset = patch_extent_y
                xy_offset_x = 0
                xy_offset_y = random.choice([-1, 1]) * patch_extent_y * edge_gap_factor
                center_b_x = center_a_x
                center_b_y = center_a_y + xy_offset_y
            else:  # diagonal
                # Edges touch diagonally: use average extent or both extents
                avg_extent = (patch_extent_x + patch_extent_y) / 2.0
                xy_offset_x = random.choice([-1, 1]) * avg_extent * edge_gap_factor
                xy_offset_y = random.choice([-1, 1]) * avg_extent * edge_gap_factor
                center_b_x = center_a_x + xy_offset_x
                center_b_y = center_a_y + xy_offset_y
        else:
            # Reasonably apart: 20k-50k mm (20-50m) - with bends
            apart_range = (20000, 50000)  # mm
            if orientation == 'horizontal':
                xy_offset_x = random.choice([-1, 1]) * random.uniform(apart_range[0], apart_range[1])
                xy_offset_y = 0
                center_b_x = center_a_x + xy_offset_x
                center_b_y = center_a_y
            elif orientation == 'vertical':
                xy_offset_x = 0
                xy_offset_y = random.choice([-1, 1]) * random.uniform(apart_range[0], apart_range[1])
                center_b_x = center_a_x
                center_b_y = center_a_y + xy_offset_y
            else:  # diagonal
                xy_offset_x = random.choice([-1, 1]) * random.uniform(apart_range[0], apart_range[1])
                xy_offset_y = random.choice([-1, 1]) * random.uniform(apart_range[0], apart_range[1])
                center_b_x = center_a_x + xy_offset_x
                center_b_y = center_a_y + xy_offset_y
        
        # Generate base angles for sheets (small angles, relatively parallel for same plane)
        base_angle1_x = random.uniform(-5, 5)
        base_angle1_y = random.uniform(-5, 5)
        base_angle2_x = random.uniform(-5, 5)
        base_angle2_y = random.uniform(-5, 5)
        
        # For close sheets: create interconnected bends (always apply, similar patterns, facing toward/away)
        # For reasonably apart: apply bends (80% chance)
        if is_close:
            # Interconnected sheets: both sheets always bent with similar patterns
            bend_sheet_1 = True
            bend_sheet_2 = True
            
            # Determine bend axis and direction based on relative positions
            # Bend should be from the edge FARTHER from the other sheet (not the touching edge)
            # Sheet 1 bends from its edge farthest from sheet 2
            # Sheet 2 bends from its edge farthest from sheet 1
            if orientation == 'horizontal':
                # Horizontal: bend along X axis
                bend_axis = 'x'
                # If sheet 2 is to the right (xy_offset_x > 0), sheet 1 is on the left
                # Sheet 1's farthest edge from sheet 2 is LEFT, Sheet 2's farthest edge from sheet 1 is RIGHT
                if xy_offset_x > 0:
                    # Sheet 2 is to the right: sheet 1 bends from LEFT (farthest from sheet 2), sheet 2 bends from RIGHT (farthest from sheet 1)
                    bend_from_left_1 = True   # Sheet 1: bend from left (farthest edge)
                    bend_from_left_2 = False   # Sheet 2: bend from right (farthest edge)
                else:
                    # Sheet 2 is to the left: sheet 1 bends from RIGHT (farthest from sheet 2), sheet 2 bends from LEFT (farthest from sheet 1)
                    bend_from_left_1 = False  # Sheet 1: bend from right (farthest edge)
                    bend_from_left_2 = True   # Sheet 2: bend from left (farthest edge)
            elif orientation == 'vertical':
                # Vertical: bend along Y axis
                bend_axis = 'y'
                # If sheet 2 is above (xy_offset_y > 0), sheet 1 is below
                # Sheet 1's farthest edge from sheet 2 is BOTTOM, Sheet 2's farthest edge from sheet 1 is TOP
                if xy_offset_y > 0:
                    # Sheet 2 is above: sheet 1 bends from BOTTOM (farthest from sheet 2), sheet 2 bends from TOP (farthest from sheet 1)
                    bend_from_left_1 = True   # Sheet 1: bend from bottom (farthest edge)
                    bend_from_left_2 = False   # Sheet 2: bend from top (farthest edge)
                else:
                    # Sheet 2 is below: sheet 1 bends from TOP (farthest from sheet 2), sheet 2 bends from BOTTOM (farthest from sheet 1)
                    bend_from_left_1 = False  # Sheet 1: bend from top (farthest edge)
                    bend_from_left_2 = True   # Sheet 2: bend from bottom (farthest edge)
            else:  # diagonal
                # Diagonal: choose dominant direction or use both
                if abs(xy_offset_x) > abs(xy_offset_y):
                    # X is dominant: use X axis
                    bend_axis = 'x'
                    if xy_offset_x > 0:
                        # Sheet 2 is to the right: sheet 1 bends from left, sheet 2 bends from right
                        bend_from_left_1 = True
                        bend_from_left_2 = False
                    else:
                        # Sheet 2 is to the left: sheet 1 bends from right, sheet 2 bends from left
                        bend_from_left_1 = False
                        bend_from_left_2 = True
                else:
                    # Y is dominant: use Y axis
                    bend_axis = 'y'
                    if xy_offset_y > 0:
                        # Sheet 2 is above: sheet 1 bends from bottom, sheet 2 bends from top
                        bend_from_left_1 = True
                        bend_from_left_2 = False
                    else:
                        # Sheet 2 is below: sheet 1 bends from top, sheet 2 bends from bottom
                        bend_from_left_1 = False
                        bend_from_left_2 = True
            
            # Transition point: at least 50% of points should be bent (facing toward/away), up to almost 90% bent
            # transition_point = 0.5 means 50% parallel, 50% bent
            # transition_point = 0.1 means 10% parallel, 90% bent (almost 90%)
            transition_point = random.uniform(0.1, 0.5)  # 10-50% parallel, 50-90% bent (at least 50% bent, up to almost 90%)
            
            # Create transition mask for bend - each sheet bends from its far edge
            if bend_axis == 'x':
                if bend_from_left_1:
                    transition_mask_1 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
                else:
                    transition_mask_1 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1)
                if bend_from_left_2:
                    transition_mask_2 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
                else:
                    transition_mask_2 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1)
            else:
                if bend_from_left_1:
                    transition_mask_1 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
                else:
                    transition_mask_1 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1)
                if bend_from_left_2:
                    transition_mask_2 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
                else:
                    transition_mask_2 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1)
            
            # Calculate direction from sheet 1 to sheet 2 for facing toward/away
            direction_to_sheet2 = np.array([xy_offset_x, xy_offset_y])
            if np.linalg.norm(direction_to_sheet2) > 0:
                direction_to_sheet2 = direction_to_sheet2 / np.linalg.norm(direction_to_sheet2)
            else:
                direction_to_sheet2 = np.array([1, 0])  # Default direction
            
            # Determine if sheets face toward or away from each other
            face_toward = random.random() > 0.5  # 50% chance toward, 50% away
            
            # Base bend magnitude: up to almost 90 degrees (similar for both - continuation effect)
            base_bend_magnitude_x = random.uniform(45, 85)  # 45-85 degrees (up to almost 90)
            base_bend_magnitude_y = random.uniform(30, 70)  # 30-70 degrees
            
            # Sheet 1: bend toward or away from sheet 2
            if face_toward:
                # Face toward: bend in direction of sheet 2
                bend_dir_1_x = np.sign(direction_to_sheet2[0]) if abs(direction_to_sheet2[0]) > 0.1 else random.choice([-1, 1])
                bend_dir_1_y = np.sign(direction_to_sheet2[1]) if abs(direction_to_sheet2[1]) > 0.1 else random.choice([-1, 1])
            else:
                # Face away: bend opposite to direction of sheet 2
                bend_dir_1_x = -np.sign(direction_to_sheet2[0]) if abs(direction_to_sheet2[0]) > 0.1 else random.choice([-1, 1])
                bend_dir_1_y = -np.sign(direction_to_sheet2[1]) if abs(direction_to_sheet2[1]) > 0.1 else random.choice([-1, 1])
            
            # Sheet 2: bend toward or away from sheet 1 (opposite direction)
            if face_toward:
                # Face toward: bend toward sheet 1 (opposite direction)
                bend_dir_2_x = -bend_dir_1_x
                bend_dir_2_y = -bend_dir_1_y
            else:
                # Face away: bend away from sheet 1 (same direction as sheet 1)
                bend_dir_2_x = bend_dir_1_x
                bend_dir_2_y = bend_dir_1_y
            
            # Apply similar bends with slight variation (continuation effect)
            bent_angle1_x = base_angle1_x + bend_dir_1_x * base_bend_magnitude_x * random.uniform(0.9, 1.1)
            bent_angle1_y = base_angle1_y + bend_dir_1_y * base_bend_magnitude_y * random.uniform(0.9, 1.1)
            bent_angle2_x = base_angle2_x + bend_dir_2_x * base_bend_magnitude_x * random.uniform(0.9, 1.1)
            bent_angle2_y = base_angle2_y + bend_dir_2_y * base_bend_magnitude_y * random.uniform(0.9, 1.1)
            
            # Interpolate angles based on transition mask
            angle1_x_map = base_angle1_x + (bent_angle1_x - base_angle1_x) * (1.0 - transition_mask_1)
            angle1_y_map = base_angle1_y + (bent_angle1_y - base_angle1_y) * (1.0 - transition_mask_1)
            angle2_x_map = base_angle2_x + (bent_angle2_x - base_angle2_x) * (1.0 - transition_mask_2)
            angle2_y_map = base_angle2_y + (bent_angle2_y - base_angle2_y) * (1.0 - transition_mask_2)
            
        else:
            # Reasonably apart: apply bends (80% chance)
            use_bend = random.random() < 0.8
            
            if use_bend:
                # Choose which sheet(s) to bend (both can be bent, or just one)
                bend_sheet_1 = random.random() < 0.7  # 70% chance sheet 1 is bent
                bend_sheet_2 = random.random() < 0.7  # 70% chance sheet 2 is bent
                
                # Choose bend axis and direction
                bend_axis = random.choice(['x', 'y'])
                bend_from_left = random.random() > 0.5
                
                # Slight bend: 20-40% of sheet is bent (similar to good patches)
                transition_point = random.uniform(0.6, 0.8)  # 60-80% parallel, 20-40% bent
                
                # Create transition mask for slight bend
                if bend_axis == 'x':
                    if bend_from_left:
                        transition_mask_1 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                        transition_mask_2 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
                    else:
                        transition_mask_1 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                        transition_mask_2 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
                else:
                    if bend_from_left:
                        transition_mask_1 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                        transition_mask_2 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
                    else:
                        transition_mask_1 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                        transition_mask_2 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
                
                # Apply slight bends: add 10-30 degrees (slight, not extreme)
                if bend_sheet_1:
                    bent_angle1_x = base_angle1_x + random.choice([-1, 1]) * random.uniform(10, 30)
                    bent_angle1_y = base_angle1_y + random.choice([-1, 1]) * random.uniform(0, 15)
                    # Interpolate angles based on transition mask
                    angle1_x_map = base_angle1_x + (bent_angle1_x - base_angle1_x) * (1.0 - transition_mask_1)
                    angle1_y_map = base_angle1_y + (bent_angle1_y - base_angle1_y) * (1.0 - transition_mask_1)
                else:
                    angle1_x_map = np.full_like(X_a_norm, base_angle1_x)
                    angle1_y_map = np.full_like(Y_a_norm, base_angle1_y)
                
                if bend_sheet_2:
                    bent_angle2_x = base_angle2_x + random.choice([-1, 1]) * random.uniform(10, 30)
                    bent_angle2_y = base_angle2_y + random.choice([-1, 1]) * random.uniform(0, 15)
                    # Interpolate angles based on transition mask
                    angle2_x_map = base_angle2_x + (bent_angle2_x - base_angle2_x) * (1.0 - transition_mask_2)
                    angle2_y_map = base_angle2_y + (bent_angle2_y - base_angle2_y) * (1.0 - transition_mask_2)
                else:
                    angle2_x_map = np.full_like(X_a_norm, base_angle2_x)
                    angle2_y_map = np.full_like(Y_a_norm, base_angle2_y)
            else:
                # No bend - use uniform angles
                angle1_x_map = np.full_like(X_a_norm, base_angle1_x)
                angle1_y_map = np.full_like(Y_a_norm, base_angle1_y)
                angle2_x_map = np.full_like(X_a_norm, base_angle2_x)
                angle2_y_map = np.full_like(Y_a_norm, base_angle2_y)
        
        # Convert to radians
        angle1_x_rad = np.radians(angle1_x_map)
        angle1_y_rad = np.radians(angle1_y_map)
        angle2_x_rad = np.radians(angle2_x_map)
        angle2_y_rad = np.radians(angle2_y_map)
        
        # Generate textures independently for each sheet (in mm)
        texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
        texture_type_1 = random.choice(texture_types)
        texture_type_2 = random.choice(texture_types)
        texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
        texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
        
        # Generate curvatures independently for each sheet (in mm)
        use_too_high_curvature = random.random() < 0.3  # 30% chance for too high curvature
        curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
        curvature_type_1 = random.choice(curvature_types)
        curvature_type_2 = random.choice(curvature_types)
        curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
        curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
        
        # Apply tilt effects (scaled to mm) - varies across the sheet if bent
        tilt_amplifier = 1.0
        tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
        
        # Create base Z coordinates for patch A (relative to center, in mm)
        z_a_base = tilt_factor1 + texture1 + curvature1
        
        # Final coordinates for patch A (add center offsets)
        x_a_final = X_a + center_a_x
        y_a_final = Y_a + center_a_y
        z_a_final = z_a_base + center_a_z
        
        # Create base Z coordinates for patch B (relative to center, in mm)
        z_b_base = tilt_factor2 + texture2 + curvature2
        
        # Final coordinates for patch B (add center offsets and XY spatial separation)
        x_b_final = X_a + center_b_x
        y_b_final = Y_a + center_b_y
        z_b_final = z_b_base + center_b_z
        
        # Create patch A and B XYZ arrays
        xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        
        # For interconnected sheets, ensure edges touch but don't intersect through middle
        # For other cases, use standard intersection check
        if is_close:
            # For interconnected sheets, verify edges touch (not intersecting through middle)
            # Calculate bounding boxes
            xyz_a_min = xyz_a.min(axis=0)
            xyz_a_max = xyz_a.max(axis=0)
            xyz_b_min = xyz_b.min(axis=0)
            xyz_b_max = xyz_b.max(axis=0)
            
            # Check if sheets overlap in the middle (bad for interconnected)
            # They should only touch at edges
            overlap_x = not (xyz_a_max[0] < xyz_b_min[0] or xyz_b_max[0] < xyz_a_min[0])
            overlap_y = not (xyz_a_max[1] < xyz_b_min[1] or xyz_b_max[1] < xyz_a_min[1])
            
            # If they overlap in both X and Y (intersecting through middle), adjust slightly
            if overlap_x and overlap_y:
                # Adjust to ensure only edge contact
                if orientation == 'horizontal':
                    # Ensure only X edges touch
                    if xy_offset_x > 0:
                        center_b_x = center_a_x + patch_extent_x * 1.01  # Slight gap
                    else:
                        center_b_x = center_a_x - patch_extent_x * 1.01
                    x_b_final = X_a + center_b_x
                    y_b_final = Y_a + center_b_y
                    z_b_final = z_b_base + center_b_z
                    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
                elif orientation == 'vertical':
                    # Ensure only Y edges touch
                    if xy_offset_y > 0:
                        center_b_y = center_a_y + patch_extent_y * 1.01  # Slight gap
                    else:
                        center_b_y = center_a_y - patch_extent_y * 1.01
                    x_b_final = X_a + center_b_x
                    y_b_final = Y_a + center_b_y
                    z_b_final = z_b_base + center_b_z
                    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
                else:  # diagonal
                    # Adjust to ensure only corner edges touch
                    avg_extent = (patch_extent_x + patch_extent_y) / 2.0
                    if xy_offset_x > 0:
                        center_b_x = center_a_x + avg_extent * 1.01
                    else:
                        center_b_x = center_a_x - avg_extent * 1.01
                    if xy_offset_y > 0:
                        center_b_y = center_a_y + avg_extent * 1.01
                    else:
                        center_b_y = center_a_y - avg_extent * 1.01
                    x_b_final = X_a + center_b_x
                    y_b_final = Y_a + center_b_y
                    z_b_final = z_b_base + center_b_z
                    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        else:
            # For reasonably apart sheets, use standard intersection check
            center_b_x_current = center_b_x
            center_b_y_current = center_b_y
            
            xyz_a, xyz_b, center_b_x_new, center_b_y_new, center_b_z = ensure_sheets_no_intersection(
                xyz_a, xyz_b, X_a, Y_a, z_a_final, z_b_base,
                center_a_x, center_a_y, center_a_z,
                center_b_x_current, center_b_y_current, center_b_z,
                min_separation_mm=50.0
            )
            # Update final coordinates if offsets changed
            if center_b_x_new != center_b_x_current or center_b_y_new != center_b_y_current:
                x_b_final = X_a + center_b_x_new
                y_b_final = Y_a + center_b_y_new
                z_b_final = z_b_base + center_b_z
                xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
                # Update center_b values for consistency
                center_b_x = center_b_x_new
                center_b_y = center_b_y_new
        
        # Pivot both sheets together as a unit to add variation
        xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
        
        # Centralize the centroid of the pair to origin
        xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
        
        # Save using the same format as good patches
        prefix = f"bad_sheet{pair_idx:03d}"
        
        # Save as numpy arrays
        np.save(os.path.join(category_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
        np.save(os.path.join(category_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
        
        # Save as text files (CSV format)
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Save as PLY point cloud files
        save_ply(xyz_a, os.path.join(category_dir, f"{prefix}_sheet1.ply"))
        save_ply(xyz_b, os.path.join(category_dir, f"{prefix}_sheet2.ply"))
        
        if pair_idx % 25 == 0 or pair_idx == 1:
            xy_distance = np.sqrt(xy_offset_x**2 + xy_offset_y**2) if xy_offset_x != 0 or xy_offset_y != 0 else 0
            z_sep_actual = center_b_z - center_a_z
            bend_info = ""
            if is_close:
                # Interconnected sheets always have bends
                bend_info = f", Interconnected: Both bent, Face={'toward' if face_toward is not None and face_toward else 'away'}, Axis={bend_axis if bend_axis else 'N/A'}"
            else:
                # For apart sheets, check if bends were applied
                if use_bend:
                    bend_info = f", Bent: Sheet1={bend_sheet_1}, Sheet2={bend_sheet_2}, Axis={bend_axis if bend_axis else 'N/A'}"
            print(f"Generated pair {pair_idx}/{num_pairs}:")
            print(f"  Patch A center: [{center_a_x:.1f}, {center_a_y:.1f}, {center_a_z:.1f}] mm")
            print(f"  Patch B center: [{center_b_x:.1f}, {center_b_y:.1f}, {center_b_z:.1f}] mm")
            print(f"  XY separation: {xy_distance:.1f} mm, Z separation: {z_sep_actual:.1f} mm ({'close' if is_close else 'apart'}){bend_info}")
            print(f"  Texture A: {texture_type_1}, Texture B: {texture_type_2}")
            print(f"  Curvature A: {curvature_type_1}, Curvature B: {curvature_type_2}")
            print(f"  Saved: {category_dir}/{prefix}_sheet1_xyz.npy, {category_dir}/{prefix}_sheet2_xyz.npy")
            print()
    
    print(f"Successfully generated {num_pairs} bad same-plane pairs")
    print(f"Each sheet has {num_points} points (32x32 grid)")
    print(f"Coordinates are in millimeters (mm)")
    print(f"\nFiles saved in '{category_dir}' directory:")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.npy and bad_sheet{{i:03d}}_sheet2_xyz.npy (NumPy arrays)")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.txt and bad_sheet{{i:03d}}_sheet2_xyz.txt (CSV text files)")
    print(f"  - bad_sheet{{i:03d}}_sheet1.ply and bad_sheet{{i:03d}}_sheet2.ply (PLY point clouds)")

def create_bad_large_angle_pairs(num_pairs=100):
    """
    Category 2: Sheets with huge angles between them (angle > 25°).
    These violate the maximum angle requirement.
    
    Uses millimeter scale and saves in same format as good patches.
    """
    import os
    
    # Create output directory (same structure as good patches)
    output_dir = "bad_patches"
    category_dir = os.path.join(output_dir, "large_angle")
    os.makedirs(category_dir, exist_ok=True)
    
    # Parameters - use millimeter scale like good patches
    patch_size = 32
    num_points = patch_size * patch_size  # 1024 points
    
    # Patch A (cam1) ranges in mm (similar to good patches)
    patch_a_x_range = (-4000, 4000)  # mm
    patch_a_y_range = (-4000, 4000)   # mm
    patch_a_z_range = (30000, 65000)  # mm
    
    # Separation ranges (in mm) - normal separation for Z
    z_separation_range = (30000, 65000)  # 2-8cm in mm
    
    # Patch extents in mm - ensure no elongation
    base_extent = random.uniform(28000, 30000)  # mm
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    # Tilt amplifier for large angles (increased to make angles visible, but controlled to prevent elongation)
    large_amplifier = 3.0  # Increased to 3.0 to ensure angles are clearly visible
    
    print(f"Creating {num_pairs} BAD pairs: LARGE ANGLES (angle > 25°)")
    print(f"Patch size: {patch_size}x{patch_size} = {num_points} points")
    print(f"Coordinates in millimeters (mm)")
    print(f"Files will be saved in '{category_dir}' directory")
    print()
    
    for pair_idx in range(1, num_pairs + 1):
        # Generate patch A (cam1) center
        center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
        center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
        center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
        
        # Generate Z separation (in mm)
        z_sep = random.uniform(z_separation_range[0], z_separation_range[1])
        center_b_z = center_a_z - z_sep  # Patch B is further away (negative Z direction)
        
        # Create coordinate grids for patch A (relative to center, in mm)
        x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
        y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
        X_a, Y_a = np.meshgrid(x_a, y_a)
        
        # Normalize X, Y to [-1, 1] range for frequency-based textures/curvatures
        X_a_norm = X_a / (patch_extent_x / 2)
        Y_a_norm = Y_a / (patch_extent_y / 2)
        
        # Generate textures independently for each sheet (in mm)
        texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
        texture_type_1 = random.choice(texture_types)
        texture_type_2 = random.choice(texture_types)
        texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
        texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
        
        # Generate curvatures independently for each sheet (in mm)
        use_too_high_curvature = random.random() < 0.3  # 30% chance for too high curvature
        curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
        curvature_type_1 = random.choice(curvature_types)
        curvature_type_2 = random.choice(curvature_types)
        curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
        curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
        
        # Generate EXTREME angles: angle BETWEEN sheets must be 35 to 90 degrees
        # For bad pairs, we want large angles between sheets (35-90 degrees)
        min_angle_between_sheets = 30.0  # Minimum angle between sheets in degrees (violates good patch requirement)
        max_angle_between_sheets = 90.0  # Maximum angle between sheets in degrees (up to perpendicular)
        
        def calculate_plane_normal(angle_x_rad, angle_y_rad):
            """Calculate normal vector for a plane tilted by angle_x and angle_y."""
            normal = np.array([
                np.tan(angle_y_rad),
                np.tan(angle_x_rad),
                1.0
            ])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            return normal
        
        # Generate angles iteratively until we get a valid geometric angle
        max_attempts = 10
        for attempt in range(max_attempts):
            # Generate base angle for sheet 1 - ensure it's substantial (not too flat)
            # Minimum magnitude of 15 degrees to ensure visible tilt
            angle1_x_mag = random.uniform(15, 40)  # Magnitude of tilt around X-axis
            angle1_y_mag = random.uniform(15, 40)  # Magnitude of tilt around Y-axis
            angle1_x = angle1_x_mag * random.choice([-1, 1])  # Base tilt around X-axis for sheet 1
            angle1_y = angle1_y_mag * random.choice([-1, 1])  # Base tilt around Y-axis for sheet 1
            angle1_x_rad = np.radians(angle1_x)
            angle1_y_rad = np.radians(angle1_y)
            
            # Generate angle difference for sheet 2 (aim for 35-90 degrees geometric angle)
            # Ensure substantial difference - at least 30 degrees in at least one component
            angle_diff_x = random.uniform(30, 60) * random.choice([-1, 1])
            angle_diff_y = random.uniform(30, 60) * random.choice([-1, 1])
            
            # Calculate sheet 2 angles
            angle2_x = angle1_x + angle_diff_x
            angle2_y = angle1_y + angle_diff_y
            angle2_x_rad = np.radians(angle2_x)
            angle2_y_rad = np.radians(angle2_y)
            
            # Calculate the actual geometric angle between planes
            normal1 = calculate_plane_normal(angle1_x_rad, angle1_y_rad)
            normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
            
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            angle_between_planes_rad = np.arccos(dot_product)
            angle_between_planes_deg = np.degrees(angle_between_planes_rad)
            
            # If angle is in valid range, we're done
            if min_angle_between_sheets < angle_between_planes_deg < max_angle_between_sheets:
                break
            
            # If angle is too small, increase the differences
            if angle_between_planes_deg <= min_angle_between_sheets:
                scale_factor = (min_angle_between_sheets + 5) / angle_between_planes_deg
                angle_diff_x = angle_diff_x * scale_factor
                angle_diff_y = angle_diff_y * scale_factor
                angle2_x = angle1_x + angle_diff_x
                angle2_y = angle1_y + angle_diff_y
                angle2_x_rad = np.radians(angle2_x)
                angle2_y_rad = np.radians(angle2_y)
                
                # Recalculate
                normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
                dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
                angle_between_planes_rad = np.arccos(dot_product)
                angle_between_planes_deg = np.degrees(angle_between_planes_rad)
            
            # If angle is too large (close to 90°), reduce the differences
            if angle_between_planes_deg >= max_angle_between_sheets:
                scale_factor = (max_angle_between_sheets - 5) / angle_between_planes_deg
                angle_diff_x = angle_diff_x * scale_factor
                angle_diff_y = angle_diff_y * scale_factor
                angle2_x = angle1_x + angle_diff_x
                angle2_y = angle1_y + angle_diff_y
                angle2_x_rad = np.radians(angle2_x)
                angle2_y_rad = np.radians(angle2_y)
                
                # Recalculate
                normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
                dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
                angle_between_planes_rad = np.arccos(dot_product)
                angle_between_planes_deg = np.degrees(angle_between_planes_rad)
        
        # Final verification
        angle_diff_x_deg = abs(angle2_x - angle1_x)
        angle_diff_y_deg = abs(angle2_y - angle1_y)
        
        # Main augmentation: Apply bends to sheets (80% chance - main augmentation)
        use_bend = random.random() < 0.8
        
        if use_bend:
            # Choose which sheet(s) to bend (both can be bent, or just one)
            bend_sheet_1 = random.random() < 0.8  # 80% chance sheet 1 is bent
            bend_sheet_2 = random.random() < 0.8  # 80% chance sheet 2 is bent
            
            # Choose bend axis and direction
            bend_axis = random.choice(['x', 'y'])
            bend_from_left = random.random() > 0.5
            
            # Bend: 30-50% of sheet is bent (more than slight bends)
            transition_point = random.uniform(0.5, 0.7)  # 50-70% parallel, 30-50% bent
            
            # Create transition mask for bend
            if bend_axis == 'x':
                if bend_from_left:
                    transition_mask_1 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                    transition_mask_2 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
                else:
                    transition_mask_1 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                    transition_mask_2 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
            else:
                if bend_from_left:
                    transition_mask_1 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                    transition_mask_2 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
                else:
                    transition_mask_1 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                    transition_mask_2 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
            
            # Apply bends: add 15-40 degrees (moderate to strong bends)
            if bend_sheet_1:
                bent_angle1_x = angle1_x + random.choice([-1, 1]) * random.uniform(15, 40)
                bent_angle1_y = angle1_y + random.choice([-1, 1]) * random.uniform(10, 30)
                # Interpolate angles based on transition mask
                angle1_x_map = angle1_x + (bent_angle1_x - angle1_x) * (1.0 - transition_mask_1)
                angle1_y_map = angle1_y + (bent_angle1_y - angle1_y) * (1.0 - transition_mask_1)
            else:
                angle1_x_map = np.full_like(X_a_norm, angle1_x)
                angle1_y_map = np.full_like(Y_a_norm, angle1_y)
            
            if bend_sheet_2:
                bent_angle2_x = angle2_x + random.choice([-1, 1]) * random.uniform(15, 40)
                bent_angle2_y = angle2_y + random.choice([-1, 1]) * random.uniform(10, 30)
                # Interpolate angles based on transition mask
                angle2_x_map = angle2_x + (bent_angle2_x - angle2_x) * (1.0 - transition_mask_2)
                angle2_y_map = angle2_y + (bent_angle2_y - angle2_y) * (1.0 - transition_mask_2)
            else:
                angle2_x_map = np.full_like(X_a_norm, angle2_x)
                angle2_y_map = np.full_like(Y_a_norm, angle2_y)
            
            # Convert to radians for tilt calculation
            angle1_x_rad_map = np.radians(angle1_x_map)
            angle1_y_rad_map = np.radians(angle1_y_map)
            angle2_x_rad_map = np.radians(angle2_x_map)
            angle2_y_rad_map = np.radians(angle2_y_map)
            
            # Apply tilt effects with varying angles (bent sheets)
            tilt_factor1 = (np.tan(angle1_x_rad_map) * Y_a + np.tan(angle1_y_rad_map) * X_a) * large_amplifier
            tilt_factor2 = (np.tan(angle2_x_rad_map) * Y_a + np.tan(angle2_y_rad_map) * X_a) * large_amplifier
        else:
            # No bend - use uniform angles
            # Apply tilt effects (scaled to mm, using large amplifier for extreme angles)
            tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * large_amplifier
            tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * large_amplifier
        
        # Create base Z coordinates for patch A (relative to center, in mm)
        z_a_base = tilt_factor1 + texture1 + curvature1
        
        # Clamp Z values to prevent extreme elongation (limit to reasonable range)
        # Increased to 0.4 to allow more angle visibility while still preventing elongation
        max_z_variation = patch_extent_x * 0.4  # Max 40% of patch extent in Z
        z_a_base = np.clip(z_a_base, -max_z_variation, max_z_variation)
        
        # Final coordinates for patch A (add center offsets)
        x_a_final = X_a + center_a_x
        y_a_final = Y_a + center_a_y
        z_a_final = z_a_base + center_a_z
        
        # Create base Z coordinates for patch B (relative to center, in mm)
        z_b_base = tilt_factor2 + texture2 + curvature2
        
        # Clamp Z values to prevent extreme elongation
        z_b_base = np.clip(z_b_base, -max_z_variation, max_z_variation)
        
        # Small XY offset to keep sheets close together (not elongated)
        # Use small offset regardless of angle to ensure points stay close
        offset_factor = random.uniform(0.05, 0.15)  # Small offset to keep sheets close
        min_xy_offset = offset_factor * max(patch_extent_x, patch_extent_y)
        offset_direction = random.uniform(0, 2 * np.pi)
        xy_offset_x = np.cos(offset_direction) * min_xy_offset
        xy_offset_y = np.sin(offset_direction) * min_xy_offset
        
        center_b_x = center_a_x + xy_offset_x
        center_b_y = center_a_y + xy_offset_y
        
        # Final coordinates for patch B (add center offsets - keep sheets close)
        x_b_final = X_a + center_b_x
        y_b_final = Y_a + center_b_y
        z_b_final = z_b_base + center_b_z
        
        # Create patch A and B XYZ arrays
        xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        
        # Ensure points stay close together regardless of angle
        # Calculate distances between corresponding points
        point_distances = np.linalg.norm(xyz_a - xyz_b, axis=1)
        max_distance = np.max(point_distances)
        mean_distance = np.mean(point_distances)
        
        # If points are too far apart, adjust to keep them closer
        # Target: mean distance should be around 1.5-2x the Z separation
        target_mean_distance = z_sep * 1.8
        if mean_distance > target_mean_distance * 1.5:
            # Points are too spread out - reduce the XY offset
            scale_factor = (target_mean_distance * 1.2) / mean_distance
            xy_offset_x = xy_offset_x * scale_factor
            xy_offset_y = xy_offset_y * scale_factor
            center_b_x = center_a_x + xy_offset_x
            center_b_y = center_a_y + xy_offset_y
            
            # Recalculate coordinates
            x_b_final = X_a + center_b_x
            y_b_final = Y_a + center_b_y
            z_b_final = z_b_base + center_b_z
            xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        
        # Ensure sheets do not intersect (or only edges touch)
        # Check for intersection and adjust if needed
        xyz_a_min = xyz_a.min(axis=0)
        xyz_a_max = xyz_a.max(axis=0)
        xyz_b_min = xyz_b.min(axis=0)
        xyz_b_max = xyz_b.max(axis=0)
        
        # Check if sheets overlap in the middle (bad - they should not intersect)
        overlap_x = not (xyz_a_max[0] < xyz_b_min[0] or xyz_b_max[0] < xyz_a_min[0])
        overlap_y = not (xyz_a_max[1] < xyz_b_min[1] or xyz_b_max[1] < xyz_a_min[1])
        
        # If they overlap in both X and Y (intersecting through middle), adjust
        if overlap_x and overlap_y:
            # Calculate separation needed to avoid intersection
            # Use bounding box centers to determine direction
            center_a_bb = (xyz_a_min + xyz_a_max) / 2.0
            center_b_bb = (xyz_b_min + xyz_b_max) / 2.0
            direction = center_b_bb[:2] - center_a_bb[:2]
            if np.linalg.norm(direction) < 1e-6:
                # If centers are too close, use random direction
                direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            direction = direction / np.linalg.norm(direction)
            
            # Calculate minimum separation needed
            extent_a = np.max(xyz_a_max[:2] - xyz_a_min[:2])
            extent_b = np.max(xyz_b_max[:2] - xyz_b_min[:2])
            min_sep_needed = (extent_a + extent_b) / 2.0 + 100.0  # Add small buffer
            
            # Adjust XY offset to ensure no intersection
            xy_offset_x = direction[0] * min_sep_needed
            xy_offset_y = direction[1] * min_sep_needed
            center_b_x = center_a_x + xy_offset_x
            center_b_y = center_a_y + xy_offset_y
            
            # Recalculate coordinates
            x_b_final = X_a + center_b_x
            y_b_final = Y_a + center_b_y
            z_b_final = z_b_base + center_b_z
            xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        else:
            # Use standard intersection check to ensure minimum separation
            center_b_x_current = center_b_x
            center_b_y_current = center_b_y
            
            xyz_a, xyz_b, center_b_x_new, center_b_y_new, center_b_z = ensure_sheets_no_intersection(
                xyz_a, xyz_b, X_a, Y_a, z_a_final, z_b_base,
                center_a_x, center_a_y, center_a_z,
                center_b_x_current, center_b_y_current, center_b_z,
                min_separation_mm=50.0
            )
            # Update final coordinates if offsets changed
            if center_b_x_new != center_b_x_current or center_b_y_new != center_b_y_current:
                x_b_final = X_a + center_b_x_new
                y_b_final = Y_a + center_b_y_new
                z_b_final = z_b_base + center_b_z
                xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
                # Update center_b values for consistency
                center_b_x = center_b_x_new
                center_b_y = center_b_y_new
        
        # Pivot both sheets together as a unit to add variation
        xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
        
        # Centralize the centroid of the pair to origin
        xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
        
        # Save using the same format as good patches
        prefix = f"bad_sheet{pair_idx:03d}"
        
        # Save as numpy arrays
        np.save(os.path.join(category_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
        np.save(os.path.join(category_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
        
        # Save as text files (CSV format)
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Save as PLY point cloud files
        save_ply(xyz_a, os.path.join(category_dir, f"{prefix}_sheet1.ply"))
        save_ply(xyz_b, os.path.join(category_dir, f"{prefix}_sheet2.ply"))
        
        if pair_idx % 25 == 0 or pair_idx == 1:
            # Calculate final geometric angle
            normal1 = calculate_plane_normal(angle1_x_rad, angle1_y_rad)
            normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            angle_between_planes_deg = np.degrees(np.arccos(dot_product))
            xy_distance = np.sqrt(xy_offset_x**2 + xy_offset_y**2)
            # Recalculate mean distance for display
            final_distances = np.linalg.norm(xyz_a - xyz_b, axis=1)
            final_mean_distance = np.mean(final_distances)
            print(f"Generated pair {pair_idx}/{num_pairs}:")
            print(f"  Patch A center: [{center_a_x:.1f}, {center_a_y:.1f}, {center_a_z:.1f}] mm")
            print(f"  Patch B center: [{center_b_x:.1f}, {center_b_y:.1f}, {center_b_z:.1f}] mm")
            print(f"  Z separation: {z_sep:.1f} mm, XY offset: {xy_distance:.1f} mm")
            print(f"  Angles: Sheet1=({angle1_x:.1f}°,{angle1_y:.1f}°), Sheet2=({angle2_x:.1f}°,{angle2_y:.1f}°)")
            print(f"  Geometric Angle={angle_between_planes_deg:.2f}° (target > 25°)")
            print(f"  Mean point distance: {final_mean_distance:.1f} mm (target: ~{target_mean_distance:.1f} mm)")
            print(f"  Texture A: {texture_type_1}, Texture B: {texture_type_2}")
            print(f"  Curvature A: {curvature_type_1}, Curvature B: {curvature_type_2}")
            print()
    
    print(f"Successfully generated {num_pairs} bad large-angle pairs")
    print(f"Each sheet has {num_points} points (32x32 grid)")
    print(f"Coordinates are in millimeters (mm)")
    print(f"\nFiles saved in '{category_dir}' directory:")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.npy and bad_sheet{{i:03d}}_sheet2_xyz.npy (NumPy arrays)")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.txt and bad_sheet{{i:03d}}_sheet2_xyz.txt (CSV text files)")
    print(f"  - bad_sheet{{i:03d}}_sheet1.ply and bad_sheet{{i:03d}}_sheet2.ply (PLY point clouds)")

def create_bad_extreme_texture_curve_pairs(num_pairs=100):
    """
    Category 3: Sheets with EXTREME texture (variation > 80mm = 8cm).
    These violate the maximum texture variation requirement.
    Note: Slight spherical curvature is NOT considered bad - only extreme texture variations.
    
    Uses millimeter scale and saves in same format as good patches.
    """
    import os
    
    # Create output directory (same structure as good patches)
    output_dir = "bad_patches"
    category_dir = os.path.join(output_dir, "extreme_texture")
    os.makedirs(category_dir, exist_ok=True)
    
    # Parameters - use millimeter scale like good patches
    patch_size = 32
    num_points = patch_size * patch_size  # 1024 points
    
    # Patch A (cam1) ranges in mm (similar to good patches)
    patch_a_x_range = (-4000, 4000)  # mm
    patch_a_y_range = (-4000, 4000)   # mm
    patch_a_z_range = (30000, 65000)  # mm
    
    # Separation ranges (in mm) - normal separation for Z
    z_separation_range = (30000, 65000)  # 2-8cm in mm
    
    # Patch extents in mm - ensure no elongation
    base_extent = random.uniform(28000, 30000)  # mm
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    print(f"Creating {num_pairs} BAD pairs: EXTREME TEXTURE (variation > 80mm = 8cm, no curvature)")
    print(f"Patch size: {patch_size}x{patch_size} = {num_points} points")
    print(f"Coordinates in millimeters (mm)")
    print(f"Files will be saved in '{category_dir}' directory")
    print()
    
    for pair_idx in range(1, num_pairs + 1):
        # Generate patch A (cam1) center
        center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
        center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
        center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
        
        # Generate Z separation (in mm)
        z_sep = random.uniform(z_separation_range[0], z_separation_range[1])
        center_b_z = center_a_z - z_sep  # Patch B is further away
        
        # Create coordinate grids for patch A (relative to center, in mm)
        x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
        y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
        X_a, Y_a = np.meshgrid(x_a, y_a)
        
        # Normalize X, Y to [-1, 1] range for frequency-based textures
        X_a_norm = X_a / (patch_extent_x / 2)
        Y_a_norm = Y_a / (patch_extent_y / 2)
        
        # ONLY extreme texture variations - NO curvature (spherical curvature is acceptable)
        # Use LARGE texture variations (40-80mm = 4-8cm instead of normal 2-5mm)
        # Still bad (exceeds 80mm threshold) but maintains sheet-like appearance
        texture_types = ["random", "wave", "noise", "grid", "spots"]
        texture_type_1 = random.choice(texture_types)
        texture_type_2 = random.choice(texture_types)
        texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=True)
        texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=True)
        
        # No curvature for extreme texture pairs
        curvature1 = np.zeros((patch_size, patch_size))
        curvature2 = np.zeros((patch_size, patch_size))
        
        # Create base Z coordinates for patch A (relative to center, in mm)
        z_a_base = texture1 + curvature1
        
        # Final coordinates for patch A (add center offsets)
        x_a_final = X_a + center_a_x
        y_a_final = Y_a + center_a_y
        z_a_final = z_a_base + center_a_z
        
        # Create base Z coordinates for patch B (relative to center, in mm)
        z_b_base = texture2 + curvature2
        
        # Final coordinates for patch B (add center offsets)
        x_b_final = X_a + center_a_x
        y_b_final = Y_a + center_a_y
        z_b_final = z_b_base + center_b_z
        
        # Create patch A and B XYZ arrays
        xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        
        # Ensure sheets do not intersect
        # Use center_b values directly (they are always defined)
        center_b_x_current = center_b_x
        center_b_y_current = center_b_y
        
        xyz_a, xyz_b, center_b_x_new, center_b_y_new, center_b_z = ensure_sheets_no_intersection(
            xyz_a, xyz_b, X_a, Y_a, z_a_final, z_b_base,
            center_a_x, center_a_y, center_a_z,
            center_b_x_current, center_b_y_current, center_b_z,
            min_separation_mm=50.0
        )
        # Update final coordinates if offsets changed
        if center_b_x_new != center_b_x_current or center_b_y_new != center_b_y_current:
            x_b_final = X_a + center_b_x_new
            y_b_final = Y_a + center_b_y_new
            z_b_final = z_b_base + center_b_z
            xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
            # Update center_b values for consistency
            center_b_x = center_b_x_new
            center_b_y = center_b_y_new
        
        # Pivot both sheets together as a unit to add variation
        xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
        
        # Centralize the centroid of the pair to origin
        xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
        
        # Save using the same format as good patches
        prefix = f"bad_sheet{pair_idx:03d}"
        
        # Save as numpy arrays
        np.save(os.path.join(category_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
        np.save(os.path.join(category_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
        
        # Save as text files (CSV format)
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Save as PLY point cloud files
        save_ply(xyz_a, os.path.join(category_dir, f"{prefix}_sheet1.ply"))
        save_ply(xyz_b, os.path.join(category_dir, f"{prefix}_sheet2.ply"))
        
        if pair_idx % 25 == 0 or pair_idx == 1:
            var1 = (z_a_base.max() - z_a_base.min())
            var2 = (z_b_base.max() - z_b_base.min())
            print(f"Generated pair {pair_idx}/{num_pairs}:")
            print(f"  Patch A center: [{center_a_x:.1f}, {center_a_y:.1f}, {center_a_z:.1f}] mm")
            print(f"  Patch B center: [{center_a_x:.1f}, {center_a_y:.1f}, {center_b_z:.1f}] mm")
            print(f"  Z separation: {z_sep:.1f} mm")
            print(f"  Texture variations: {var1:.1f}mm / {var2:.1f}mm (target > 80mm)")
            print(f"  Texture A: {texture_type_1}, Texture B: {texture_type_2}")
            print()
    
    print(f"Successfully generated {num_pairs} bad extreme texture pairs")
    print(f"Each sheet has {num_points} points (32x32 grid)")
    print(f"Coordinates are in millimeters (mm)")
    print(f"\nFiles saved in '{category_dir}' directory:")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.npy and bad_sheet{{i:03d}}_sheet2_xyz.npy (NumPy arrays)")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.txt and bad_sheet{{i:03d}}_sheet2_xyz.txt (CSV text files)")
    print(f"  - bad_sheet{{i:03d}}_sheet1.ply and bad_sheet{{i:03d}}_sheet2.ply (PLY point clouds)")

def create_bad_too_much_xy_separation_pairs(num_pairs=100):
    """
    Category: Sheets with excessive XY spatial separation.
    These violate the maximum XY spatial separation requirement.
    Sheets are separated too far in X and/or Y directions (80mm-220mm = 8cm-22cm instead of normal 38mm-77mm).
    
    Uses millimeter scale and saves in same format as good patches.
    """
    import os
    
    # Create output directory (same structure as good patches)
    output_dir = "bad_patches"
    category_dir = os.path.join(output_dir, "too_much_xy_separation")
    os.makedirs(category_dir, exist_ok=True)
    
    # Parameters - use millimeter scale like good patches
    patch_size = 32
    num_points = patch_size * patch_size  # 1024 points
    
    # Patch A (cam1) ranges in mm (similar to good patches)
    patch_a_x_range = (-4000, 4000)  # mm
    patch_a_y_range = (-4000, 4000)   # mm
    patch_a_z_range = (30000, 65000)  # mm
    
    # Separation ranges (in mm) - normal separation for Z
    z_separation_range = (30000, 65000)  # 2-8cm in mm
    
    # Excessive XY separation range (in mm) - 80mm to 220mm (8cm to 22cm)
    excessive_xy_separation_range = (5000, 150000)  # mm
    
    # Patch extents in mm - ensure no elongation
    base_extent = random.uniform(28000, 30000)  # mm
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    print(f"Creating {num_pairs} BAD pairs: TOO MUCH XY SEPARATION (excessive spatial separation)")
    print(f"Patch size: {patch_size}x{patch_size} = {num_points} points")
    print(f"Coordinates in millimeters (mm)")
    print(f"Files will be saved in '{category_dir}' directory")
    print()
    
    for pair_idx in range(1, num_pairs + 1):
        # Generate patch A (cam1) center
        center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
        center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
        center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
        
        # Generate Z separation (in mm) - normal range
        z_sep = random.uniform(z_separation_range[0], z_separation_range[1])
        center_b_z = center_a_z - z_sep  # Patch B is further away
        
        # EXCESSIVE XY spatial offset (80mm to 220mm = 8cm to 22cm)
        excessive_xy_sep = random.uniform(excessive_xy_separation_range[0], excessive_xy_separation_range[1])
        offset_direction = random.uniform(0, 2 * np.pi)
        xy_offset_x = np.cos(offset_direction) * excessive_xy_sep
        xy_offset_y = np.sin(offset_direction) * excessive_xy_sep
        
        center_b_x = center_a_x + xy_offset_x
        center_b_y = center_a_y + xy_offset_y
        
        # Create coordinate grids for patch A (relative to center, in mm)
        x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
        y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
        X_a, Y_a = np.meshgrid(x_a, y_a)
        
        # Normalize X, Y to [-1, 1] range for frequency-based textures/curvatures
        X_a_norm = X_a / (patch_extent_x / 2)
        Y_a_norm = Y_a / (patch_extent_y / 2)
        
        # Generate textures independently for each sheet (in mm)
        texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
        texture_type_1 = random.choice(texture_types)
        texture_type_2 = random.choice(texture_types)
        texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
        texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
        
        # Generate curvatures independently for each sheet (in mm)
        use_too_high_curvature = random.random() < 0.3  # 30% chance for too high curvature
        curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
        curvature_type_1 = random.choice(curvature_types)
        curvature_type_2 = random.choice(curvature_types)
        curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
        curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
        
        # Generate small tilt angles (within good patch range)
        angle1_x = random.uniform(-5, 5)
        angle1_y = random.uniform(-5, 5)
        angle2_x = random.uniform(-5, 5)
        angle2_y = random.uniform(-5, 5)
        
        angle1_x_rad = np.radians(angle1_x)
        angle1_y_rad = np.radians(angle1_y)
        angle2_x_rad = np.radians(angle2_x)
        angle2_y_rad = np.radians(angle2_y)
        
        # Apply tilt effects (scaled to mm) - varies across the sheet if bent
        tilt_amplifier = 1.0
        tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
        
        # Create base Z coordinates for patch A (relative to center, in mm)
        z_a_base = tilt_factor1 + texture1 + curvature1
        
        # Final coordinates for patch A (add center offsets)
        x_a_final = X_a + center_a_x
        y_a_final = Y_a + center_a_y
        z_a_final = z_a_base + center_a_z
        
        # Create base Z coordinates for patch B (relative to center, in mm)
        z_b_base = tilt_factor2 + texture2 + curvature2
        
        # Final coordinates for patch B (add center offsets and excessive XY separation)
        x_b_final = X_a + center_b_x
        y_b_final = Y_a + center_b_y
        z_b_final = z_b_base + center_b_z
        
        # Create patch A and B XYZ arrays
        xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        
        # Ensure sheets do not intersect
        # Use center_b values directly (they are always defined)
        center_b_x_current = center_b_x
        center_b_y_current = center_b_y
        
        xyz_a, xyz_b, center_b_x_new, center_b_y_new, center_b_z = ensure_sheets_no_intersection(
            xyz_a, xyz_b, X_a, Y_a, z_a_final, z_b_base,
            center_a_x, center_a_y, center_a_z,
            center_b_x_current, center_b_y_current, center_b_z,
            min_separation_mm=50.0
        )
        # Update final coordinates if offsets changed
        if center_b_x_new != center_b_x_current or center_b_y_new != center_b_y_current:
            x_b_final = X_a + center_b_x_new
            y_b_final = Y_a + center_b_y_new
            z_b_final = z_b_base + center_b_z
            xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
            # Update center_b values for consistency
            center_b_x = center_b_x_new
            center_b_y = center_b_y_new
        
        # Pivot both sheets together as a unit to add variation
        xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
        
        # Centralize the centroid of the pair to origin
        xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
        
        # Save using the same format as good patches
        prefix = f"bad_sheet{pair_idx:03d}"
        
        # Save as numpy arrays
        np.save(os.path.join(category_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
        np.save(os.path.join(category_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
        
        # Save as text files (CSV format)
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Save as PLY point cloud files
        save_ply(xyz_a, os.path.join(category_dir, f"{prefix}_sheet1.ply"))
        save_ply(xyz_b, os.path.join(category_dir, f"{prefix}_sheet2.ply"))
        
        if pair_idx % 25 == 0 or pair_idx == 1:
            xy_distance = np.sqrt(xy_offset_x**2 + xy_offset_y**2)
            print(f"Generated pair {pair_idx}/{num_pairs}:")
            print(f"  Patch A center: [{center_a_x:.1f}, {center_a_y:.1f}, {center_a_z:.1f}] mm")
            print(f"  Patch B center: [{center_b_x:.1f}, {center_b_y:.1f}, {center_b_z:.1f}] mm")
            print(f"  Z separation: {z_sep:.1f} mm, XY separation: {xy_distance:.1f} mm (excessive, violates requirement)")
            print(f"  Texture A: {texture_type_1}, Texture B: {texture_type_2}")
            print(f"  Curvature A: {curvature_type_1}, Curvature B: {curvature_type_2}")
            print()
    
    print(f"Successfully generated {num_pairs} bad too-much-XY-separation pairs")
    print(f"Each sheet has {num_points} points (32x32 grid)")
    print(f"Coordinates are in millimeters (mm)")
    print(f"\nFiles saved in '{category_dir}' directory:")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.npy and bad_sheet{{i:03d}}_sheet2_xyz.npy (NumPy arrays)")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.txt and bad_sheet{{i:03d}}_sheet2_xyz.txt (CSV text files)")
    print(f"  - bad_sheet{{i:03d}}_sheet1.ply and bad_sheet{{i:03d}}_sheet2.ply (PLY point clouds)")

def create_bad_multi_violation_pairs(num_pairs=100):
    """
    Category 5: Sheets that violate multiple criteria at once.
    Combination of excessive XY separation AND bent sheets (70%+ bent).
    
    Uses millimeter scale and saves in same format as good patches.
    """
    import os
    
    # Create output directory (same structure as good patches)
    output_dir = "bad_patches"
    category_dir = os.path.join(output_dir, "multi_violation")
    os.makedirs(category_dir, exist_ok=True)
    
    # Parameters - use millimeter scale like good patches
    patch_size = 32
    num_points = patch_size * patch_size  # 1024 points
    
    # Patch A (cam1) ranges in mm (similar to good patches)
    patch_a_x_range = (-4000, 4000)  # mm
    patch_a_y_range = (-4000, 4000)   # mm
    patch_a_z_range = (30000, 65000)  # mm
    
    # Normal separation range (in mm)
    normal_z_separation_range = (30000, 65000)  # 2-4cm in mm
    
    # Excessive XY separation range (in mm)
    excessive_xy_separation_range = (5000, 150000)  # 60mm-180mm
    
    # Patch extents in mm - ensure no elongation
    base_extent = random.uniform(28000, 30000)  # mm
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    print(f"Creating {num_pairs} BAD pairs: MULTI-VIOLATION (excessive XY separation + bent sheets)")
    print(f"Patch size: {patch_size}x{patch_size} = {num_points} points")
    print(f"Coordinates in millimeters (mm)")
    print(f"Files will be saved in '{category_dir}' directory")
    print()
    
    for pair_idx in range(1, num_pairs + 1):
        # Choose violation combination (only excessive XY separation now)
        violation_type = "too_much_xy_sep"
        
        # Generate patch A (cam1) center
        center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
        center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
        center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
        
        # Normal Z separation
        z_sep = random.uniform(normal_z_separation_range[0], normal_z_separation_range[1])  # in mm
        center_b_z = center_a_z - z_sep  # Patch B is further away
        
        # Handle XY separation violations
        if "too_much_xy_sep" in violation_type:
            # EXCESSIVE XY spatial offset (80,000-220,000mm = 80-220m)
            excessive_xy_sep = random.uniform(excessive_xy_separation_range[0], excessive_xy_separation_range[1])
            offset_direction = random.uniform(0, 2 * np.pi)
            xy_offset_x = np.cos(offset_direction) * excessive_xy_sep
            xy_offset_y = np.sin(offset_direction) * excessive_xy_sep
        else:
            # Normal XY separation (small or no offset)
            xy_offset_x = 0
            xy_offset_y = 0
        
        center_b_x = center_a_x + xy_offset_x
        center_b_y = center_a_y + xy_offset_y
        
        # Create coordinate grids for patch A (relative to center, in mm)
        x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
        y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
        X_a, Y_a = np.meshgrid(x_a, y_a)
        
        # Normalize X, Y to [-1, 1] range for frequency-based textures/curvatures
        X_a_norm = X_a / (patch_extent_x / 2)
        Y_a_norm = Y_a / (patch_extent_y / 2)
        
        # Generate textures independently for each sheet (in mm)
        texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
        texture_type_1 = random.choice(texture_types)
        texture_type_2 = random.choice(texture_types)
        texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
        texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
        
        # Generate curvatures independently for each sheet (in mm)
        use_too_high_curvature = random.random() < 0.3  # 30% chance for too high curvature
        curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
        curvature_type_1 = random.choice(curvature_types)
        curvature_type_2 = random.choice(curvature_types)
        curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
        curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
        
        # Multi-violation: Combine excessive XY separation with bent sheets
        # Choose which sheet will be bent (sheet 1 or sheet 2)
        bent_sheet = random.choice([1, 2])
        
        # Generate base angles for the flat sheet (small angles, relatively parallel)
        if bent_sheet == 1:
            # Sheet 2 is flat, sheet 1 will be bent
            base_angle2_x = random.uniform(-5, 5)
            base_angle2_y = random.uniform(-5, 5)
            base_angle1_x = random.uniform(-5, 5)  # Starting angle for sheet 1
            base_angle1_y = random.uniform(-5, 5)
        else:
            # Sheet 1 is flat, sheet 2 will be bent
            base_angle1_x = random.uniform(-5, 5)
            base_angle1_y = random.uniform(-5, 5)
            base_angle2_x = random.uniform(-5, 5)  # Starting angle for sheet 2
            base_angle2_y = random.uniform(-5, 5)
        
        # Calculate the normal vector of the flat sheet to determine perpendicular direction
        def calculate_plane_normal(angle_x_rad, angle_y_rad):
            """Calculate normal vector for a plane tilted by angle_x and angle_y."""
            normal = np.array([
                np.tan(angle_y_rad),
                np.tan(angle_x_rad),
                1.0
            ])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            return normal
        
        if bent_sheet == 1:
            # Sheet 2 is flat - calculate its normal
            angle2_x_rad_flat = np.radians(base_angle2_x)
            angle2_y_rad_flat = np.radians(base_angle2_y)
            flat_normal = calculate_plane_normal(angle2_x_rad_flat, angle2_y_rad_flat)
        else:
            # Sheet 1 is flat - calculate its normal
            angle1_x_rad_flat = np.radians(base_angle1_x)
            angle1_y_rad_flat = np.radians(base_angle1_y)
            flat_normal = calculate_plane_normal(angle1_x_rad_flat, angle1_y_rad_flat)
        
        # Choose bend axis and direction (towards or away from the other sheet)
        bend_axis = random.choice(['x', 'y'])
        bend_direction = random.choice(['towards', 'away'])  # Towards or away from the other sheet
        # Use smaller transition_point to ensure 70%+ bent (0.15 means 85% bent, 0.25 means 75% bent)
        transition_point = random.uniform(0.15, 0.25)  # 15-25% parallel, rest bent (ensures 70%+ bent)
        
        # Determine bend direction along axis
        bend_from_left = random.random() > 0.5
        
        if bend_axis == 'x':
            # Bend along X axis
            if bend_from_left:
                # Bend from left (small X) to right (large X)
                transition_mask = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
            else:
                # Bend from right (large X) to left (small X)
                transition_mask = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1)
        else:
            # Bend along Y axis
            if bend_from_left:
                # Bend from bottom (small Y) to top (large Y)
                transition_mask = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
            else:
                # Bend from top (large Y) to bottom (small Y)
                transition_mask = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1)
        
        # Ensure at least 70% is bent
        bent_fraction = np.mean(transition_mask < 1.0)
        if bent_fraction < 0.7:
            transition_point = 0.15  # Ensure 85% bent
            if bend_axis == 'x':
                if bend_from_left:
                    transition_mask = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
                else:
                    transition_mask = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1)
            else:
                if bend_from_left:
                    transition_mask = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
                else:
                    transition_mask = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1)
        
        # Find the bend point location (where transition_mask transitions from 1 to <1)
        # Center the patch at the bend point OR on the bent part where it's perpendicular/extreme angle
        center_on_bend_line = random.random() < 0.5  # 50% chance to center on bend line, 50% on bent part
        
        if center_on_bend_line:
            # Center at the bend line (transition point)
            if bend_axis == 'x':
                # Find X coordinate where bend occurs
                if bend_from_left:
                    bend_x_norm = -1.0 + 2.0 * transition_point  # X coordinate of bend
                else:
                    bend_x_norm = 1.0 - 2.0 * transition_point
                bend_x = bend_x_norm * (patch_extent_x / 2)
                # Adjust center to be at bend point
                center_a_x = center_a_x - bend_x
                center_b_x = center_b_x - bend_x
            else:
                # Find Y coordinate where bend occurs
                if bend_from_left:
                    bend_y_norm = -1.0 + 2.0 * transition_point
                else:
                    bend_y_norm = 1.0 - 2.0 * transition_point
                bend_y = bend_y_norm * (patch_extent_y / 2)
                # Adjust center to be at bend point
                center_a_y = center_a_y - bend_y
                center_b_y = center_b_y - bend_y
        else:
            # Center on the bent part where points are perpendicular/extreme angle
            # Find a location in the bent region (where transition_mask < 1.0)
            # Choose a point in the middle-to-end of the bent portion
            bent_region_fraction = random.uniform(0.3, 0.8)  # 30-80% into the bent region
            
            if bend_axis == 'x':
                if bend_from_left:
                    # Bent region is from transition_point to 1.0
                    bent_x_norm = -1.0 + 2.0 * (transition_point + (1.0 - transition_point) * bent_region_fraction)
                else:
                    # Bent region is from -1.0 to (1.0 - 2*transition_point)
                    bent_x_norm = -1.0 + (1.0 - 2.0 * transition_point) * (1.0 - bent_region_fraction)
                bent_x = bent_x_norm * (patch_extent_x / 2)
                center_a_x = center_a_x - bent_x
                center_b_x = center_b_x - bent_x
            else:
                if bend_from_left:
                    bent_y_norm = -1.0 + 2.0 * (transition_point + (1.0 - transition_point) * bent_region_fraction)
                else:
                    bent_y_norm = -1.0 + (1.0 - 2.0 * transition_point) * (1.0 - bent_region_fraction)
                bent_y = bent_y_norm * (patch_extent_y / 2)
                center_a_y = center_a_y - bent_y
                center_b_y = center_b_y - bent_y
        
        # Generate bent angles that are perpendicular to the flat sheet
        # Find angles that create a normal vector perpendicular to flat_normal
        # Try different angle combinations to get ~90 degrees from flat sheet
        if bent_sheet == 1:
            # Sheet 1 will be bent perpendicular to sheet 2
            # Try to find angles that make sheet 1 perpendicular to sheet 2
            for attempt in range(20):
                # Generate candidate bent angles
                bent_angle1_x = base_angle1_x + random.choice([-1, 1]) * random.uniform(70, 90)
                bent_angle1_y = base_angle1_y + random.choice([-1, 1]) * random.uniform(0, 20)
                
                bent_angle1_x_rad = np.radians(bent_angle1_x)
                bent_angle1_y_rad = np.radians(bent_angle1_y)
                bent_normal = calculate_plane_normal(bent_angle1_x_rad, bent_angle1_y_rad)
                
                # Check if bent normal is approximately perpendicular to flat normal
                dot_product = np.clip(np.dot(bent_normal, flat_normal), -1.0, 1.0)
                angle_between = np.degrees(np.arccos(abs(dot_product)))
                
                # We want angle between normals to be close to 90 degrees (perpendicular)
                if 75 <= angle_between <= 105:
                    break
            
            # Adjust bent angle direction based on whether it should go towards or away
            # If towards, the bent portion should point in the direction of the other sheet
            # If away, it should point away from the other sheet
            # This is controlled by the sign of the angle change
            if bend_direction == 'away':
                # Reverse the direction to point away
                bent_angle1_x = base_angle1_x - (bent_angle1_x - base_angle1_x)
            
            # Interpolate angles for sheet 1 based on transition mask
            angle1_x_map = base_angle1_x + (bent_angle1_x - base_angle1_x) * (1.0 - transition_mask)
            angle1_y_map = base_angle1_y + (bent_angle1_y - base_angle1_y) * (1.0 - transition_mask)
            # Sheet 2 stays flat
            angle2_x_map = np.full_like(X_a_norm, base_angle2_x)
            angle2_y_map = np.full_like(Y_a_norm, base_angle2_y)
        else:
            # Sheet 2 will be bent perpendicular to sheet 1
            for attempt in range(20):
                bent_angle2_x = base_angle2_x + random.choice([-1, 1]) * random.uniform(70, 90)
                bent_angle2_y = base_angle2_y + random.choice([-1, 1]) * random.uniform(0, 20)
                
                bent_angle2_x_rad = np.radians(bent_angle2_x)
                bent_angle2_y_rad = np.radians(bent_angle2_y)
                bent_normal = calculate_plane_normal(bent_angle2_x_rad, bent_angle2_y_rad)
                
                dot_product = np.clip(np.dot(bent_normal, flat_normal), -1.0, 1.0)
                angle_between = np.degrees(np.arccos(abs(dot_product)))
                
                if 75 <= angle_between <= 105:
                    break
            
            # Adjust for direction
            if bend_direction == 'away':
                bent_angle2_x = base_angle2_x - (bent_angle2_x - base_angle2_x)
            
            # Sheet 1 stays flat
            angle1_x_map = np.full_like(X_a_norm, base_angle1_x)
            angle1_y_map = np.full_like(Y_a_norm, base_angle1_y)
            # Interpolate angles for sheet 2 based on transition mask
            angle2_x_map = base_angle2_x + (bent_angle2_x - base_angle2_x) * (1.0 - transition_mask)
            angle2_y_map = base_angle2_y + (bent_angle2_y - base_angle2_y) * (1.0 - transition_mask)
        
        # Convert to radians
        angle1_x_rad = np.radians(angle1_x_map)
        angle1_y_rad = np.radians(angle1_y_map)
        angle2_x_rad = np.radians(angle2_x_map)
        angle2_y_rad = np.radians(angle2_y_map)
        
        # Apply tilt effects (scaled to mm) - varies across the sheet
        tilt_amplifier = 1.0
        tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
        
        # Create base Z coordinates for patch A (relative to center, in mm)
        z_a_base = tilt_factor1 + texture1 + curvature1
        
        # Final coordinates for patch A (add center offsets)
        x_a_final = X_a + center_a_x
        y_a_final = Y_a + center_a_y
        z_a_final = z_a_base + center_a_z
        
        # Create base Z coordinates for patch B (relative to center, in mm)
        z_b_base = tilt_factor2 + texture2 + curvature2
        
        # Final coordinates for patch B (add center offsets and XY separation)
        x_b_final = X_a + center_b_x
        y_b_final = Y_a + center_b_y
        z_b_final = z_b_base + center_b_z
        
        # Ensure bent sheet points do not go past the other sheet
        # Check if bent sheet (sheet 1 or 2) has points that extend past the flat sheet
        if bent_sheet == 1:
            # Sheet 1 is bent, sheet 2 is flat
            flat_sheet_z = z_b_final  # Sheet 2 is flat
            bent_sheet_z = z_a_final  # Sheet 1 is bent
            
            # Calculate the minimum Z of the flat sheet for reference
            flat_sheet_min_z = np.min(flat_sheet_z)
            
            # If bending towards, bent sheet should not go past flat sheet
            # If bending away, bent sheet should not go too far past
            if bend_direction == 'towards':
                # Bent sheet should not go past flat sheet
                # Clip bent sheet Z to not exceed flat sheet min Z (with some margin)
                max_allowed_z = flat_sheet_min_z - 100.0  # 100mm margin
                z_a_final = np.clip(z_a_final, z_a_final.min(), max_allowed_z)
            else:
                # Bent sheet bending away - ensure it doesn't go too far
                # Limit the maximum Z difference to keep sheets close
                max_z_diff = z_sep * 1.2  # Don't go more than 1.2x the separation
                z_a_final = np.clip(z_a_final, z_a_final.min(), center_a_z + max_z_diff)
        else:
            # Sheet 2 is bent, sheet 1 is flat
            flat_sheet_z = z_a_final  # Sheet 1 is flat
            bent_sheet_z = z_b_final  # Sheet 2 is bent
            
            flat_sheet_min_z = np.min(flat_sheet_z)
            
            if bend_direction == 'towards':
                # Bent sheet should not go past flat sheet
                max_allowed_z = flat_sheet_min_z - 100.0  # 100mm margin
                z_b_final = np.clip(z_b_final, z_b_final.min(), max_allowed_z)
            else:
                # Bent sheet bending away
                max_z_diff = z_sep * 1.2
                z_b_final = np.clip(z_b_final, z_b_final.min(), center_b_z + max_z_diff)
        
        # Create patch A and B XYZ arrays
        xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        
        # Ensure sheets do not intersect
        # Use center_b values directly (they are always defined)
        center_b_x_current = center_b_x
        center_b_y_current = center_b_y
        
        xyz_a, xyz_b, center_b_x_new, center_b_y_new, center_b_z = ensure_sheets_no_intersection(
            xyz_a, xyz_b, X_a, Y_a, z_a_final, z_b_base,
            center_a_x, center_a_y, center_a_z,
            center_b_x_current, center_b_y_current, center_b_z,
            min_separation_mm=50.0
        )
        # Update final coordinates if offsets changed
        if center_b_x_new != center_b_x_current or center_b_y_new != center_b_y_current:
            x_b_final = X_a + center_b_x_new
            y_b_final = Y_a + center_b_y_new
            z_b_final = z_b_base + center_b_z
            xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
            # Update center_b values for consistency
            center_b_x = center_b_x_new
            center_b_y = center_b_y_new
        
        # Pivot both sheets together as a unit to add variation
        xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
        
        # Centralize the centroid of the pair to origin
        xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
        
        # Save using the same format as good patches
        prefix = f"bad_sheet{pair_idx:03d}"
        
        # Save as numpy arrays
        np.save(os.path.join(category_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
        np.save(os.path.join(category_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
        
        # Save as text files (CSV format)
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(category_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Save as PLY point cloud files
        save_ply(xyz_a, os.path.join(category_dir, f"{prefix}_sheet1.ply"))
        save_ply(xyz_b, os.path.join(category_dir, f"{prefix}_sheet2.ply"))
        
        if pair_idx % 25 == 0 or pair_idx == 1:
            xy_distance = np.sqrt(xy_offset_x**2 + xy_offset_y**2) if "too_much_xy_sep" in violation_type else 0
            bent_pct = np.mean(transition_mask < 1.0) * 100
            print(f"Generated pair {pair_idx}/{num_pairs}:")
            print(f"  Violation type: {violation_type} + bent sheets")
            print(f"  Patch A center: [{center_a_x:.1f}, {center_a_y:.1f}, {center_a_z:.1f}] mm")
            print(f"  Patch B center: [{center_b_x:.1f}, {center_b_y:.1f}, {center_b_z:.1f}] mm")
            print(f"  Z separation: {z_sep:.1f} mm, XY separation: {xy_distance:.1f} mm")
            print(f"  Bent sheet: Sheet {bent_sheet}, Bend axis: {bend_axis}, Direction: {bend_direction}, Bent area: {bent_pct:.1f}%")
            print()
    
    print(f"Successfully generated {num_pairs} bad multi-violation pairs")
    print(f"Each sheet has {num_points} points (32x32 grid)")
    print(f"Coordinates are in millimeters (mm)")
    print(f"\nFiles saved in '{category_dir}' directory:")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.npy and bad_sheet{{i:03d}}_sheet2_xyz.npy (NumPy arrays)")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.txt and bad_sheet{{i:03d}}_sheet2_xyz.txt (CSV text files)")
    print(f"  - bad_sheet{{i:03d}}_sheet1.ply and bad_sheet{{i:03d}}_sheet2.ply (PLY point clouds)")

def create_all_bad_patches_random_order(num_pairs_per_category=100):
    """
    Create all bad patch categories and save them directly in bad_patches directory.
    
    Args:
        num_pairs_per_category: Number of pairs to generate per category (default: 100)
    """
    import os
    
    # Create output directory
    output_dir = "bad_patches"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    patch_size = 32
    num_points = patch_size * patch_size  # 1024 points
    
    print(f"Generating bad patches from all categories...")
    print(f"Each category will generate {num_pairs_per_category} pairs")
    print(f"Total pairs: {num_pairs_per_category * 5}")
    print(f"Files will be saved directly in '{output_dir}' directory (no folder separation)")
    print()
    
    # Global counter for unique file names across all categories
    global_pair_idx = 1
    
    # Category 1: Same plane pairs
    print("Generating same plane pairs...")
    for pair_idx in range(1, num_pairs_per_category + 1):
        pair_data = _generate_same_plane_pair(patch_size)
        if pair_data:
            xyz_a, xyz_b, metadata = pair_data
            prefix = f"bad_sheet{global_pair_idx:03d}"
            
            # Save as numpy arrays
            np.save(os.path.join(output_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
            np.save(os.path.join(output_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
            
            # Save as text files (CSV format)
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            
            # Save as PLY point cloud files
            save_ply(xyz_a, os.path.join(output_dir, f"{prefix}_sheet1.ply"))
            save_ply(xyz_b, os.path.join(output_dir, f"{prefix}_sheet2.ply"))
            
            if pair_idx % 25 == 0 or pair_idx == 1:
                print(f"  Saved pair {pair_idx}/{num_pairs_per_category}: {prefix}_*")
            global_pair_idx += 1
    print(f"  Completed same plane pairs")
    
    # Category 2: Large angle pairs
    print("\nGenerating large angle pairs...")
    for pair_idx in range(1, num_pairs_per_category + 1):
        pair_data = _generate_large_angle_pair(patch_size)
        if pair_data:
            xyz_a, xyz_b, metadata = pair_data
            prefix = f"bad_sheet{global_pair_idx:03d}"
            
            # Save as numpy arrays
            np.save(os.path.join(output_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
            np.save(os.path.join(output_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
            
            # Save as text files (CSV format)
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            
            # Save as PLY point cloud files
            save_ply(xyz_a, os.path.join(output_dir, f"{prefix}_sheet1.ply"))
            save_ply(xyz_b, os.path.join(output_dir, f"{prefix}_sheet2.ply"))
            
            if pair_idx % 25 == 0 or pair_idx == 1:
                print(f"  Saved pair {pair_idx}/{num_pairs_per_category}: {prefix}_*")
            global_pair_idx += 1
    print(f"  Completed large angle pairs")
    
    # Category 3: Too much XY separation pairs
    print("\nGenerating too much XY separation pairs...")
    for pair_idx in range(1, num_pairs_per_category + 1):
        pair_data = _generate_too_much_xy_separation_pair(patch_size)
        if pair_data:
            xyz_a, xyz_b, metadata = pair_data
            prefix = f"bad_sheet{global_pair_idx:03d}"
            
            # Save as numpy arrays
            np.save(os.path.join(output_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
            np.save(os.path.join(output_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
            
            # Save as text files (CSV format)
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            
            # Save as PLY point cloud files
            save_ply(xyz_a, os.path.join(output_dir, f"{prefix}_sheet1.ply"))
            save_ply(xyz_b, os.path.join(output_dir, f"{prefix}_sheet2.ply"))
            
            if pair_idx % 25 == 0 or pair_idx == 1:
                print(f"  Saved pair {pair_idx}/{num_pairs_per_category}: {prefix}_*")
            global_pair_idx += 1
    print(f"  Completed too much XY separation pairs")
    
    # Category 4: Too close sheets pairs (very little separation)
    print("\nGenerating too close sheets pairs...")
    for pair_idx in range(1, num_pairs_per_category + 1):
        pair_data = _generate_too_close_sheets_pair(patch_size)
        if pair_data:
            xyz_a, xyz_b, metadata = pair_data
            prefix = f"bad_sheet{global_pair_idx:03d}"
            
            # Save as numpy arrays
            np.save(os.path.join(output_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
            np.save(os.path.join(output_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
            
            # Save as text files (CSV format)
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            
            # Save as PLY point cloud files
            save_ply(xyz_a, os.path.join(output_dir, f"{prefix}_sheet1.ply"))
            save_ply(xyz_b, os.path.join(output_dir, f"{prefix}_sheet2.ply"))
            
            if pair_idx % 25 == 0 or pair_idx == 1:
                print(f"  Saved pair {pair_idx}/{num_pairs_per_category}: {prefix}_*")
            global_pair_idx += 1
    print(f"  Completed too close sheets pairs")
    
    # Category 5: Multi-violation pairs (excessive XY separation + bent sheets)
    print("\nGenerating multi-violation pairs...")
    for pair_idx in range(1, num_pairs_per_category + 1):
        pair_data = _generate_multi_violation_pair(patch_size)
        if pair_data:
            xyz_a, xyz_b, metadata = pair_data
            prefix = f"bad_sheet{global_pair_idx:03d}"
            
            # Save as numpy arrays
            np.save(os.path.join(output_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
            np.save(os.path.join(output_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
            
            # Save as text files (CSV format)
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            np.savetxt(os.path.join(output_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                       header='X,Y,Z', comments='', fmt='%.6f')
            
            # Save as PLY point cloud files
            save_ply(xyz_a, os.path.join(output_dir, f"{prefix}_sheet1.ply"))
            save_ply(xyz_b, os.path.join(output_dir, f"{prefix}_sheet2.ply"))
            
            if pair_idx % 25 == 0 or pair_idx == 1:
                print(f"  Saved pair {pair_idx}/{num_pairs_per_category}: {prefix}_*")
            global_pair_idx += 1
    print(f"  Completed multi-violation pairs")
    
    total_pairs = num_pairs_per_category * 5
    print(f"\nSuccessfully generated and saved {total_pairs} bad patch pairs")
    print(f"Each sheet has {num_points} points (32x32 grid)")
    print(f"Coordinates are in millimeters (mm)")
    print(f"\nAll files saved directly in '{output_dir}' directory:")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.npy and bad_sheet{{i:03d}}_sheet2_xyz.npy (NumPy arrays)")
    print(f"  - bad_sheet{{i:03d}}_sheet1_xyz.txt and bad_sheet{{i:03d}}_sheet2_xyz.txt (CSV text files)")
    print(f"  - bad_sheet{{i:03d}}_sheet1.ply and bad_sheet{{i:03d}}_sheet2.ply (PLY point clouds)")

def _generate_same_plane_pair(patch_size):
    """Generate a single same plane pair. Returns (xyz_a, xyz_b, metadata) or None."""
    # Patch A (cam1) ranges in mm
    patch_a_x_range = (-4000, 4000)
    patch_a_y_range = (-4000, 4000)
    patch_a_z_range = (30000, 65000)
    
    # Generate patch extents - ensure no elongation
    base_extent = random.uniform(28000, 30000)
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    # Generate patch A center
    center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
    center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
    center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
    
    # Small Z separation (0-10mm) for same plane
    small_z_sep = random.uniform(0, 10)
    center_b_z = center_a_z + small_z_sep
    
    # Create coordinate grids
    x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
    y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
    X_a, Y_a = np.meshgrid(x_a, y_a)
    
    X_a_norm = X_a / (patch_extent_x / 2)
    Y_a_norm = Y_a / (patch_extent_y / 2)
    
    # Generate textures and curvatures
    texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
    texture_type_1 = random.choice(texture_types)
    texture_type_2 = random.choice(texture_types)
    texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
    texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
    
    use_too_high_curvature = random.random() < 0.3
    curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
    curvature_type_1 = random.choice(curvature_types)
    curvature_type_2 = random.choice(curvature_types)
    curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    
    # Determine separation first (needed for bend logic)
    # Choose whether sheets are close together (interconnected) or reasonably apart
    # 30% chance for close sheets (interconnected), 70% for reasonably apart
    is_close = random.random() < 0.3
    
    # Choose whether sheets are side-by-side horizontally, vertically, or diagonally
    orientation = random.choice(['horizontal', 'vertical', 'diagonal'])
    
    if is_close:
        # Interconnected sheets: edges should touch, not intersect through middle
        # Calculate offset so edges touch (approximately patch_extent distance)
        # Add small gap (0-2% of patch extent) to ensure edges touch but don't overlap
        edge_gap_factor = random.uniform(0.98, 1.02)  # 98-102% of extent (slight gap or slight overlap at edges only)
        
        if orientation == 'horizontal':
            # Edges touch horizontally: offset = patch_extent_x (so right edge of A touches left edge of B)
            xy_offset_x = random.choice([-1, 1]) * patch_extent_x * edge_gap_factor
            xy_offset_y = 0
        elif orientation == 'vertical':
            # Edges touch vertically: offset = patch_extent_y
            xy_offset_x = 0
            xy_offset_y = random.choice([-1, 1]) * patch_extent_y * edge_gap_factor
        else:  # diagonal
            # Edges touch diagonally: use average extent or both extents
            avg_extent = (patch_extent_x + patch_extent_y) / 2.0
            xy_offset_x = random.choice([-1, 1]) * avg_extent * edge_gap_factor
            xy_offset_y = random.choice([-1, 1]) * avg_extent * edge_gap_factor
    else:
        # Reasonably apart: 20k-50k mm (20-50m) - with bends
        apart_range = (20000, 50000)  # mm
        if orientation == 'horizontal':
            xy_offset_x = random.choice([-1, 1]) * random.uniform(apart_range[0], apart_range[1])
            xy_offset_y = 0
        elif orientation == 'vertical':
            xy_offset_x = 0
            xy_offset_y = random.choice([-1, 1]) * random.uniform(apart_range[0], apart_range[1])
        else:  # diagonal
            xy_offset_x = random.choice([-1, 1]) * random.uniform(apart_range[0], apart_range[1])
            xy_offset_y = random.choice([-1, 1]) * random.uniform(apart_range[0], apart_range[1])
    
    center_b_x = center_a_x + xy_offset_x
    center_b_y = center_a_y + xy_offset_y
    
    # Generate base angles for sheets (small angles, relatively parallel for same plane)
    base_angle1_x = random.uniform(-5, 5)
    base_angle1_y = random.uniform(-5, 5)
    base_angle2_x = random.uniform(-5, 5)
    base_angle2_y = random.uniform(-5, 5)
    
    # For close sheets: create interconnected bends (always apply, similar patterns, facing toward/away)
    # For reasonably apart: apply bends (80% chance)
    if is_close:
        # Interconnected sheets: both sheets always bent with similar patterns
        bend_sheet_1 = True
        bend_sheet_2 = True
        
        # Determine bend axis and direction based on relative positions
        # Bend should be from the edge FARTHER from the other sheet (not the touching edge)
        # Sheet 1 bends from its edge farthest from sheet 2
        # Sheet 2 bends from its edge farthest from sheet 1
        if orientation == 'horizontal':
            # Horizontal: bend along X axis
            bend_axis = 'x'
            # If sheet 2 is to the right (xy_offset_x > 0), sheet 1 is on the left
            # Sheet 1's farthest edge from sheet 2 is LEFT, Sheet 2's farthest edge from sheet 1 is RIGHT
            if xy_offset_x > 0:
                # Sheet 2 is to the right: sheet 1 bends from LEFT (farthest from sheet 2), sheet 2 bends from RIGHT (farthest from sheet 1)
                bend_from_left_1 = True   # Sheet 1: bend from left (farthest edge)
                bend_from_left_2 = False   # Sheet 2: bend from right (farthest edge)
            else:
                # Sheet 2 is to the left: sheet 1 bends from RIGHT (farthest from sheet 2), sheet 2 bends from LEFT (farthest from sheet 1)
                bend_from_left_1 = False  # Sheet 1: bend from right (farthest edge)
                bend_from_left_2 = True   # Sheet 2: bend from left (farthest edge)
        elif orientation == 'vertical':
            # Vertical: bend along Y axis
            bend_axis = 'y'
            # If sheet 2 is above (xy_offset_y > 0), sheet 1 is below
            # Sheet 1's farthest edge from sheet 2 is BOTTOM, Sheet 2's farthest edge from sheet 1 is TOP
            if xy_offset_y > 0:
                # Sheet 2 is above: sheet 1 bends from BOTTOM (farthest from sheet 2), sheet 2 bends from TOP (farthest from sheet 1)
                bend_from_left_1 = True   # Sheet 1: bend from bottom (farthest edge)
                bend_from_left_2 = False   # Sheet 2: bend from top (farthest edge)
            else:
                # Sheet 2 is below: sheet 1 bends from TOP (farthest from sheet 2), sheet 2 bends from BOTTOM (farthest from sheet 1)
                bend_from_left_1 = False  # Sheet 1: bend from top (farthest edge)
                bend_from_left_2 = True   # Sheet 2: bend from bottom (farthest edge)
        else:  # diagonal
            # Diagonal: choose dominant direction or use both
            if abs(xy_offset_x) > abs(xy_offset_y):
                # X is dominant: use X axis
                bend_axis = 'x'
                if xy_offset_x > 0:
                    # Sheet 2 is to the right: sheet 1 bends from left, sheet 2 bends from right
                    bend_from_left_1 = True
                    bend_from_left_2 = False
                else:
                    # Sheet 2 is to the left: sheet 1 bends from right, sheet 2 bends from left
                    bend_from_left_1 = False
                    bend_from_left_2 = True
            else:
                # Y is dominant: use Y axis
                bend_axis = 'y'
                if xy_offset_y > 0:
                    # Sheet 2 is above: sheet 1 bends from bottom, sheet 2 bends from top
                    bend_from_left_1 = True
                    bend_from_left_2 = False
                else:
                    # Sheet 2 is below: sheet 1 bends from top, sheet 2 bends from bottom
                    bend_from_left_1 = False
                    bend_from_left_2 = True
        
        # Transition point: at least 50% of points should be bent (facing toward/away), up to almost 90% bent
        # transition_point = 0.5 means 50% parallel, 50% bent
        # transition_point = 0.1 means 10% parallel, 90% bent (almost 90%)
        transition_point = random.uniform(0.1, 0.5)  # 10-50% parallel, 50-90% bent (at least 50% bent, up to almost 90%)
        
        # Create transition mask for bend - each sheet bends from its far edge
        if bend_axis == 'x':
            if bend_from_left_1:
                transition_mask_1 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
            else:
                transition_mask_1 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1)
            if bend_from_left_2:
                transition_mask_2 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
            else:
                transition_mask_2 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1)
        else:
            if bend_from_left_1:
                transition_mask_1 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
            else:
                transition_mask_1 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1)
            if bend_from_left_2:
                transition_mask_2 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
            else:
                transition_mask_2 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1)
        
        # Calculate direction from sheet 1 to sheet 2 for facing toward/away
        direction_to_sheet2 = np.array([xy_offset_x, xy_offset_y])
        if np.linalg.norm(direction_to_sheet2) > 0:
            direction_to_sheet2 = direction_to_sheet2 / np.linalg.norm(direction_to_sheet2)
        else:
            direction_to_sheet2 = np.array([1, 0])  # Default direction
        
        # Determine if sheets face toward or away from each other
        face_toward = random.random() > 0.5  # 50% chance toward, 50% away
        
        # Base bend magnitude: up to almost 90 degrees (similar for both - continuation effect)
        base_bend_magnitude_x = random.uniform(45, 85)  # 45-85 degrees (up to almost 90)
        base_bend_magnitude_y = random.uniform(30, 70)  # 30-70 degrees
        
        # Sheet 1: bend toward or away from sheet 2
        if face_toward:
            # Face toward: bend in direction of sheet 2
            bend_dir_1_x = np.sign(direction_to_sheet2[0]) if abs(direction_to_sheet2[0]) > 0.1 else random.choice([-1, 1])
            bend_dir_1_y = np.sign(direction_to_sheet2[1]) if abs(direction_to_sheet2[1]) > 0.1 else random.choice([-1, 1])
        else:
            # Face away: bend opposite to direction of sheet 2
            bend_dir_1_x = -np.sign(direction_to_sheet2[0]) if abs(direction_to_sheet2[0]) > 0.1 else random.choice([-1, 1])
            bend_dir_1_y = -np.sign(direction_to_sheet2[1]) if abs(direction_to_sheet2[1]) > 0.1 else random.choice([-1, 1])
        
        # Sheet 2: bend toward or away from sheet 1 (opposite direction)
        if face_toward:
            # Face toward: bend toward sheet 1 (opposite direction)
            bend_dir_2_x = -bend_dir_1_x
            bend_dir_2_y = -bend_dir_1_y
        else:
            # Face away: bend away from sheet 1 (same direction as sheet 1)
            bend_dir_2_x = bend_dir_1_x
            bend_dir_2_y = bend_dir_1_y
        
        # Apply similar bends with slight variation (continuation effect)
        bent_angle1_x = base_angle1_x + bend_dir_1_x * base_bend_magnitude_x * random.uniform(0.9, 1.1)
        bent_angle1_y = base_angle1_y + bend_dir_1_y * base_bend_magnitude_y * random.uniform(0.9, 1.1)
        bent_angle2_x = base_angle2_x + bend_dir_2_x * base_bend_magnitude_x * random.uniform(0.9, 1.1)
        bent_angle2_y = base_angle2_y + bend_dir_2_y * base_bend_magnitude_y * random.uniform(0.9, 1.1)
        
        # Interpolate angles based on transition mask
        angle1_x_map = base_angle1_x + (bent_angle1_x - base_angle1_x) * (1.0 - transition_mask_1)
        angle1_y_map = base_angle1_y + (bent_angle1_y - base_angle1_y) * (1.0 - transition_mask_1)
        angle2_x_map = base_angle2_x + (bent_angle2_x - base_angle2_x) * (1.0 - transition_mask_2)
        angle2_y_map = base_angle2_y + (bent_angle2_y - base_angle2_y) * (1.0 - transition_mask_2)
        
    else:
        # Reasonably apart: apply bends (80% chance)
        use_bend = random.random() < 0.8
        
        if use_bend:
            # Choose which sheet(s) to bend (both can be bent, or just one)
            bend_sheet_1 = random.random() < 0.7  # 70% chance sheet 1 is bent
            bend_sheet_2 = random.random() < 0.7  # 70% chance sheet 2 is bent
            
            # Choose bend axis and direction
            bend_axis = random.choice(['x', 'y'])
            bend_from_left = random.random() > 0.5
            
            # Slight bend: 20-40% of sheet is bent (similar to good patches)
            transition_point = random.uniform(0.6, 0.8)  # 60-80% parallel, 20-40% bent
            
            # Create transition mask for slight bend
            if bend_axis == 'x':
                if bend_from_left:
                    transition_mask_1 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                    transition_mask_2 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
                else:
                    transition_mask_1 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                    transition_mask_2 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
            else:
                if bend_from_left:
                    transition_mask_1 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                    transition_mask_2 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
                else:
                    transition_mask_1 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                    transition_mask_2 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
            
            # Apply slight bends: add 10-30 degrees (slight, not extreme)
            if bend_sheet_1:
                bent_angle1_x = base_angle1_x + random.choice([-1, 1]) * random.uniform(10, 30)
                bent_angle1_y = base_angle1_y + random.choice([-1, 1]) * random.uniform(0, 15)
                # Interpolate angles based on transition mask
                angle1_x_map = base_angle1_x + (bent_angle1_x - base_angle1_x) * (1.0 - transition_mask_1)
                angle1_y_map = base_angle1_y + (bent_angle1_y - base_angle1_y) * (1.0 - transition_mask_1)
            else:
                angle1_x_map = np.full_like(X_a_norm, base_angle1_x)
                angle1_y_map = np.full_like(Y_a_norm, base_angle1_y)
            
            if bend_sheet_2:
                bent_angle2_x = base_angle2_x + random.choice([-1, 1]) * random.uniform(10, 30)
                bent_angle2_y = base_angle2_y + random.choice([-1, 1]) * random.uniform(0, 15)
                # Interpolate angles based on transition mask
                angle2_x_map = base_angle2_x + (bent_angle2_x - base_angle2_x) * (1.0 - transition_mask_2)
                angle2_y_map = base_angle2_y + (bent_angle2_y - base_angle2_y) * (1.0 - transition_mask_2)
            else:
                angle2_x_map = np.full_like(X_a_norm, base_angle2_x)
                angle2_y_map = np.full_like(Y_a_norm, base_angle2_y)
        else:
            # No bend - use uniform angles
            angle1_x_map = np.full_like(X_a_norm, base_angle1_x)
            angle1_y_map = np.full_like(Y_a_norm, base_angle1_y)
            angle2_x_map = np.full_like(X_a_norm, base_angle2_x)
            angle2_y_map = np.full_like(Y_a_norm, base_angle2_y)
    
    # Convert to radians
    angle1_x_rad = np.radians(angle1_x_map)
    angle1_y_rad = np.radians(angle1_y_map)
    angle2_x_rad = np.radians(angle2_x_map)
    angle2_y_rad = np.radians(angle2_y_map)
    
    tilt_amplifier = 1.0
    tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
    tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
    
    # Create coordinates
    z_a_base = tilt_factor1 + texture1 + curvature1
    z_b_base = tilt_factor2 + texture2 + curvature2
    
    x_a_final = X_a + center_a_x
    y_a_final = Y_a + center_a_y
    z_a_final = z_a_base + center_a_z
    
    x_b_final = X_a + center_b_x
    y_b_final = Y_a + center_b_y
    z_b_final = z_b_base + center_b_z
    
    xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # For interconnected sheets, ensure edges touch but don't intersect through middle
    # For other cases, no special check needed (sheets are reasonably apart)
    if is_close:
        # For interconnected sheets, verify edges touch (not intersecting through middle)
        # Calculate bounding boxes
        xyz_a_min = xyz_a.min(axis=0)
        xyz_a_max = xyz_a.max(axis=0)
        xyz_b_min = xyz_b.min(axis=0)
        xyz_b_max = xyz_b.max(axis=0)
        
        # Check if sheets overlap in the middle (bad for interconnected)
        # They should only touch at edges
        overlap_x = not (xyz_a_max[0] < xyz_b_min[0] or xyz_b_max[0] < xyz_a_min[0])
        overlap_y = not (xyz_a_max[1] < xyz_b_min[1] or xyz_b_max[1] < xyz_a_min[1])
        
        # If they overlap in both X and Y (intersecting through middle), adjust slightly
        if overlap_x and overlap_y:
            # Adjust to ensure only edge contact
            if orientation == 'horizontal':
                # Ensure only X edges touch
                if xy_offset_x > 0:
                    center_b_x = center_a_x + patch_extent_x * 1.01  # Slight gap
                else:
                    center_b_x = center_a_x - patch_extent_x * 1.01
                x_b_final = X_a + center_b_x
                y_b_final = Y_a + center_b_y
                z_b_final = z_b_base + center_b_z
                xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
            elif orientation == 'vertical':
                # Ensure only Y edges touch
                if xy_offset_y > 0:
                    center_b_y = center_a_y + patch_extent_y * 1.01  # Slight gap
                else:
                    center_b_y = center_a_y - patch_extent_y * 1.01
                x_b_final = X_a + center_b_x
                y_b_final = Y_a + center_b_y
                z_b_final = z_b_base + center_b_z
                xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
            else:  # diagonal
                # Adjust to ensure only corner edges touch
                avg_extent = (patch_extent_x + patch_extent_y) / 2.0
                if xy_offset_x > 0:
                    center_b_x = center_a_x + avg_extent * 1.01
                else:
                    center_b_x = center_a_x - avg_extent * 1.01
                if xy_offset_y > 0:
                    center_b_y = center_a_y + avg_extent * 1.01
                else:
                    center_b_y = center_a_y - avg_extent * 1.01
                x_b_final = X_a + center_b_x
                y_b_final = Y_a + center_b_y
                z_b_final = z_b_base + center_b_z
                xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # Pivot both sheets together as a unit to add variation
    xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
    
    # Centralize the centroid of the pair to origin
    xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
    
    metadata = {
        "category": "same_plane",
        "texture_1": texture_type_1,
        "texture_2": texture_type_2,
        "curvature_1": curvature_type_1,
        "curvature_2": curvature_type_2
    }
    
    return (xyz_a, xyz_b, metadata)

def _generate_large_angle_pair(patch_size):
    """Generate a single large angle pair. Returns (xyz_a, xyz_b, metadata) or None."""
    # Similar structure but with large angles - I'll implement the key parts
    # This is a simplified version - you may want to copy the full logic from create_bad_large_angle_pairs
    patch_a_x_range = (-4000, 4000)
    patch_a_y_range = (-4000, 4000)
    patch_a_z_range = (30000, 65000)
    z_separation_range = (30000, 65000)
    # Ensure no elongation - keep aspect ratio close to 1
    base_extent = random.uniform(28000, 30000)
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    large_amplifier = 3.0  # Increased to 3.0 to ensure angles are clearly visible
    
    center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
    center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
    center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
    z_sep = random.uniform(z_separation_range[0], z_separation_range[1])
    center_b_z = center_a_z - z_sep
    
    x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
    y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
    X_a, Y_a = np.meshgrid(x_a, y_a)
    X_a_norm = X_a / (patch_extent_x / 2)
    Y_a_norm = Y_a / (patch_extent_y / 2)
    
    texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
    texture_type_1 = random.choice(texture_types)
    texture_type_2 = random.choice(texture_types)
    texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
    texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
    
    use_too_high_curvature = random.random() < 0.3
    curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
    curvature_type_1 = random.choice(curvature_types)
    curvature_type_2 = random.choice(curvature_types)
    curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    
    # Generate large angles (35-90 degrees between sheets)
    min_angle_between_sheets = 30.0
    max_angle_between_sheets = 90.0  # Up to perpendicular
    
    def calculate_plane_normal(angle_x_rad, angle_y_rad):
        normal = np.array([np.tan(angle_y_rad), np.tan(angle_x_rad), 1.0])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        return normal
    
    for attempt in range(10):
        # Generate base angle for sheet 1 - ensure it's substantial (not too flat)
        # Minimum magnitude of 15 degrees to ensure visible tilt
        angle1_x_mag = random.uniform(15, 40)  # Magnitude of tilt around X-axis
        angle1_y_mag = random.uniform(15, 40)  # Magnitude of tilt around Y-axis
        angle1_x = angle1_x_mag * random.choice([-1, 1])  # Base tilt around X-axis for sheet 1
        angle1_y = angle1_y_mag * random.choice([-1, 1])  # Base tilt around Y-axis for sheet 1
        angle1_x_rad = np.radians(angle1_x)
        angle1_y_rad = np.radians(angle1_y)
        
        # Generate angle difference for sheet 2 - ensure substantial difference
        # At least 30 degrees in at least one component
        angle_diff_x = random.uniform(30, 60) * random.choice([-1, 1])
        angle_diff_y = random.uniform(30, 60) * random.choice([-1, 1])
        
        angle2_x = angle1_x + angle_diff_x
        angle2_y = angle1_y + angle_diff_y
        angle2_x_rad = np.radians(angle2_x)
        angle2_y_rad = np.radians(angle2_y)
        
        normal1 = calculate_plane_normal(angle1_x_rad, angle1_y_rad)
        normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
        dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
        angle_between_planes_deg = np.degrees(np.arccos(dot_product))
        
        if min_angle_between_sheets < angle_between_planes_deg < max_angle_between_sheets:
            break
        
        if angle_between_planes_deg <= min_angle_between_sheets:
            scale_factor = (min_angle_between_sheets + 5) / angle_between_planes_deg
            angle_diff_x *= scale_factor
            angle_diff_y *= scale_factor
            angle2_x = angle1_x + angle_diff_x
            angle2_y = angle1_y + angle_diff_y
            angle2_x_rad = np.radians(angle2_x)
            angle2_y_rad = np.radians(angle2_y)
        elif angle_between_planes_deg >= max_angle_between_sheets:
            scale_factor = (max_angle_between_sheets - 5) / angle_between_planes_deg
            angle_diff_x *= scale_factor
            angle_diff_y *= scale_factor
            angle2_x = angle1_x + angle_diff_x
            angle2_y = angle1_y + angle_diff_y
            angle2_x_rad = np.radians(angle2_x)
            angle2_y_rad = np.radians(angle2_y)
    
    # Main augmentation: Apply bends to sheets (80% chance - main augmentation)
    use_bend = random.random() < 0.8
    
    if use_bend:
        # Choose which sheet(s) to bend (both can be bent, or just one)
        bend_sheet_1 = random.random() < 0.8  # 80% chance sheet 1 is bent
        bend_sheet_2 = random.random() < 0.8  # 80% chance sheet 2 is bent
        
        # Choose bend axis and direction
        bend_axis = random.choice(['x', 'y'])
        bend_from_left = random.random() > 0.5
        
        # Bend: 30-50% of sheet is bent (more than slight bends)
        transition_point = random.uniform(0.5, 0.7)  # 50-70% parallel, 30-50% bent
        
        # Create transition mask for bend
        if bend_axis == 'x':
            if bend_from_left:
                transition_mask_1 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                transition_mask_2 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
            else:
                transition_mask_1 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                transition_mask_2 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
        else:
            if bend_from_left:
                transition_mask_1 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                transition_mask_2 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
            else:
                transition_mask_1 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                transition_mask_2 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
        
        # Apply bends: add 15-40 degrees (moderate to strong bends)
        if bend_sheet_1:
            bent_angle1_x = angle1_x + random.choice([-1, 1]) * random.uniform(15, 40)
            bent_angle1_y = angle1_y + random.choice([-1, 1]) * random.uniform(10, 30)
            # Interpolate angles based on transition mask
            angle1_x_map = angle1_x + (bent_angle1_x - angle1_x) * (1.0 - transition_mask_1)
            angle1_y_map = angle1_y + (bent_angle1_y - angle1_y) * (1.0 - transition_mask_1)
        else:
            angle1_x_map = np.full_like(X_a_norm, angle1_x)
            angle1_y_map = np.full_like(Y_a_norm, angle1_y)
        
        if bend_sheet_2:
            bent_angle2_x = angle2_x + random.choice([-1, 1]) * random.uniform(15, 40)
            bent_angle2_y = angle2_y + random.choice([-1, 1]) * random.uniform(10, 30)
            # Interpolate angles based on transition mask
            angle2_x_map = angle2_x + (bent_angle2_x - angle2_x) * (1.0 - transition_mask_2)
            angle2_y_map = angle2_y + (bent_angle2_y - angle2_y) * (1.0 - transition_mask_2)
        else:
            angle2_x_map = np.full_like(X_a_norm, angle2_x)
            angle2_y_map = np.full_like(Y_a_norm, angle2_y)
        
        # Convert to radians for tilt calculation
        angle1_x_rad_map = np.radians(angle1_x_map)
        angle1_y_rad_map = np.radians(angle1_y_map)
        angle2_x_rad_map = np.radians(angle2_x_map)
        angle2_y_rad_map = np.radians(angle2_y_map)
        
        # Apply tilt effects with varying angles (bent sheets)
        tilt_factor1 = (np.tan(angle1_x_rad_map) * Y_a + np.tan(angle1_y_rad_map) * X_a) * large_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad_map) * Y_a + np.tan(angle2_y_rad_map) * X_a) * large_amplifier
    else:
        # No bend - use uniform angles
        tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * large_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * large_amplifier
    
    # Small XY offset to keep sheets close together (not elongated)
    # Use small offset regardless of angle to ensure points stay close
    offset_factor = random.uniform(0.05, 0.15)  # Small offset to keep sheets close
    min_xy_offset = offset_factor * max(patch_extent_x, patch_extent_y)
    offset_direction = random.uniform(0, 2 * np.pi)
    xy_offset_x = np.cos(offset_direction) * min_xy_offset
    xy_offset_y = np.sin(offset_direction) * min_xy_offset
    
    center_b_x = center_a_x + xy_offset_x
    center_b_y = center_a_y + xy_offset_y
    
    z_a_base = tilt_factor1 + texture1 + curvature1
    z_b_base = tilt_factor2 + texture2 + curvature2
    
    # Clamp Z values to prevent extreme elongation (limit to reasonable range)
    # Increased to 0.4 to allow more angle visibility while still preventing elongation
    max_z_variation = patch_extent_x * 0.4  # Max 40% of patch extent in Z
    z_a_base = np.clip(z_a_base, -max_z_variation, max_z_variation)
    z_b_base = np.clip(z_b_base, -max_z_variation, max_z_variation)
    
    x_a_final = X_a + center_a_x
    y_a_final = Y_a + center_a_y
    z_a_final = z_a_base + center_a_z
    
    x_b_final = X_a + center_b_x
    y_b_final = Y_a + center_b_y
    z_b_final = z_b_base + center_b_z
    
    xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # Ensure points stay close together regardless of angle
    point_distances = np.linalg.norm(xyz_a - xyz_b, axis=1)
    mean_distance = np.mean(point_distances)
    target_mean_distance = z_sep * 1.8
    if mean_distance > target_mean_distance * 1.5:
        # Points are too spread out - reduce the XY offset
        scale_factor = (target_mean_distance * 1.2) / mean_distance
        xy_offset_x = xy_offset_x * scale_factor
        xy_offset_y = xy_offset_y * scale_factor
        center_b_x = center_a_x + xy_offset_x
        center_b_y = center_a_y + xy_offset_y
        
        # Recalculate coordinates
        x_b_final = X_a + center_b_x
        y_b_final = Y_a + center_b_y
        z_b_final = z_b_base + center_b_z
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # Ensure sheets do not intersect (or only edges touch)
    # Check for intersection and adjust if needed
    xyz_a_min = xyz_a.min(axis=0)
    xyz_a_max = xyz_a.max(axis=0)
    xyz_b_min = xyz_b.min(axis=0)
    xyz_b_max = xyz_b.max(axis=0)
    
    # Check if sheets overlap in the middle (bad - they should not intersect)
    overlap_x = not (xyz_a_max[0] < xyz_b_min[0] or xyz_b_max[0] < xyz_a_min[0])
    overlap_y = not (xyz_a_max[1] < xyz_b_min[1] or xyz_b_max[1] < xyz_a_min[1])
    
    # If they overlap in both X and Y (intersecting through middle), adjust
    if overlap_x and overlap_y:
        # Calculate separation needed to avoid intersection
        # Use bounding box centers to determine direction
        center_a_bb = (xyz_a_min + xyz_a_max) / 2.0
        center_b_bb = (xyz_b_min + xyz_b_max) / 2.0
        direction = center_b_bb[:2] - center_a_bb[:2]
        if np.linalg.norm(direction) < 1e-6:
            # If centers are too close, use random direction
            direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        direction = direction / np.linalg.norm(direction)
        
        # Calculate minimum separation needed
        extent_a = np.max(xyz_a_max[:2] - xyz_a_min[:2])
        extent_b = np.max(xyz_b_max[:2] - xyz_b_min[:2])
        min_sep_needed = (extent_a + extent_b) / 2.0 + 100.0  # Add small buffer
        
        # Adjust XY offset to ensure no intersection
        xy_offset_x = direction[0] * min_sep_needed
        xy_offset_y = direction[1] * min_sep_needed
        center_b_x = center_a_x + xy_offset_x
        center_b_y = center_a_y + xy_offset_y
        
        # Recalculate coordinates
        x_b_final = X_a + center_b_x
        y_b_final = Y_a + center_b_y
        z_b_final = z_b_base + center_b_z
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # Pivot both sheets together as a unit to add variation
    xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
    
    # Centralize the centroid of the pair to origin
    xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
    
    metadata = {
        "category": "large_angle",
        "texture_1": texture_type_1,
        "texture_2": texture_type_2,
        "curvature_1": curvature_type_1,
        "curvature_2": curvature_type_2
    }
    
    return (xyz_a, xyz_b, metadata)

def _generate_extreme_texture_pair(patch_size):
    """Generate a single extreme texture pair. Returns (xyz_a, xyz_b, metadata) or None."""
    patch_a_x_range = (-4000, 4000)
    patch_a_y_range = (-4000, 4000)
    patch_a_z_range = (30000, 65000)
    z_separation_range = (30000, 65000)
    # Ensure no elongation - keep aspect ratio close to 1
    base_extent = random.uniform(28000, 30000)
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
    center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
    center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
    z_sep = random.uniform(z_separation_range[0], z_separation_range[1])
    center_b_z = center_a_z - z_sep
    
    x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
    y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
    X_a, Y_a = np.meshgrid(x_a, y_a)
    X_a_norm = X_a / (patch_extent_x / 2)
    Y_a_norm = Y_a / (patch_extent_y / 2)
    
    texture_types = ["random", "wave", "noise", "grid", "spots"]
    texture_type_1 = random.choice(texture_types)
    texture_type_2 = random.choice(texture_types)
    texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=True)
    texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=True)
    
    curvature1 = np.zeros((patch_size, patch_size))
    curvature2 = np.zeros((patch_size, patch_size))
    
    z_a_base = texture1 + curvature1
    z_b_base = texture2 + curvature2
    
    x_a_final = X_a + center_a_x
    y_a_final = Y_a + center_a_y
    z_a_final = z_a_base + center_a_z
    
    x_b_final = X_a + center_a_x
    y_b_final = Y_a + center_a_y
    z_b_final = z_b_base + center_b_z
    
    xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # Pivot both sheets together as a unit to add variation
    xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
    
    # Centralize the centroid of the pair to origin
    xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
    
    metadata = {
        "category": "extreme_texture",
        "texture_1": texture_type_1,
        "texture_2": texture_type_2,
        "curvature_1": "none",
        "curvature_2": "none"
    }
    
    return (xyz_a, xyz_b, metadata)


def _generate_too_much_xy_separation_pair(patch_size):
    """Generate a single too much XY separation pair. Returns (xyz_a, xyz_b, metadata) or None."""
    patch_a_x_range = (-4000, 4000)
    patch_a_y_range = (-4000, 4000)
    patch_a_z_range = (30000, 65000)
    z_separation_range = (30000, 65000)
    excessive_xy_separation_range = (35000, 90000)
    # Ensure no elongation - keep aspect ratio close to 1
    base_extent = random.uniform(28000, 30000)
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
    center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
    center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
    z_sep = random.uniform(z_separation_range[0], z_separation_range[1])
    center_b_z = center_a_z - z_sep
    
    excessive_xy_sep = random.uniform(excessive_xy_separation_range[0], excessive_xy_separation_range[1])
    offset_direction = random.uniform(0, 2 * np.pi)
    xy_offset_x = np.cos(offset_direction) * excessive_xy_sep
    xy_offset_y = np.sin(offset_direction) * excessive_xy_sep
    
    center_b_x = center_a_x + xy_offset_x
    center_b_y = center_a_y + xy_offset_y
    
    x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
    y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
    X_a, Y_a = np.meshgrid(x_a, y_a)
    X_a_norm = X_a / (patch_extent_x / 2)
    Y_a_norm = Y_a / (patch_extent_y / 2)
    
    texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
    texture_type_1 = random.choice(texture_types)
    texture_type_2 = random.choice(texture_types)
    texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
    texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
    
    use_too_high_curvature = random.random() < 0.3
    curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
    curvature_type_1 = random.choice(curvature_types)
    curvature_type_2 = random.choice(curvature_types)
    curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    
    angle1_x = random.uniform(-5, 5)  # Reduced
    angle1_y = random.uniform(-5, 5)  # Reduced
    angle2_x = random.uniform(-5, 5)  # Reduced
    angle2_y = random.uniform(-5, 5)  # Reduced
    
    # Apply bent sheet augmentation (80% chance - main augmentation)
    use_bend = random.random() < 0.8
    
    if use_bend:
        # Choose which sheet(s) to bend (both can be bent, or just one)
        bend_sheet_1 = random.random() < 0.8  # 80% chance sheet 1 is bent
        bend_sheet_2 = random.random() < 0.8  # 80% chance sheet 2 is bent
        
        # Choose bend axis and direction
        bend_axis = random.choice(['x', 'y'])
        bend_from_left = random.random() > 0.5
        
        # Bend: 30-50% of sheet is bent (more than slight bends)
        transition_point = random.uniform(0.5, 0.7)  # 50-70% parallel, 30-50% bent
        
        # Create transition mask for bend
        if bend_axis == 'x':
            if bend_from_left:
                transition_mask_1 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                transition_mask_2 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
            else:
                transition_mask_1 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                transition_mask_2 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
        else:
            if bend_from_left:
                transition_mask_1 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                transition_mask_2 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
            else:
                transition_mask_1 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                transition_mask_2 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
        
        # Apply bends: add 15-40 degrees (moderate to strong bends)
        if bend_sheet_1:
            bent_angle1_x = angle1_x + random.choice([-1, 1]) * random.uniform(15, 40)
            bent_angle1_y = angle1_y + random.choice([-1, 1]) * random.uniform(10, 30)
            # Interpolate angles based on transition mask
            angle1_x_map = angle1_x + (bent_angle1_x - angle1_x) * (1.0 - transition_mask_1)
            angle1_y_map = angle1_y + (bent_angle1_y - angle1_y) * (1.0 - transition_mask_1)
        else:
            angle1_x_map = np.full_like(X_a_norm, angle1_x)
            angle1_y_map = np.full_like(Y_a_norm, angle1_y)
        
        if bend_sheet_2:
            bent_angle2_x = angle2_x + random.choice([-1, 1]) * random.uniform(15, 40)
            bent_angle2_y = angle2_y + random.choice([-1, 1]) * random.uniform(10, 30)
            # Interpolate angles based on transition mask
            angle2_x_map = angle2_x + (bent_angle2_x - angle2_x) * (1.0 - transition_mask_2)
            angle2_y_map = angle2_y + (bent_angle2_y - angle2_y) * (1.0 - transition_mask_2)
        else:
            angle2_x_map = np.full_like(X_a_norm, angle2_x)
            angle2_y_map = np.full_like(Y_a_norm, angle2_y)
        
        # Convert to radians for tilt calculation
        angle1_x_rad_map = np.radians(angle1_x_map)
        angle1_y_rad_map = np.radians(angle1_y_map)
        angle2_x_rad_map = np.radians(angle2_x_map)
        angle2_y_rad_map = np.radians(angle2_y_map)
        
        # Apply tilt effects with varying angles (bent sheets)
        tilt_amplifier = 1.0
        tilt_factor1 = (np.tan(angle1_x_rad_map) * Y_a + np.tan(angle1_y_rad_map) * X_a) * tilt_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad_map) * Y_a + np.tan(angle2_y_rad_map) * X_a) * tilt_amplifier
    else:
        # No bend - use uniform angles
        angle1_x_rad = np.radians(angle1_x)
        angle1_y_rad = np.radians(angle1_y)
        angle2_x_rad = np.radians(angle2_x)
        angle2_y_rad = np.radians(angle2_y)
        
        tilt_amplifier = 1.0
        tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
    
    z_a_base = tilt_factor1 + texture1 + curvature1
    z_b_base = tilt_factor2 + texture2 + curvature2
    
    x_a_final = X_a + center_a_x
    y_a_final = Y_a + center_a_y
    z_a_final = z_a_base + center_a_z
    
    x_b_final = X_a + center_b_x
    y_b_final = Y_a + center_b_y
    z_b_final = z_b_base + center_b_z
    
    xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # Pivot both sheets together as a unit to add variation
    xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
    
    # Centralize the centroid of the pair to origin
    xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
    
    # Prepare metadata with bend information
    bend_info = {}
    if 'use_bend' in locals() and use_bend:
        bend_info = {
            "bent_sheets": True,
            "bend_sheet_1": bend_sheet_1 if 'bend_sheet_1' in locals() else False,
            "bend_sheet_2": bend_sheet_2 if 'bend_sheet_2' in locals() else False,
            "bend_axis": bend_axis if 'bend_axis' in locals() else None
        }
    else:
        bend_info = {"bent_sheets": False}
    
    metadata = {
        "category": "too_much_xy_separation",
        **bend_info,
        "texture_1": texture_type_1,
        "texture_2": texture_type_2,
        "curvature_1": curvature_type_1,
        "curvature_2": curvature_type_2
    }
    
    return (xyz_a, xyz_b, metadata)

def _generate_too_close_sheets_pair(patch_size):
    """Generate a single too close sheets pair. Returns (xyz_a, xyz_b, metadata) or None."""
    patch_a_x_range = (-4000, 4000)
    patch_a_y_range = (-4000, 4000)
    patch_a_z_range = (30000, 65000)
    # Very small Z separation (1-10mm) - too close
    too_close_z_separation_range = (1000, 10000)  # 1-10mm in mm
    # Ensure no elongation - keep aspect ratio close to 1
    base_extent = random.uniform(28000, 30000)
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
    center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
    center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
    
    # Very small Z separation - sheets are too close
    z_sep = random.uniform(too_close_z_separation_range[0], too_close_z_separation_range[1])
    center_b_z = center_a_z - z_sep
    
    # Very small XY offset to prevent intersection (keep sheets close in all dimensions)
    # Use similar range as Z separation: 1-10mm
    too_close_xy_offset_range = (1000, 10000)  # 1-10mm in mm
    xy_offset = random.uniform(too_close_xy_offset_range[0], too_close_xy_offset_range[1])
    offset_direction = random.uniform(0, 2 * np.pi)
    xy_offset_x = np.cos(offset_direction) * xy_offset
    xy_offset_y = np.sin(offset_direction) * xy_offset
    
    center_b_x = center_a_x + xy_offset_x
    center_b_y = center_a_y + xy_offset_y
    
    x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
    y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
    X_a, Y_a = np.meshgrid(x_a, y_a)
    X_a_norm = X_a / (patch_extent_x / 2)
    Y_a_norm = Y_a / (patch_extent_y / 2)
    
    texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
    texture_type_1 = random.choice(texture_types)
    texture_type_2 = random.choice(texture_types)
    texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
    texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
    
    use_too_high_curvature = random.random() < 0.3
    curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
    curvature_type_1 = random.choice(curvature_types)
    curvature_type_2 = random.choice(curvature_types)
    curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    
    # Small angles (within good patch range) - sheets are close but not necessarily parallel
    angle1_x = random.uniform(-5, 5)
    angle1_y = random.uniform(-5, 5)
    angle2_x = random.uniform(-5, 5)
    angle2_y = random.uniform(-5, 5)
    
    # Apply bent sheet augmentation (80% chance - main augmentation)
    use_bend = random.random() < 0.8
    
    if use_bend:
        # Choose which sheet(s) to bend (both can be bent, or just one)
        bend_sheet_1 = random.random() < 0.8  # 80% chance sheet 1 is bent
        bend_sheet_2 = random.random() < 0.8  # 80% chance sheet 2 is bent
        
        # Choose bend axis and direction
        bend_axis = random.choice(['x', 'y'])
        bend_from_left = random.random() > 0.5
        
        # Bend: 30-50% of sheet is bent (more than slight bends)
        transition_point = random.uniform(0.5, 0.7)  # 50-70% parallel, 30-50% bent
        
        # Create transition mask for bend
        if bend_axis == 'x':
            if bend_from_left:
                transition_mask_1 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                transition_mask_2 = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
            else:
                transition_mask_1 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(X_a_norm)
                transition_mask_2 = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(X_a_norm)
        else:
            if bend_from_left:
                transition_mask_1 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                transition_mask_2 = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
            else:
                transition_mask_1 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_1 else np.ones_like(Y_a_norm)
                transition_mask_2 = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1) if bend_sheet_2 else np.ones_like(Y_a_norm)
        
        # Apply bends: add 15-40 degrees (moderate to strong bends)
        if bend_sheet_1:
            bent_angle1_x = angle1_x + random.choice([-1, 1]) * random.uniform(15, 40)
            bent_angle1_y = angle1_y + random.choice([-1, 1]) * random.uniform(10, 30)
            # Interpolate angles based on transition mask
            angle1_x_map = angle1_x + (bent_angle1_x - angle1_x) * (1.0 - transition_mask_1)
            angle1_y_map = angle1_y + (bent_angle1_y - angle1_y) * (1.0 - transition_mask_1)
        else:
            angle1_x_map = np.full_like(X_a_norm, angle1_x)
            angle1_y_map = np.full_like(Y_a_norm, angle1_y)
        
        if bend_sheet_2:
            bent_angle2_x = angle2_x + random.choice([-1, 1]) * random.uniform(15, 40)
            bent_angle2_y = angle2_y + random.choice([-1, 1]) * random.uniform(10, 30)
            # Interpolate angles based on transition mask
            angle2_x_map = angle2_x + (bent_angle2_x - angle2_x) * (1.0 - transition_mask_2)
            angle2_y_map = angle2_y + (bent_angle2_y - angle2_y) * (1.0 - transition_mask_2)
        else:
            angle2_x_map = np.full_like(X_a_norm, angle2_x)
            angle2_y_map = np.full_like(Y_a_norm, angle2_y)
        
        # Convert to radians for tilt calculation
        angle1_x_rad_map = np.radians(angle1_x_map)
        angle1_y_rad_map = np.radians(angle1_y_map)
        angle2_x_rad_map = np.radians(angle2_x_map)
        angle2_y_rad_map = np.radians(angle2_y_map)
        
        # Apply tilt effects with varying angles (bent sheets)
        tilt_amplifier = 1.0
        tilt_factor1 = (np.tan(angle1_x_rad_map) * Y_a + np.tan(angle1_y_rad_map) * X_a) * tilt_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad_map) * Y_a + np.tan(angle2_y_rad_map) * X_a) * tilt_amplifier
    else:
        # No bend - use uniform angles
        angle1_x_rad = np.radians(angle1_x)
        angle1_y_rad = np.radians(angle1_y)
        angle2_x_rad = np.radians(angle2_x)
        angle2_y_rad = np.radians(angle2_y)
        
        tilt_amplifier = 1.0
        tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
        tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
    
    z_a_base = tilt_factor1 + texture1 + curvature1
    z_b_base = tilt_factor2 + texture2 + curvature2
    
    x_a_final = X_a + center_a_x
    y_a_final = Y_a + center_a_y
    z_a_final = z_a_base + center_a_z
    
    x_b_final = X_a + center_b_x
    y_b_final = Y_a + center_b_y
    z_b_final = z_b_base + center_b_z
    
    xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # For too close sheets, only check for actual intersection without enforcing large separation
    # Use a very small minimum separation (1mm) to allow sheets to be very close
    center_b_x_current = center_b_x
    center_b_y_current = center_b_y
    
    # Check if sheets actually intersect (not just close)
    if check_sheets_intersect(xyz_a, xyz_b):
        # If they intersect, make a tiny adjustment (just enough to prevent intersection)
        # Add a very small offset (1-2mm) in a random direction
        tiny_offset = random.uniform(1000, 2000)  # 1-2mm
        offset_direction = random.uniform(0, 2 * np.pi)
        center_b_x = center_a_x + np.cos(offset_direction) * tiny_offset
        center_b_y = center_a_y + np.sin(offset_direction) * tiny_offset
        
        # Recalculate with new offsets
        x_b_final = X_a + center_b_x
        y_b_final = Y_a + center_b_y
        z_b_final = z_b_base + center_b_z
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # Pivot both sheets together as a unit to add variation
    xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
    
    # Centralize the centroid of the pair to origin
    xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
    
    # Prepare metadata with bend information
    bend_info = {}
    if 'use_bend' in locals() and use_bend:
        bend_info = {
            "bent_sheets": True,
            "bend_sheet_1": bend_sheet_1 if 'bend_sheet_1' in locals() else False,
            "bend_sheet_2": bend_sheet_2 if 'bend_sheet_2' in locals() else False,
            "bend_axis": bend_axis if 'bend_axis' in locals() else None
        }
    else:
        bend_info = {"bent_sheets": False}
    
    metadata = {
        "category": "too_close_sheets",
        **bend_info,
        "texture_1": texture_type_1,
        "texture_2": texture_type_2,
        "curvature_1": curvature_type_1,
        "curvature_2": curvature_type_2
    }
    
    return (xyz_a, xyz_b, metadata)

def _generate_multi_violation_pair(patch_size):
    """Generate a single multi-violation pair. Returns (xyz_a, xyz_b, metadata) or None."""
    patch_a_x_range = (-4000, 4000)
    patch_a_y_range = (-4000, 4000)
    patch_a_z_range = (30000, 65000)
    normal_z_separation_range = (30000, 65000)
    excessive_xy_separation_range = (5000, 150000)
    # Ensure no elongation - keep aspect ratio close to 1
    base_extent = random.uniform(28000, 30000)
    patch_extent_x = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    patch_extent_y = base_extent * random.uniform(0.9, 1.1)  # Within 10% of base
    
    # Only excessive XY separation violation now
    violation_type = "too_much_xy_sep"
    
    center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
    center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
    center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
    
    z_sep = random.uniform(normal_z_separation_range[0], normal_z_separation_range[1])
    center_b_z = center_a_z - z_sep
    
    if "too_much_xy_sep" in violation_type:
        excessive_xy_sep = random.uniform(excessive_xy_separation_range[0], excessive_xy_separation_range[1])
        offset_direction = random.uniform(0, 2 * np.pi)
        xy_offset_x = np.cos(offset_direction) * excessive_xy_sep
        xy_offset_y = np.sin(offset_direction) * excessive_xy_sep
    else:
        xy_offset_x = 0
        xy_offset_y = 0
    
    center_b_x = center_a_x + xy_offset_x
    center_b_y = center_a_y + xy_offset_y
    
    x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
    y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
    X_a, Y_a = np.meshgrid(x_a, y_a)
    X_a_norm = X_a / (patch_extent_x / 2)
    Y_a_norm = Y_a / (patch_extent_y / 2)
    
    texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
    texture_type_1 = random.choice(texture_types)
    texture_type_2 = random.choice(texture_types)
    texture1 = generate_texture_mm(texture_type_1, X_a_norm, Y_a_norm, patch_size, extreme=False)
    texture2 = generate_texture_mm(texture_type_2, X_a_norm, Y_a_norm, patch_size, extreme=False)
    
    use_too_high_curvature = random.random() < 0.3
    curvature_types = ["convex", "concave", "mixed", "strong_convex", "strong_concave", "opposite_sphere", "none"]
    curvature_type_1 = random.choice(curvature_types)
    curvature_type_2 = random.choice(curvature_types)
    curvature1 = generate_curvature_mm(curvature_type_1, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    curvature2 = generate_curvature_mm(curvature_type_2, X_a_norm, Y_a_norm, patch_size, too_high=use_too_high_curvature)
    
    # Multi-violation: Combine excessive XY separation with bent sheets
    # Choose which sheet will be bent (sheet 1 or sheet 2)
    bent_sheet = random.choice([1, 2])
    
    # Generate base angles for the flat sheet (small angles, relatively parallel)
    if bent_sheet == 1:
        # Sheet 2 is flat, sheet 1 will be bent
        base_angle2_x = random.uniform(-5, 5)
        base_angle2_y = random.uniform(-5, 5)
        base_angle1_x = random.uniform(-5, 5)  # Starting angle for sheet 1
        base_angle1_y = random.uniform(-5, 5)
    else:
        # Sheet 1 is flat, sheet 2 will be bent
        base_angle1_x = random.uniform(-5, 5)
        base_angle1_y = random.uniform(-5, 5)
        base_angle2_x = random.uniform(-5, 5)  # Starting angle for sheet 2
        base_angle2_y = random.uniform(-5, 5)
    
    # Calculate the normal vector of the flat sheet to determine perpendicular direction
    def calculate_plane_normal(angle_x_rad, angle_y_rad):
        """Calculate normal vector for a plane tilted by angle_x and angle_y."""
        normal = np.array([
            np.tan(angle_y_rad),
            np.tan(angle_x_rad),
            1.0
        ])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        return normal
    
    if bent_sheet == 1:
        # Sheet 2 is flat - calculate its normal
        angle2_x_rad_flat = np.radians(base_angle2_x)
        angle2_y_rad_flat = np.radians(base_angle2_y)
        flat_normal = calculate_plane_normal(angle2_x_rad_flat, angle2_y_rad_flat)
    else:
        # Sheet 1 is flat - calculate its normal
        angle1_x_rad_flat = np.radians(base_angle1_x)
        angle1_y_rad_flat = np.radians(base_angle1_y)
        flat_normal = calculate_plane_normal(angle1_x_rad_flat, angle1_y_rad_flat)
    
    # Choose bend axis and direction (towards or away from the other sheet)
    bend_axis = random.choice(['x', 'y'])
    bend_direction = random.choice(['towards', 'away'])  # Towards or away from the other sheet
    # Use smaller transition_point to ensure 70%+ bent (0.15 means 85% bent, 0.25 means 75% bent)
    transition_point = random.uniform(0.15, 0.25)  # 15-25% parallel, rest bent (ensures 70%+ bent)
    
    # Determine bend direction along axis
    bend_from_left = random.random() > 0.5
    
    if bend_axis == 'x':
        # Bend along X axis
        if bend_from_left:
            # Bend from left (small X) to right (large X)
            transition_mask = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
        else:
            # Bend from right (large X) to left (small X)
            transition_mask = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1)
    else:
        # Bend along Y axis
        if bend_from_left:
            # Bend from bottom (small Y) to top (large Y)
            transition_mask = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
        else:
            # Bend from top (large Y) to bottom (small Y)
            transition_mask = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1)
    
    # Ensure at least 70% is bent
    bent_fraction = np.mean(transition_mask < 1.0)
    if bent_fraction < 0.7:
        transition_point = 0.15  # Ensure 85% bent
        if bend_axis == 'x':
            if bend_from_left:
                transition_mask = np.clip((X_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
            else:
                transition_mask = np.clip((1.0 - X_a_norm) / (2.0 * transition_point), 0, 1)
        else:
            if bend_from_left:
                transition_mask = np.clip((Y_a_norm + 1.0) / (2.0 * transition_point), 0, 1)
            else:
                transition_mask = np.clip((1.0 - Y_a_norm) / (2.0 * transition_point), 0, 1)
    
    # Find the bend point location (where transition_mask transitions from 1 to <1)
    # Center the patch at the bend point OR on the bent part where it's perpendicular/extreme angle
    center_on_bend_line = random.random() < 0.5  # 50% chance to center on bend line, 50% on bent part
    
    if center_on_bend_line:
        # Center at the bend line (transition point)
        if bend_axis == 'x':
            # Find X coordinate where bend occurs
            if bend_from_left:
                bend_x_norm = -1.0 + 2.0 * transition_point  # X coordinate of bend
            else:
                bend_x_norm = 1.0 - 2.0 * transition_point
            bend_x = bend_x_norm * (patch_extent_x / 2)
            # Adjust center to be at bend point
            center_a_x = center_a_x - bend_x
            center_b_x = center_b_x - bend_x
        else:
            # Find Y coordinate where bend occurs
            if bend_from_left:
                bend_y_norm = -1.0 + 2.0 * transition_point
            else:
                bend_y_norm = 1.0 - 2.0 * transition_point
            bend_y = bend_y_norm * (patch_extent_y / 2)
            # Adjust center to be at bend point
            center_a_y = center_a_y - bend_y
            center_b_y = center_b_y - bend_y
    else:
        # Center on the bent part where points are perpendicular/extreme angle
        # Find a location in the bent region (where transition_mask < 1.0)
        # Choose a point in the middle-to-end of the bent portion
        bent_region_fraction = random.uniform(0.3, 0.8)  # 30-80% into the bent region
        
        if bend_axis == 'x':
            if bend_from_left:
                # Bent region is from transition_point to 1.0
                bent_x_norm = -1.0 + 2.0 * (transition_point + (1.0 - transition_point) * bent_region_fraction)
            else:
                # Bent region is from -1.0 to (1.0 - 2*transition_point)
                bent_x_norm = -1.0 + (1.0 - 2.0 * transition_point) * (1.0 - bent_region_fraction)
            bent_x = bent_x_norm * (patch_extent_x / 2)
            center_a_x = center_a_x - bent_x
            center_b_x = center_b_x - bent_x
        else:
            if bend_from_left:
                bent_y_norm = -1.0 + 2.0 * (transition_point + (1.0 - transition_point) * bent_region_fraction)
            else:
                bent_y_norm = -1.0 + (1.0 - 2.0 * transition_point) * (1.0 - bent_region_fraction)
            bent_y = bent_y_norm * (patch_extent_y / 2)
            center_a_y = center_a_y - bent_y
            center_b_y = center_b_y - bent_y
    
    # Generate bent angles that are perpendicular to the flat sheet
    # Find angles that create a normal vector perpendicular to flat_normal
    # Try different angle combinations to get ~90 degrees from flat sheet
    if bent_sheet == 1:
        # Sheet 1 will be bent perpendicular to sheet 2
        # Try to find angles that make sheet 1 perpendicular to sheet 2
        for attempt in range(20):
            # Generate candidate bent angles
            bent_angle1_x = base_angle1_x + random.choice([-1, 1]) * random.uniform(70, 90)
            bent_angle1_y = base_angle1_y + random.choice([-1, 1]) * random.uniform(0, 20)
            
            bent_angle1_x_rad = np.radians(bent_angle1_x)
            bent_angle1_y_rad = np.radians(bent_angle1_y)
            bent_normal = calculate_plane_normal(bent_angle1_x_rad, bent_angle1_y_rad)
            
            # Check if bent normal is approximately perpendicular to flat normal
            dot_product = np.clip(np.dot(bent_normal, flat_normal), -1.0, 1.0)
            angle_between = np.degrees(np.arccos(abs(dot_product)))
            
            # We want angle between normals to be close to 90 degrees (perpendicular)
            if 75 <= angle_between <= 105:
                break
        
        # Adjust bent angle direction based on whether it should go towards or away
        # If towards, the bent portion should point in the direction of the other sheet
        # If away, it should point away from the other sheet
        # This is controlled by the sign of the angle change
        if bend_direction == 'away':
            # Reverse the direction to point away
            bent_angle1_x = base_angle1_x - (bent_angle1_x - base_angle1_x)
        
        # Interpolate angles for sheet 1 based on transition mask
        angle1_x_map = base_angle1_x + (bent_angle1_x - base_angle1_x) * (1.0 - transition_mask)
        angle1_y_map = base_angle1_y + (bent_angle1_y - base_angle1_y) * (1.0 - transition_mask)
        # Sheet 2 stays flat
        angle2_x_map = np.full_like(X_a_norm, base_angle2_x)
        angle2_y_map = np.full_like(Y_a_norm, base_angle2_y)
    else:
        # Sheet 2 will be bent perpendicular to sheet 1
        for attempt in range(20):
            bent_angle2_x = base_angle2_x + random.choice([-1, 1]) * random.uniform(70, 90)
            bent_angle2_y = base_angle2_y + random.choice([-1, 1]) * random.uniform(0, 20)
            
            bent_angle2_x_rad = np.radians(bent_angle2_x)
            bent_angle2_y_rad = np.radians(bent_angle2_y)
            bent_normal = calculate_plane_normal(bent_angle2_x_rad, bent_angle2_y_rad)
            
            dot_product = np.clip(np.dot(bent_normal, flat_normal), -1.0, 1.0)
            angle_between = np.degrees(np.arccos(abs(dot_product)))
            
            if 75 <= angle_between <= 105:
                break
        
        # Adjust for direction
        if bend_direction == 'away':
            bent_angle2_x = base_angle2_x - (bent_angle2_x - base_angle2_x)
        
        # Sheet 1 stays flat
        angle1_x_map = np.full_like(X_a_norm, base_angle1_x)
        angle1_y_map = np.full_like(Y_a_norm, base_angle1_y)
        # Interpolate angles for sheet 2 based on transition mask
        angle2_x_map = base_angle2_x + (bent_angle2_x - base_angle2_x) * (1.0 - transition_mask)
        angle2_y_map = base_angle2_y + (bent_angle2_y - base_angle2_y) * (1.0 - transition_mask)
    
    # Convert to radians
    angle1_x_rad = np.radians(angle1_x_map)
    angle1_y_rad = np.radians(angle1_y_map)
    angle2_x_rad = np.radians(angle2_x_map)
    angle2_y_rad = np.radians(angle2_y_map)
    
    # Apply tilt effects (scaled to mm) - varies across the sheet
    tilt_amplifier = 1.0
    tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
    tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
    
    # Create base Z coordinates for patch A (relative to center, in mm)
    z_a_base = tilt_factor1 + texture1 + curvature1
    
    # Final coordinates for patch A (add center offsets)
    x_a_final = X_a + center_a_x
    y_a_final = Y_a + center_a_y
    z_a_final = z_a_base + center_a_z
    
    # Create base Z coordinates for patch B (relative to center, in mm)
    z_b_base = tilt_factor2 + texture2 + curvature2
    
    # Final coordinates for patch B (add center offsets and XY separation)
    x_b_final = X_a + center_b_x
    y_b_final = Y_a + center_b_y
    z_b_final = z_b_base + center_b_z
    
    # Ensure bent sheet points do not go past the other sheet
    # Check if bent sheet (sheet 1 or 2) has points that extend past the flat sheet
    if bent_sheet == 1:
        # Sheet 1 is bent, sheet 2 is flat
        flat_sheet_z = z_b_final  # Sheet 2 is flat
        bent_sheet_z = z_a_final  # Sheet 1 is bent
        
        # Calculate the minimum Z of the flat sheet for reference
        flat_sheet_min_z = np.min(flat_sheet_z)
        
        # If bending towards, bent sheet should not go past flat sheet
        # If bending away, bent sheet should not go too far past
        if bend_direction == 'towards':
            # Bent sheet should not go past flat sheet
            # Clip bent sheet Z to not exceed flat sheet min Z (with some margin)
            max_allowed_z = flat_sheet_min_z - 100.0  # 100mm margin
            z_a_final = np.clip(z_a_final, z_a_final.min(), max_allowed_z)
        else:
            # Bent sheet bending away - ensure it doesn't go too far
            # Limit the maximum Z difference to keep sheets close
            max_z_diff = z_sep * 1.2  # Don't go more than 1.2x the separation
            z_a_final = np.clip(z_a_final, z_a_final.min(), center_a_z + max_z_diff)
    else:
        # Sheet 2 is bent, sheet 1 is flat
        flat_sheet_z = z_a_final  # Sheet 1 is flat
        bent_sheet_z = z_b_final  # Sheet 2 is bent
        
        flat_sheet_min_z = np.min(flat_sheet_z)
        
        if bend_direction == 'towards':
            # Bent sheet should not go past flat sheet
            max_allowed_z = flat_sheet_min_z - 100.0  # 100mm margin
            z_b_final = np.clip(z_b_final, z_b_final.min(), max_allowed_z)
        else:
            # Bent sheet bending away
            max_z_diff = z_sep * 1.2
            z_b_final = np.clip(z_b_final, z_b_final.min(), center_b_z + max_z_diff)
    
    xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
    xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
    
    # Pivot both sheets together as a unit to add variation
    xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
    
    # Centralize the centroid of the pair to origin
    xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
    
    metadata = {
        "category": "multi_violation",
        "violation_type": violation_type,
        "bent_sheet": bent_sheet,
        "bend_axis": bend_axis,
        "bend_direction": bend_direction,
        "texture_1": texture_type_1,
        "texture_2": texture_type_2,
        "curvature_1": curvature_type_1,
        "curvature_2": curvature_type_2
    }
    
    return (xyz_a, xyz_b, metadata)

def main():
    """Main function to create and visualize depth maps."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create depth maps for parallel planes')
    parser.add_argument('--multiple', action='store_true',
                       help='Create multiple depth map pairs with varying separations (2cm-8cm)')
    parser.add_argument('--textured', action='store_true',
                       help='Create 300 textured depth map pairs with varying separations and textures')
    parser.add_argument('--curved', action='store_true',
                       help='Create 300 curved depth map pairs with varying separations and curvatures')
    parser.add_argument('--angled', action='store_true',
                       help='Create 200 angled depth map pairs with varying separations and angles')
    parser.add_argument('--bad', action='store_true',
                       help='Create BAD depth pairs (all 5 categories with regular texture)')
    parser.add_argument('--bad-same-plane', action='store_true',
                       help='Create BAD: same plane pairs')
    parser.add_argument('--bad-large-angle', action='store_true',
                       help='Create BAD: large angle pairs')
    parser.add_argument('--bad-too-much-xy-sep', action='store_true',
                       help='Create BAD: too much XY separation pairs (excessive spatial separation)')
    parser.add_argument('--bad-multi', action='store_true',
                       help='Create BAD: multi-violation pairs')
    parser.add_argument('--num-pairs', type=int, default=100,
                       help='Number of patch pairs to generate per category (default: 100)')
    parser.add_argument('--separation', type=float, default=2.0,
                       help='Separation between planes in cm (default: 2.0)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization (only create depth maps)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.bad:
        # Create ALL bad categories in random order, saved directly under bad_patches/
        print("\n" + "="*70)
        print("CREATING BAD DEPTH PAIRS - ALL CATEGORIES IN RANDOM ORDER")
        print("="*70 + "\n")
        create_all_bad_patches_random_order(num_pairs_per_category=args.num_pairs)
        print("\n" + "="*70)
        print("ALL BAD DEPTH PAIRS CREATED SUCCESSFULLY!")
        print(f"Total: {args.num_pairs * 5} pairs across 5 categories (saved in random order)")
        print("="*70 + "\n")
    elif args.bad_same_plane:
        create_bad_same_plane_pairs(num_pairs=args.num_pairs)
    elif args.bad_large_angle:
        create_bad_large_angle_pairs(num_pairs=args.num_pairs)
    elif args.bad_too_much_xy_sep:
        create_bad_too_much_xy_separation_pairs(num_pairs=args.num_pairs)
    elif args.bad_multi:
        create_bad_multi_violation_pairs(num_pairs=args.num_pairs)
    elif args.angled:
        # Create angled depth map pairs
        create_angled_depth_pairs()
    elif args.curved:
        # Create curved depth map pairs
        create_curved_depth_pairs()
    elif args.textured:
        # Create textured depth map pairs
        create_textured_depth_pairs()
    elif args.multiple:
        # Create multiple depth map pairs
        create_multiple_depth_pairs()
    else:
        # Create single depth map pair
        sep_m = args.separation / 100.0  # Convert cm to meters
        print(f"Creating depth maps for parallel planes {args.separation}cm apart...")
        
        # Create depth maps with random rotations
        depth1, depth2 = create_depth_maps(plane_separation=sep_m)
        
        # Print rotation information
        print("\n" + "="*60)
        print("DEPTH SHEET ROTATION INFORMATION")
        print("="*60)
        print("Both depth sheets are rotated around different axes:")
        print("  - X-axis rotation: Tilts the sheet up/down")
        print("  - Y-axis rotation: Tilts the sheet left/right") 
        print("  - Z-axis rotation: Rotates the sheet in-plane")
        print("="*60)
        print("This creates more realistic, non-parallel depth sheets")
        print("that better represent real-world objects and surfaces.")
        print("="*60)
        
        # Print statistics
        print_depth_stats(depth1, depth2)
        
        if not args.no_viz:
            # Visualize
            print("\nCreating visualization...")
            fig = visualize_depth_maps(depth1, depth2)
            
            # Save plots
            plt.savefig('depth_maps_visualization.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as 'depth_maps_visualization.png'")
            
            # Show plot
            plt.show()
        
        # Save depth maps
        save_depth_maps(depth1, depth2)

if __name__ == "__main__":
    main()
