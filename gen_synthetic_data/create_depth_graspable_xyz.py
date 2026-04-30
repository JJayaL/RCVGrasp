#!/usr/bin/env python3
"""
Create two sheets as XYZ point clouds representing parallel planes 3cm apart.
One plane is visible to camera 1, the other to camera 2.
The second plane's coordinates are transformed to camera 1's coordinate system.

Point clouds are generated using camera intrinsics from camera.json
to ensure proper spatial extent for 32x32 point grids.
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

def create_depth_maps(plane_separation=0.02, rotation_angles=None, xy_offset_range=(0.0, 0.015)):
    """
    Create two 32x32 sheets as XYZ point clouds with specified separation and rotation.
    Both planes can be rotated around different axes for more realistic appearance.
    
    Args:
        plane_separation: Distance between planes in meters
        rotation_angles: Dict with rotation angles for each plane {'plane1': {'x': 0, 'y': 0, 'z': 0}, 'plane2': {'x': 0, 'y': 0, 'z': 0}}
        xy_offset_range: Tuple (min, max) for X/Y spatial offset in meters (default: 0-3cm for good pairs)
    
    Returns:
        xyz1: Nx3 array of XYZ coordinates for sheet 1
        xyz2: Nx3 array of XYZ coordinates for sheet 2
        depth1: Depth map for plane 1 (camera 1 view) - for backwards compatibility
        depth2: Depth map for plane 2 (camera 2 view, transformed to camera 1 coords) - for backwards compatibility
        X: 2D meshgrid of X coordinates
        Y: 2D meshgrid of Y coordinates
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
    
    # Add small X/Y spatial offset for good pairs (optional augmentation)
    xy_offset_x = 0.0
    xy_offset_y = 0.0
    if xy_offset_range[1] > 0.0:  # If offset range is specified
        xy_offset_x = random.uniform(xy_offset_range[0], xy_offset_range[1]) * random.choice([-1, 1])
        xy_offset_y = random.uniform(xy_offset_range[0], xy_offset_range[1]) * random.choice([-1, 1])
    
    # Convert depth maps to XYZ point clouds
    xyz1 = depth_to_xyz(depth1, X, Y)
    # Apply X/Y offset to second sheet
    X2_offset = X + xy_offset_x
    Y2_offset = Y + xy_offset_y
    xyz2 = depth_to_xyz(depth2, X2_offset, Y2_offset)
    
    if xy_offset_x != 0.0 or xy_offset_y != 0.0:
        print(f"XY offset applied: X={xy_offset_x*100:.2f}cm, Y={xy_offset_y*100:.2f}cm")
    
    print(f"Sheet 1 XYZ coordinates: {xyz1.shape} points")
    print(f"Sheet 2 XYZ coordinates: {xyz2.shape} points")
    
    return xyz1, xyz2, depth1, depth2, X, Y

def create_textured_depth_maps(plane_separation=0.02, texture_type="random", rotation_angles=None, xy_offset_range=(0.0, 0.03)):
    """
    Create two 32x32 sheets as XYZ point clouds with textures.
    
    Args:
        plane_separation: Distance between planes in meters
        texture_type: Type of texture ("random", "wave", "noise", "grid", "spots")
        xy_offset_range: Tuple (min, max) for X/Y spatial offset in meters (default: 0-1.5cm for good pairs)
    
    Returns:
        xyz1: Nx3 array of XYZ coordinates for sheet 1
        xyz2: Nx3 array of XYZ coordinates for sheet 2
        depth1: Depth map for plane 1 with texture - for backwards compatibility
        depth2: Depth map for plane 2 with texture - for backwards compatibility
        X: 2D meshgrid of X coordinates
        Y: 2D meshgrid of Y coordinates
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
    
    # Add small X/Y spatial offset for good pairs (optional augmentation)
    xy_offset_x = 0.0
    xy_offset_y = 0.0
    if xy_offset_range[1] > 0.0:  # If offset range is specified
        xy_offset_x = random.uniform(xy_offset_range[0], xy_offset_range[1]) * random.choice([-1, 1])
        xy_offset_y = random.uniform(xy_offset_range[0], xy_offset_range[1]) * random.choice([-1, 1])
    
    # Convert depth maps to XYZ point clouds
    xyz1 = depth_to_xyz(depth1, X, Y)
    # Apply X/Y offset to second sheet
    X2_offset = X + xy_offset_x
    Y2_offset = Y + xy_offset_y
    xyz2 = depth_to_xyz(depth2, X2_offset, Y2_offset)
    
    print(f"Textured Sheet 1 XYZ coordinates: {xyz1.shape} points")
    print(f"Textured Sheet 2 XYZ coordinates: {xyz2.shape} points")
    
    return xyz1, xyz2, depth1, depth2, X, Y

def create_curved_depth_maps(plane_separation=0.02, curvature_type="random", rotation_angles=None, xy_offset_range=(0.0, 0.03)):
    """
    Create two 32x32 sheets as XYZ point clouds with curvature.
    
    Args:
        plane_separation: Distance between planes in meters
        curvature_type: Type of curvature ("random", "convex", "concave", "wave", "saddle")
        xy_offset_range: Tuple (min, max) for X/Y spatial offset in meters (default: 0-1.5cm for good pairs)
    
    Returns:
        xyz1: Nx3 array of XYZ coordinates for sheet 1
        xyz2: Nx3 array of XYZ coordinates for sheet 2
        depth1: Depth map for plane 1 with curvature - for backwards compatibility
        depth2: Depth map for plane 2 with curvature - for backwards compatibility
        X: 2D meshgrid of X coordinates
        Y: 2D meshgrid of Y coordinates
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
    
    # Add small X/Y spatial offset for good pairs (optional augmentation)
    xy_offset_x = 0.0
    xy_offset_y = 0.0
    if xy_offset_range[1] > 0.0:  # If offset range is specified
        xy_offset_x = random.uniform(xy_offset_range[0], xy_offset_range[1]) * random.choice([-1, 1])
        xy_offset_y = random.uniform(xy_offset_range[0], xy_offset_range[1]) * random.choice([-1, 1])
    
    # Convert depth maps to XYZ point clouds
    xyz1 = depth_to_xyz(depth1, X, Y)
    # Apply X/Y offset to second sheet
    X2_offset = X + xy_offset_x
    Y2_offset = Y + xy_offset_y
    xyz2 = depth_to_xyz(depth2, X2_offset, Y2_offset)
    
    print(f"Curved Sheet 1 XYZ coordinates: {xyz1.shape} points")
    print(f"Curved Sheet 2 XYZ coordinates: {xyz2.shape} points")
    
    return xyz1, xyz2, depth1, depth2, X, Y

def create_multiple_depth_pairs():
    """
    Create multiple XYZ sheet pairs with varying separations from 2cm to 8cm.
    Saves them in organized folders with clean naming.
    """
    import os
    
    # Create main output directory
    output_dir = "depth_map_pairs"
    no_tex_dir = os.path.join(output_dir, "no_tex")
    os.makedirs(no_tex_dir, exist_ok=True)
    
    # Create PNG, NPY, and XYZ subdirectories
    png_dir = os.path.join(no_tex_dir, "depth_png")
    npy_dir = os.path.join(no_tex_dir, "depth_npy")
    xyz_dir = os.path.join(no_tex_dir, "xyz_coords")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)
    
    # Define separations from 2cm to 8cm with 100 different values
    separations_cm = np.linspace(2, 8, 300)
    
    print(f"Creating 300 clean XYZ sheet pairs with separations from 2cm to 8cm")
    print(f"Systematic rotation across all 360 degrees for each axis")
    print(f"Clean rotation-only sheets (no textures, no surface variations)")
    print(f"Output directory: {no_tex_dir}/")
    print(f"PNG files: {png_dir}/")
    print(f"NPY files: {npy_dir}/")
    print(f"XYZ files: {xyz_dir}/")
    
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
        
        # Create XYZ sheets with systematic rotation (no textures)
        # Add small X/Y offset for good pairs (0.5-3cm range)
        xy_offset_range = (0.005, 0.015)  # 0.5cm to 1.5cm
        xyz1, xyz2, depth1, depth2, X, Y = create_depth_maps(plane_separation=sep_m, rotation_angles=rotation_angles, xy_offset_range=xy_offset_range)
        
        # Convert to 16-bit PNG (depth in millimeters)
        depth1_mm = (depth1 * 1000).astype(np.uint16)
        depth2_mm = (depth2 * 1000).astype(np.uint16)
        
        # Save PNG files (for backwards compatibility)
        cv2.imwrite(os.path.join(png_dir, f"depth{i:03d}_a.png"), depth1_mm)
        cv2.imwrite(os.path.join(png_dir, f"depth{i:03d}_b.png"), depth2_mm)
        
        # Save NPY depth files (for backwards compatibility)
        np.save(os.path.join(npy_dir, f"depth{i:03d}_a.npy"), depth1)
        np.save(os.path.join(npy_dir, f"depth{i:03d}_b.npy"), depth2)
        
        # Save XYZ coordinates
        np.save(os.path.join(xyz_dir, f"xyz{i:03d}_a.npy"), xyz1)
        np.save(os.path.join(xyz_dir, f"xyz{i:03d}_b.npy"), xyz2)
        np.savetxt(os.path.join(xyz_dir, f"xyz{i:03d}_a.txt"), xyz1, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(xyz_dir, f"xyz{i:03d}_b.txt"), xyz2, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Print progress every 10 pairs
        if i % 10 == 0 or i == 1:
            print(f"Progress: {i}/300 - Separation {sep_cm:.2f}cm (depth{i:03d})")
            print(f"  Sheet A: {xyz1.shape[0]} points, depth {depth1.min():.3f}m")
            print(f"  Sheet B: {xyz2.shape[0]} points, depth {depth2.min():.3f}m")
            print(f"  Actual separation: {(depth2.min() - depth1.min())*100:.2f}cm")
    
    print(f"\nAll 300 non-textured XYZ sheet pairs created successfully!")
    print(f"Total pairs: {len(separations_cm)}")
    print(f"Separation range: 2.00cm - 8.00cm")
    print(f"Files saved in: {png_dir}/, {npy_dir}/, and {xyz_dir}/")
    print(f"File naming:")
    print(f"  Depth: depth001_a.png, depth001_b.png, ..., depth300_a.png, depth300_b.png")
    print(f"  XYZ: xyz001_a.npy, xyz001_b.npy, ..., xyz300_a.npy, xyz300_b.npy")
    print(f"  XYZ text: xyz001_a.txt, xyz001_b.txt, ..., xyz300_a.txt, xyz300_b.txt")

def create_textured_depth_pairs():
    """
    Create 300 textured XYZ sheet pairs with varying separations and textures.
    Saves them in depth_tex folder.
    """
    import os
    
    # Create main output directory
    output_dir = "depth_map_pairs"
    tex_dir = os.path.join(output_dir, "depth_tex")
    os.makedirs(tex_dir, exist_ok=True)
    
    # Create PNG, NPY, and XYZ subdirectories
    png_dir = os.path.join(tex_dir, "depth_png")
    npy_dir = os.path.join(tex_dir, "depth_npy")
    xyz_dir = os.path.join(tex_dir, "xyz_coords")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)
    
    # Define texture types
    texture_types = ["random", "wave", "noise", "grid", "spots"]
    
    # Define separations from 2cm to 8cm
    separations_cm = np.linspace(2, 8, 100)
    
    print(f"Creating 300 textured XYZ sheet pairs")
    print(f"Output directory: {tex_dir}/")
    print(f"PNG files: {png_dir}/")
    print(f"NPY files: {npy_dir}/")
    print(f"XYZ files: {xyz_dir}/")
    print(f"Texture types: {texture_types}")
    
    pair_count = 0
    for i in range(500):
        # Random separation (2-8cm)
        sep_cm = random.uniform(2, 8)
        sep_m = sep_cm / 100.0
        
        # Random texture type
        texture_type = random.choice(texture_types)
        
        # Create textured XYZ sheets with equal rotation
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
        
        # Add small X/Y offset for good pairs (0.5-3cm range)
        xy_offset_range = (0.005, 0.015)  # 0.5cm to 1.5cm
        xyz1, xyz2, depth1, depth2, X, Y = create_textured_depth_maps(plane_separation=sep_m, texture_type=texture_type, rotation_angles=rotation_angles, xy_offset_range=xy_offset_range)
        
        # Convert to 16-bit PNG (depth in millimeters)
        depth1_mm = (depth1 * 1000).astype(np.uint16)
        depth2_mm = (depth2 * 1000).astype(np.uint16)
        
        # Save PNG files (for backwards compatibility)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_a.png"), depth1_mm)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_b.png"), depth2_mm)
        
        # Save NPY depth files (for backwards compatibility)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_a.npy"), depth1)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_b.npy"), depth2)
        
        # Save XYZ coordinates
        np.save(os.path.join(xyz_dir, f"xyz{i+1:03d}_a.npy"), xyz1)
        np.save(os.path.join(xyz_dir, f"xyz{i+1:03d}_b.npy"), xyz2)
        np.savetxt(os.path.join(xyz_dir, f"xyz{i+1:03d}_a.txt"), xyz1, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(xyz_dir, f"xyz{i+1:03d}_b.txt"), xyz2, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Print progress every 50 pairs
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {i+1}/300 - Separation {sep_cm:.2f}cm, Texture: {texture_type}")
            print(f"  Sheet A: {xyz1.shape[0]} points, depth {depth1.min():.3f}m - {depth1.max():.3f}m")
            print(f"  Sheet B: {xyz2.shape[0]} points, depth {depth2.min():.3f}m - {depth2.max():.3f}m")
            print(f"  Actual separation: {(depth2.mean() - depth1.mean())*100:.2f}cm")
    
    print(f"\nAll 300 textured XYZ sheet pairs created successfully!")
    print(f"Total pairs: 300")
    print(f"Separation range: 2.00cm - 8.00cm")
    print(f"Texture types used: {texture_types}")
    print(f"Files saved in: {png_dir}/, {npy_dir}/, and {xyz_dir}/")
    print(f"File naming:")
    print(f"  Depth: depth001_a.png, depth001_b.png, ..., depth300_a.png, depth300_b.png")
    print(f"  XYZ: xyz001_a.npy, xyz001_b.npy, ..., xyz300_a.npy, xyz300_b.npy")
    print(f"  XYZ text: xyz001_a.txt, xyz001_b.txt, ..., xyz300_a.txt, xyz300_b.txt")

def create_angled_depth_maps(plane_separation=0.02, angle_type="random", texture_type="random", curvature_type="none", rotation_angles=None, xy_offset_range=(0.0, 0.015)):
    """
    Create two 32x32 sheets as XYZ point clouds for planes at slight angles to each other with texture and optional curvature.
    
    Args:
        plane_separation: Distance between planes in meters
        angle_type: Type of angle ("random", "tilt_x", "tilt_y", "tilt_both", "rotation")
        texture_type: Type of texture ("random", "wave", "noise", "grid", "spots")
        curvature_type: Type of curvature ("none", "convex", "concave", "mixed", "concave_diff_radius", "convex_diff_radius")
        xy_offset_range: Tuple (min, max) for X/Y spatial offset in meters (default: 0-3cm for good pairs)
    
    Returns:
        xyz1: Nx3 array of XYZ coordinates for sheet 1
        xyz2: Nx3 array of XYZ coordinates for sheet 2
        depth1: Depth map for plane 1 (facing camera) with texture - for backwards compatibility
        depth2: Depth map for plane 2 (at slight angle) with texture - for backwards compatibility
        X: 2D meshgrid of X coordinates
        Y: 2D meshgrid of Y coordinates
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
    if texture_type == "none" or texture_type == "no_texture":
        # No texture - clean spherical curvature only
        texture1 = np.zeros((patch_size, patch_size))
        texture2 = np.zeros((patch_size, patch_size))
    elif texture_type == "random":
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
    
    # Generate curvature patterns (like patches from a sphere)
    curvature1 = np.zeros((patch_size, patch_size))
    curvature2 = np.zeros((patch_size, patch_size))
    
    if curvature_type == "convex":
        # Both sheets convex (bulging outward) - like outer surface of sphere
        radius = random.uniform(0.015, 0.10)  # Expanded radius range: 1.5-10cm for more variation
        # Variable amplitude for more range: 0.35 to 0.50
        amplitude = random.uniform(0.35, 0.50)
        curvature1 = amplitude * (X**2 + Y**2) / radius
        curvature2 = amplitude * (X**2 + Y**2) / radius
    elif curvature_type == "concave":
        # Both sheets concave (curving inward) - like inner surface of sphere
        radius = random.uniform(0.015, 0.10)  # Expanded radius range: 1.5-10cm for more variation
        # Variable amplitude for more range: 0.35 to 0.50
        amplitude = random.uniform(0.35, 0.50)
        curvature1 = -amplitude * (X**2 + Y**2) / radius
        curvature2 = -amplitude * (X**2 + Y**2) / radius
    elif curvature_type == "mixed":
        # Mixed: one convex, one concave (like patches from opposite sides of sphere)
        radius1 = random.uniform(0.015, 0.10)  # Expanded radius range
        radius2 = random.uniform(0.015, 0.10)
        # Variable amplitudes for more variation
        amplitude1 = random.uniform(0.35, 0.50)
        amplitude2 = random.uniform(0.35, 0.50)
        # Randomly assign which is convex and which is concave
        if random.random() > 0.5:
            # Sheet 1 convex, Sheet 2 concave
            curvature1 = amplitude1 * (X**2 + Y**2) / radius1
            curvature2 = -amplitude2 * (X**2 + Y**2) / radius2
        else:
            # Sheet 1 concave, Sheet 2 convex
            curvature1 = -amplitude1 * (X**2 + Y**2) / radius1
            curvature2 = amplitude2 * (X**2 + Y**2) / radius2
    elif curvature_type == "concave_diff_radius":
        # Both sheets concave but with slightly different radii
        radius1 = random.uniform(0.015, 0.10)  # Expanded radius range
        # Second radius is 30-70% different from first for more variation
        radius_diff_factor = random.uniform(0.3, 0.7) if random.random() > 0.5 else random.uniform(1.3, 1.7)
        radius2 = radius1 * radius_diff_factor
        amplitude = random.uniform(0.35, 0.50)  # Variable amplitude
        curvature1 = -amplitude * (X**2 + Y**2) / radius1
        curvature2 = -amplitude * (X**2 + Y**2) / radius2
    elif curvature_type == "convex_diff_radius":
        # Both sheets convex but with slightly different radii
        radius1 = random.uniform(0.015, 0.10)  # Expanded radius range
        # Second radius is 30-70% different from first for more variation
        radius_diff_factor = random.uniform(0.3, 0.7) if random.random() > 0.5 else random.uniform(1.3, 1.7)
        radius2 = radius1 * radius_diff_factor
        amplitude = random.uniform(0.35, 0.50)  # Variable amplitude
        curvature1 = amplitude * (X**2 + Y**2) / radius1
        curvature2 = amplitude * (X**2 + Y**2) / radius2
    elif curvature_type == "strong_convex":
        # Strong convex curvature (tighter sphere, very pronounced)
        radius = random.uniform(0.01, 0.06)  # Expanded range: 1-6cm for more variation
        amplitude = random.uniform(0.45, 0.60)  # Variable amplitude: 0.45-0.60
        curvature1 = amplitude * (X**2 + Y**2) / radius
        curvature2 = amplitude * (X**2 + Y**2) / radius
    elif curvature_type == "strong_concave":
        # Strong concave curvature (tighter sphere, very pronounced)
        radius = random.uniform(0.01, 0.06)  # Expanded range: 1-6cm for more variation
        amplitude = random.uniform(0.45, 0.60)  # Variable amplitude: 0.45-0.60
        curvature1 = -amplitude * (X**2 + Y**2) / radius
        curvature2 = -amplitude * (X**2 + Y**2) / radius
    elif curvature_type == "weak_convex":
        # Weak convex curvature (larger sphere, but still visible)
        radius = random.uniform(0.05, 0.15)  # Expanded range: 5-15cm for more variation
        amplitude = random.uniform(0.25, 0.40)  # Variable amplitude: 0.25-0.40
        curvature1 = amplitude * (X**2 + Y**2) / radius
        curvature2 = amplitude * (X**2 + Y**2) / radius
    elif curvature_type == "weak_concave":
        # Weak concave curvature (larger sphere, but still visible)
        radius = random.uniform(0.05, 0.15)  # Expanded range: 5-15cm for more variation
        amplitude = random.uniform(0.25, 0.40)  # Variable amplitude: 0.25-0.40
        curvature1 = -amplitude * (X**2 + Y**2) / radius
        curvature2 = -amplitude * (X**2 + Y**2) / radius
    elif curvature_type == "opposite_sphere":
        # One sheet from one side of sphere, other from opposite side (strong contrast)
        radius = random.uniform(0.015, 0.08)  # Expanded range for more variation
        # Variable amplitudes for more variation
        amplitude1 = random.uniform(0.40, 0.55)
        amplitude2 = random.uniform(0.40, 0.55)
        # Strong contrast: one strongly convex, one strongly concave
        curvature1 = amplitude1 * (X**2 + Y**2) / radius
        curvature2 = -amplitude2 * (X**2 + Y**2) / radius
    # else: curvature_type == "none", keep curvature1 and curvature2 as zeros
    
    # Generate angle variations for good pairs: angle BETWEEN sheets should be 0-30 degrees
    # For good pairs, we want small angles between sheets (0-30 degrees max)
    max_angle_between_sheets = 30.0  # Maximum angle between sheets in degrees
    
    if angle_type == "random":
        # Random angles: angle between sheets should be 0-30 degrees
        # Generate base angle for plane 1
        angle1_x = random.uniform(-15, 15)  # Base tilt around X-axis for plane 1
        angle1_y = random.uniform(-15, 15)  # Base tilt around Y-axis for plane 1
        angle1_z = random.uniform(-15, 15)  # Base rotation around Z-axis for plane 1
        
        # Generate angle difference for plane 2 (0-30 degrees from plane 1)
        angle_diff_x = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_diff_y = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_diff_z = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        
        angle_x = angle1_x
        angle_y = angle1_y
        angle_z = angle1_z
        angle2_x = angle1_x + angle_diff_x
        angle2_y = angle1_y + angle_diff_y
        angle2_z = angle1_z + angle_diff_z
    elif angle_type == "tilt_x":
        # Tilt around X-axis only - angle between sheets 0-30 degrees
        angle1_x = random.uniform(-15, 15)
        angle_diff_x = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_x = angle1_x
        angle2_x = angle1_x + angle_diff_x
        angle_y = 0
        angle_z = 0
        angle2_y = 0
        angle2_z = 0
    elif angle_type == "tilt_y":
        # Tilt around Y-axis only - angle between sheets 0-30 degrees
        angle_x = 0
        angle_z = 0
        angle2_x = 0
        angle2_z = 0
        angle1_y = random.uniform(-15, 15)
        angle_diff_y = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_y = angle1_y
        angle2_y = angle1_y + angle_diff_y
    elif angle_type == "tilt_both":
        # Tilt around both X and Y axes - angle between sheets 0-30 degrees
        angle1_x = random.uniform(-15, 15)
        angle1_y = random.uniform(-15, 15)
        angle_diff_x = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_diff_y = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_x = angle1_x
        angle_y = angle1_y
        angle2_x = angle1_x + angle_diff_x
        angle2_y = angle1_y + angle_diff_y
        angle_z = 0
        angle2_z = 0
    elif angle_type == "rotation":
        # Rotation around Z-axis only - angle between sheets 0-30 degrees
        angle_x = 0
        angle_y = 0
        angle2_x = 0
        angle2_y = 0
        angle1_z = random.uniform(-15, 15)
        angle_diff_z = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_z = angle1_z
        angle2_z = angle1_z + angle_diff_z
    else:
        # Default to random - angle between sheets 0-30 degrees
        angle1_x = random.uniform(-15, 15)
        angle1_y = random.uniform(-15, 15)
        angle1_z = random.uniform(-15, 15)
        angle_diff_x = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_diff_y = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_diff_z = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
        angle_x = angle1_x
        angle_y = angle1_y
        angle_z = angle1_z
        angle2_x = angle1_x + angle_diff_x
        angle2_y = angle1_y + angle_diff_y
        angle2_z = angle1_z + angle_diff_z
    
    # Convert angles to radians
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)
    angle2_x_rad = np.radians(angle2_x)
    angle2_y_rad = np.radians(angle2_y)
    angle2_z_rad = np.radians(angle2_z)
    
    # Calculate the actual geometric angle between the two planes
    # For a plane tilted by angle_x (around X-axis) and angle_y (around Y-axis):
    # The normal vector is approximately: [sin(angle_y), sin(angle_x), cos(angle_x)*cos(angle_y)]
    # For small angles, this simplifies to: [angle_y, angle_x, 1] (normalized)
    def calculate_plane_normal(angle_x_rad, angle_y_rad):
        """Calculate normal vector for a plane tilted by angle_x and angle_y."""
        # Normal vector for a plane: direction perpendicular to the plane
        # For small angles: normal ≈ [tan(angle_y), tan(angle_x), 1]
        normal = np.array([
            np.tan(angle_y_rad),
            np.tan(angle_x_rad),
            1.0
        ])
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        return normal
    
    # Calculate normal vectors for both planes
    normal1 = calculate_plane_normal(angle_x_rad, angle_y_rad)
    normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
    
    # Calculate angle between normals (this is the actual angle between planes)
    dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
    angle_between_planes_rad = np.arccos(dot_product)
    angle_between_planes_deg = np.degrees(angle_between_planes_rad)
    
    # If angle is too large, reduce the angle difference
    # We need to ensure both the geometric angle AND individual axis differences are ≤30°
    max_iterations = 10
    for iteration in range(max_iterations):
        if angle_between_planes_deg <= max_angle_between_sheets:
            # Check individual axis differences
            angle_diff_x_deg = abs(angle2_x - angle_x)
            angle_diff_y_deg = abs(angle2_y - angle_y)
            
            # If individual axis differences are also ≤30°, we're done
            if angle_diff_x_deg <= max_angle_between_sheets and angle_diff_y_deg <= max_angle_between_sheets:
                break
            
            # If individual axis difference is too large, scale it down
            if angle_diff_x_deg > max_angle_between_sheets:
                scale_factor_x = max_angle_between_sheets / angle_diff_x_deg
                angle_diff_x = angle_diff_x * scale_factor_x
                angle2_x = angle1_x + angle_diff_x
                angle2_x_rad = np.radians(angle2_x)
            
            if angle_diff_y_deg > max_angle_between_sheets:
                scale_factor_y = max_angle_between_sheets / angle_diff_y_deg
                angle_diff_y = angle_diff_y * scale_factor_y
                angle2_y = angle1_y + angle_diff_y
                angle2_y_rad = np.radians(angle2_y)
            
            # Recalculate geometric angle after scaling individual axes
            normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            angle_between_planes_rad = np.arccos(dot_product)
            angle_between_planes_deg = np.degrees(angle_between_planes_rad)
            
            # If geometric angle is still too large, scale both axes proportionally
            if angle_between_planes_deg > max_angle_between_sheets:
                scale_factor = max_angle_between_sheets / angle_between_planes_deg
                angle_diff_x = angle_diff_x * scale_factor
                angle_diff_y = angle_diff_y * scale_factor
                
                # Recalculate angles
                angle2_x = angle1_x + angle_diff_x
                angle2_y = angle1_y + angle_diff_y
                angle2_x_rad = np.radians(angle2_x)
                angle2_y_rad = np.radians(angle2_y)
                
                # Recalculate normal and angle
                normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
                dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
                angle_between_planes_rad = np.arccos(dot_product)
                angle_between_planes_deg = np.degrees(angle_between_planes_rad)
        else:
            # Scale down the angle differences proportionally
            scale_factor = max_angle_between_sheets / angle_between_planes_deg
            angle_diff_x = angle_diff_x * scale_factor
            angle_diff_y = angle_diff_y * scale_factor
            
            # Recalculate angles
            angle2_x = angle1_x + angle_diff_x
            angle2_y = angle1_y + angle_diff_y
            angle2_x_rad = np.radians(angle2_x)
            angle2_y_rad = np.radians(angle2_y)
            
            # Recalculate normal and angle
            normal2 = calculate_plane_normal(angle2_x_rad, angle2_y_rad)
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            angle_between_planes_rad = np.arccos(dot_product)
            angle_between_planes_deg = np.degrees(angle_between_planes_rad)
    
    # Calculate and print the angle difference between sheets
    angle_diff_x_deg = abs(angle2_x - angle_x)
    angle_diff_y_deg = abs(angle2_y - angle_y)
    print(f"Angle between sheets: Component differences - X={angle_diff_x_deg:.2f}°, Y={angle_diff_y_deg:.2f}°")
    print(f"  Actual geometric angle between planes: {angle_between_planes_deg:.2f}° (max allowed: {max_angle_between_sheets}°)")
    
    # Create rotation matrices (not used for tilt, but kept for potential future use)
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
    # Reduce amplifier for good pairs to make angles appear more realistic
    # The actual angle between planes is already constrained to 0-30 degrees
    tilt_amplifier = 2.0  # Reduced from 4.0 to make angles appear more realistic
    
    # Plane 1: Tilted with texture and curvature
    # Apply tilt based on angle type (using angle_x, angle_y, angle_z for plane 1)
    if angle_type == "tilt_x":
        # Tilt around X-axis: depth varies with Y position
        tilt_factor1 = np.tan(angle_x_rad) * Y * tilt_amplifier
        depth1 = base_depth + texture1 + tilt_factor1 + curvature1
    elif angle_type == "tilt_y":
        # Tilt around Y-axis: depth varies with X position
        tilt_factor1 = np.tan(angle_y_rad) * X * tilt_amplifier
        depth1 = base_depth + texture1 + tilt_factor1 + curvature1
    elif angle_type == "tilt_both":
        # Tilt around both axes
        tilt_factor1 = (np.tan(angle_x_rad) * Y + np.tan(angle_y_rad) * X) * tilt_amplifier
        depth1 = base_depth + texture1 + tilt_factor1 + curvature1
    else:
        # Random or rotation: apply tilt based on all axes
        tilt_factor1 = (np.tan(angle_x_rad) * Y + np.tan(angle_y_rad) * X) * tilt_amplifier
        depth1 = base_depth + texture1 + tilt_factor1 + curvature1
    
    # Plane 2: Different angle to create angle BETWEEN sheets (0-30 degrees), with curvature
    plane2_depth = base_depth + plane_separation
    
    # Apply tilt to plane 2 using angle2_x, angle2_y, angle2_z (ensuring 0-30 degree difference)
    if angle_type == "tilt_x":
        # Tilt around X-axis with controlled angle difference (0-30 degrees)
        tilt_factor2 = np.tan(angle2_x_rad) * Y * tilt_amplifier
        depth2 = plane2_depth + texture2 + tilt_factor2 + curvature2
    elif angle_type == "tilt_y":
        # Tilt around Y-axis with controlled angle difference (0-30 degrees)
        tilt_factor2 = np.tan(angle2_y_rad) * X * tilt_amplifier
        depth2 = plane2_depth + texture2 + tilt_factor2 + curvature2
    elif angle_type == "tilt_both":
        # Tilt around both axes with controlled angle differences (0-30 degrees)
        tilt_factor2 = (np.tan(angle2_x_rad) * Y + np.tan(angle2_y_rad) * X) * tilt_amplifier
        depth2 = plane2_depth + texture2 + tilt_factor2 + curvature2
    else:
        # Random: apply tilt based on all axes with controlled angle differences (0-30 degrees)
        tilt_factor2 = (np.tan(angle2_x_rad) * Y + np.tan(angle2_y_rad) * X) * tilt_amplifier
        depth2 = plane2_depth + texture2 + tilt_factor2 + curvature2
    
    # STRICT separation control for angled sheets - ensure separation stays within 2-8cm range
    min_separation = 0.02  # 2cm minimum
    max_separation = 0.08   # 8cm maximum
    
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
    
    # Add small X/Y spatial offset for good pairs (optional augmentation)
    xy_offset_x = 0.0
    xy_offset_y = 0.0
    if xy_offset_range[1] > 0.0:  # If offset range is specified
        xy_offset_x = random.uniform(xy_offset_range[0], xy_offset_range[1]) * random.choice([-1, 1])
        xy_offset_y = random.uniform(xy_offset_range[0], xy_offset_range[1]) * random.choice([-1, 1])
    
    # Convert depth maps to XYZ point clouds
    xyz1 = depth_to_xyz(depth1, X, Y)
    # Apply X/Y offset to second sheet
    X2_offset = X + xy_offset_x
    Y2_offset = Y + xy_offset_y
    xyz2 = depth_to_xyz(depth2, X2_offset, Y2_offset)
    
    print(f"Angled Sheet 1 XYZ coordinates: {xyz1.shape} points")
    print(f"Sheet 2 XYZ coordinates: {xyz2.shape} points")
    
    return xyz1, xyz2, depth1, depth2, X, Y

def create_curved_depth_pairs():
    """
    Create 300 curved XYZ sheet pairs with varying separations and curvatures.
    Saves them in depth_curved folder.
    """
    import os
    
    # Create main output directory
    output_dir = "depth_map_pairs"
    curved_dir = os.path.join(output_dir, "depth_curved")
    os.makedirs(curved_dir, exist_ok=True)
    
    # Create PNG, NPY, and XYZ subdirectories
    png_dir = os.path.join(curved_dir, "depth_png")
    npy_dir = os.path.join(curved_dir, "depth_npy")
    xyz_dir = os.path.join(curved_dir, "xyz_coords")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)
    
    # Define curvature types
    curvature_types = ["random", "convex", "concave", "wave", "saddle", "flowy", "fabric", "mixed"]
    
    print(f"Creating 300 curved XYZ sheet pairs")
    print(f"Output directory: {curved_dir}/")
    print(f"PNG files: {png_dir}/")
    print(f"NPY files: {npy_dir}/")
    print(f"XYZ files: {xyz_dir}/")
    print(f"Curvature types: {curvature_types}")
    
    for i in range(500):
        # Random separation (2-8cm)
        sep_cm = random.uniform(2, 8)
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
        
        # Add small X/Y offset for good pairs (0.5-3cm range)
        xy_offset_range = (0.005, 0.015)  # 0.5cm to 1.5cm
        xyz1, xyz2, depth1, depth2, X, Y = create_angled_depth_maps(plane_separation=sep_m, curvature_type=curvature_type, rotation_angles=rotation_angles, xy_offset_range=xy_offset_range)
        
        # Convert to 16-bit PNG (depth in millimeters)
        depth1_mm = (depth1 * 1000).astype(np.uint16)
        depth2_mm = (depth2 * 1000).astype(np.uint16)
        
        # Save PNG files (for backwards compatibility)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_a.png"), depth1_mm)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_b.png"), depth2_mm)
        
        # Save NPY depth files (for backwards compatibility)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_a.npy"), depth1)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_b.npy"), depth2)
        
        # Save XYZ coordinates
        np.save(os.path.join(xyz_dir, f"xyz{i+1:03d}_a.npy"), xyz1)
        np.save(os.path.join(xyz_dir, f"xyz{i+1:03d}_b.npy"), xyz2)
        np.savetxt(os.path.join(xyz_dir, f"xyz{i+1:03d}_a.txt"), xyz1, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(xyz_dir, f"xyz{i+1:03d}_b.txt"), xyz2, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Print progress every 50 pairs
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {i+1}/300 - Separation {sep_cm:.2f}cm, Curvature: {curvature_type}")
            print(f"  Sheet A: {xyz1.shape[0]} points, depth {depth1.min():.3f}m - {depth1.max():.3f}m")
            print(f"  Sheet B: {xyz2.shape[0]} points, depth {depth2.min():.3f}m - {depth2.max():.3f}m")
            print(f"  Actual separation: {(depth2.mean() - depth1.mean())*100:.2f}cm")
    
    print(f"\nAll 300 curved XYZ sheet pairs created successfully!")
    print(f"Total pairs: 300")
    print(f"Separation range: 2.00cm - 8.00cm")
    print(f"Curvature types used: {curvature_types}")
    print(f"Files saved in: {png_dir}/, {npy_dir}/, and {xyz_dir}/")
    print(f"File naming:")
    print(f"  Depth: depth001_a.png, depth001_b.png, ..., depth300_a.png, depth300_b.png")
    print(f"  XYZ: xyz001_a.npy, xyz001_b.npy, ..., xyz300_a.npy, xyz300_b.npy")
    print(f"  XYZ text: xyz001_a.txt, xyz001_b.txt, ..., xyz300_a.txt, xyz300_b.txt")

def create_angled_depth_pairs():
    """
    Create 500 angled XYZ sheet pairs with varying separations and angles.
    Saves them in depth_angled folder.
    """
    import os
    
    # Create main output directory
    output_dir = "depth_map_pairs"
    angled_dir = os.path.join(output_dir, "depth_angled")
    os.makedirs(angled_dir, exist_ok=True)
    
    # Create PNG, NPY, and XYZ subdirectories
    png_dir = os.path.join(angled_dir, "depth_png")
    npy_dir = os.path.join(angled_dir, "depth_npy")
    xyz_dir = os.path.join(angled_dir, "xyz_coords")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)
    
    # Define angle types
    angle_types = ["random", "tilt_x", "tilt_y", "tilt_both", "rotation"]
    
    # Define texture types (note: "none" can also be used with spherical curvatures)
    texture_types = ["random", "wave", "noise", "grid", "spots"]
    
    # Define curvature types - mix of none and various spherical curvatures
    # Note: Actual selection prioritizes spherical curvatures (75% chance)
    # When spherical curvature is present, 35% will have no texture (clean spheres)
    curvature_types = [
        "none", "convex", "concave", "mixed", 
        "concave_diff_radius", "convex_diff_radius",
        "strong_convex", "strong_concave",
        "weak_convex", "weak_concave",
        "opposite_sphere"
    ]
    
    print(f"Creating 1000 angled XYZ sheet pairs with texture and curvature")
    print(f"Output directory: {angled_dir}/")
    print(f"PNG files: {png_dir}/")
    print(f"NPY files: {npy_dir}/")
    print(f"XYZ files: {xyz_dir}/")
    print(f"Angle types: {angle_types}")
    print(f"Texture types: {texture_types} (plus 'none' for clean spherical curvature)")
    print(f"Curvature types: {curvature_types}")
    print(f"Note: 75% will have spherical curvature, and 35% of those will have no texture")
    
    for i in range(1000):
        # Random separation (2-8cm)
        sep_cm = random.uniform(2, 8)
        sep_m = sep_cm / 100.0
        
        # Random angle type
        angle_type = random.choice(angle_types)
        
        # Random curvature type - prioritize spherical curvatures (75% chance vs 25% none)
        # This ensures more spherical curvature sheets in the dataset
        if random.random() < 0.75:  # 75% chance for spherical curvature
            spherical_curvatures = [
                "convex", "concave", "mixed", 
                "concave_diff_radius", "convex_diff_radius",
                "strong_convex", "strong_concave",
                "weak_convex", "weak_concave",
                "opposite_sphere"
            ]
            curvature_type = random.choice(spherical_curvatures)
            
            # When there's spherical curvature, sometimes use no texture (35% chance)
            # This creates clean spherical curvature sheets without surface texture
            if random.random() < 0.35:  # 35% chance for no texture with spherical curvature
                texture_type = "none"
            else:  # 65% chance for texture + spherical curvature
                texture_type = random.choice(texture_types)
        else:  # 25% chance for no curvature
            curvature_type = "none"
            # When no curvature, always use texture
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
        
        # Add small X/Y offset for good pairs (0.5-3cm range)
        xy_offset_range = (0.005, 0.015)  # 0.5cm to 1.5cm
        xyz1, xyz2, depth1, depth2, X, Y = create_angled_depth_maps(plane_separation=sep_m, angle_type=angle_type, texture_type=texture_type, curvature_type=curvature_type, rotation_angles=rotation_angles, xy_offset_range=xy_offset_range)
        
        # Convert to 16-bit PNG (depth in millimeters)
        depth1_mm = (depth1 * 1000).astype(np.uint16)
        depth2_mm = (depth2 * 1000).astype(np.uint16)
        
        # Save PNG files (for backwards compatibility)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_a.png"), depth1_mm)
        cv2.imwrite(os.path.join(png_dir, f"depth{i+1:03d}_b.png"), depth2_mm)
        
        # Save NPY depth files (for backwards compatibility)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_a.npy"), depth1)
        np.save(os.path.join(npy_dir, f"depth{i+1:03d}_b.npy"), depth2)
        
        # Save XYZ coordinates
        np.save(os.path.join(xyz_dir, f"xyz{i+1:03d}_a.npy"), xyz1)
        np.save(os.path.join(xyz_dir, f"xyz{i+1:03d}_b.npy"), xyz2)
        np.savetxt(os.path.join(xyz_dir, f"xyz{i+1:03d}_a.txt"), xyz1, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(xyz_dir, f"xyz{i+1:03d}_b.txt"), xyz2, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Print progress every 50 pairs
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Progress: {i+1}/1000 - Separation {sep_cm:.2f}cm, Angle: {angle_type}, Texture: {texture_type}, Curvature: {curvature_type}")
            print(f"  Sheet A: {xyz1.shape[0]} points, depth {depth1.min():.3f}m - {depth1.max():.3f}m")
            print(f"  Sheet B: {xyz2.shape[0]} points, depth {depth2.min():.3f}m - {depth2.max():.3f}m")
            print(f"  Actual separation: {(depth2.mean() - depth1.mean())*100:.2f}cm")
    
    print(f"\nAll 1000 angled XYZ sheet pairs with texture and curvature created successfully!")
    print(f"Total pairs: 1000")
    print(f"~750 pairs with spherical curvature (~260 with no texture, ~490 with texture)")
    print(f"~250 pairs with no curvature (all have texture)")
    print(f"Separation range: 2.00cm - 8.00cm")
    print(f"Angle types used: {angle_types}")
    print(f"Texture types used: {texture_types}")
    print(f"Curvature types used: {curvature_types}")
    print(f"Files saved in: {png_dir}/, {npy_dir}/, and {xyz_dir}/")
    print(f"File naming:")
    print(f"  Depth: depth001_a.png, depth001_b.png, ..., depth1000_a.png, depth1000_b.png")
    print(f"  XYZ: xyz001_a.npy, xyz001_b.npy, ..., xyz1000_a.npy, xyz1000_b.npy")
    print(f"  XYZ text: xyz001_a.txt, xyz001_b.txt, ..., xyz1000_a.txt, xyz1000_b.txt")

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

def save_xyz_sheets(xyz1, xyz2, prefix="sheet"):
    """Save XYZ coordinates as numpy arrays and text files."""
    # Centralize the centroid of the pair to origin
    xyz1, xyz2 = centralize_pair_centroid(xyz1, xyz2)
    
    # Save as numpy arrays
    np.save(f"{prefix}_sheet1_xyz.npy", xyz1)
    np.save(f"{prefix}_sheet2_xyz.npy", xyz2)
    
    # Save as text files (CSV format)
    np.savetxt(f"{prefix}_sheet1_xyz.txt", xyz1, delimiter=',', 
               header='X,Y,Z', comments='', fmt='%.6f')
    np.savetxt(f"{prefix}_sheet2_xyz.txt", xyz2, delimiter=',', 
               header='X,Y,Z', comments='', fmt='%.6f')
    
    # Save as PLY point cloud files for visualization
    save_ply(xyz1, f"{prefix}_sheet1.ply")
    save_ply(xyz2, f"{prefix}_sheet2.ply")
    
    print(f"\nXYZ sheets saved as:")
    print(f"  - {prefix}_sheet1_xyz.npy and {prefix}_sheet2_xyz.npy (NumPy arrays)")
    print(f"  - {prefix}_sheet1_xyz.txt and {prefix}_sheet2_xyz.txt (CSV text files)")
    print(f"  - {prefix}_sheet1.ply and {prefix}_sheet2.ply (PLY point clouds)")
    print(f"  Each sheet has {xyz1.shape[0]} points (32x32 grid)")

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
    
    print(f"\nDepth maps saved as:")
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

def create_good_xyz_sheets_at_scale(num_pairs=20):
    """
    Create good XYZ sheet pairs at the scale and separation range matching existing good patches.
    Includes textures, curvatures, and tilts for realistic variation.
    
    Based on analysis of existing good patches:
    - Patch size: 32x32 = 1024 points
    - Coordinates in millimeters (mm)
    - Patch A (cam1): X ~-15k to +13k mm, Y ~-4k to +25k mm, Z ~800k-850k mm
    - Patch B (cam2): X ~170k-220k mm, Y ~40k-90k mm, Z ~-800k to -850k mm
    - Separation: X ~175-200m, Y ~35-60m, Z ~1.6km
    
    Files are saved in the 'good_patches' directory using the same format as save_xyz_sheets():
    - good_patches/good_sheet{i:05d}_sheet1_xyz.npy and good_patches/good_sheet{i:05d}_sheet2_xyz.npy
    - good_patches/good_sheet{i:05d}_sheet1_xyz.txt and good_patches/good_sheet{i:05d}_sheet2_xyz.txt (CSV format)
    - good_patches/good_sheet{i:05d}_sheet1.ply and good_patches/good_sheet{i:05d}_sheet2.ply
    
    Args:
        num_pairs: Number of patch pairs to generate
    """
    # Create output directory
    output_dir = "good_patches"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters based on analysis
    patch_size = 32
    num_points = patch_size * patch_size  # 1024 points
    
    # Patch A (cam1) ranges in mm
    patch_a_x_range = (-4000, 4000)  # ~10k mm range (adjusted for very close patches)
    patch_a_y_range = (-3000, 6000)   # ~11k mm range (adjusted for very close patches)
    patch_a_z_range = (80000, 100000)  # Adjusted for extremely close separation
    
    # Patch B (cam2) ranges in mm - adjusted for variable separation (closer and current range)
    patch_b_x_range = (-4000, 4000)  # ~11k mm range (accommodates both closer and current separations)
    patch_b_y_range = (-3000, 6000)    # ~9k mm range (accommodates both closer and current separations)
    patch_b_z_range = (80000, 100000)  # Variable Z separation (accommodates both closer and current ranges)
    
    # Separation ranges (in mm) - single continuous range for random selection
    # Combined range from very close to current range
    x_separation_range = (100, 1500)  # 0.2-2m (combined range for random selection)
    y_separation_range = (50, 1000)    # 0.1-1.5m (combined range for random selection)
    z_separation_range = (60000, 100000)  # 0.1-0.15km (combined range for random selection, ensures patch B stays negative)
    
    # Define texture and curvature types
    texture_types = ["random", "wave", "noise", "grid", "spots", "none"]
    curvature_types = [
        "none", "convex", "concave", "mixed", 
        "concave_diff_radius", "convex_diff_radius",
        "strong_convex", "strong_concave",
        "weak_convex", "weak_concave",
        "opposite_sphere"
    ]
    angle_types = ["random", "tilt_x", "tilt_y", "tilt_both", "rotation"]
    
    print(f"Generating {num_pairs} good XYZ sheet pairs at correct scale...")
    print(f"Patch size: {patch_size}x{patch_size} = {num_points} points")
    print(f"Coordinates in millimeters (mm)")
    print(f"Including textures, curvatures, and tilts")
    print(f"Files will be saved in '{output_dir}' directory")
    print()
    
    for pair_idx in range(1, num_pairs + 1):
        # Generate patch A (cam1) center
        center_a_x = random.uniform(patch_a_x_range[0], patch_a_x_range[1])
        center_a_y = random.uniform(patch_a_y_range[0], patch_a_y_range[1])
        center_a_z = random.uniform(patch_a_z_range[0], patch_a_z_range[1])
        
        # Generate separation - randomly pick from the defined range
        x_sep = random.uniform(x_separation_range[0], x_separation_range[1])
        y_sep = random.uniform(y_separation_range[0], y_separation_range[1])
        z_sep = random.uniform(z_separation_range[0], z_separation_range[1])
        
        # Calculate patch B (cam2) center based on separation
        # Note: cam2 has negative Z, so we subtract the separation
        center_b_x = center_a_x + x_sep
        center_b_y = center_a_y + y_sep
        center_b_z = center_a_z - z_sep  # Negative Z for cam2
        
        # Ensure patch B is within its valid range
        if center_b_x < patch_b_x_range[0] or center_b_x > patch_b_x_range[1]:
            # Adjust to stay in range
            center_b_x = random.uniform(patch_b_x_range[0], patch_b_x_range[1])
            center_a_x = center_b_x - x_sep
        
        if center_b_y < patch_b_y_range[0] or center_b_y > patch_b_y_range[1]:
            center_b_y = random.uniform(patch_b_y_range[0], patch_b_y_range[1])
            center_a_y = center_b_y - y_sep
        
        if center_b_z < patch_b_z_range[0] or center_b_z > patch_b_z_range[1]:
            center_b_z = random.uniform(patch_b_z_range[0], patch_b_z_range[1])
            center_a_z = center_b_z + z_sep
        
        # Generate patch extents (spatial size of patch in mm)
        # Based on analysis: patches are ~28-30k mm in X and Y
        patch_extent_x = random.uniform(28000, 30000)  # mm
        patch_extent_y = random.uniform(28000, 30000)  # mm
        patch_extent_z = random.uniform(30000, 50000)  # mm (depth variation)
        
        # Create coordinate grids for patch A (relative to center, in mm)
        x_a = np.linspace(-patch_extent_x/2, patch_extent_x/2, patch_size)
        y_a = np.linspace(-patch_extent_y/2, patch_extent_y/2, patch_size)
        X_a, Y_a = np.meshgrid(x_a, y_a)
        
        # Normalize X, Y to [-1, 1] range for frequency-based textures/curvatures
        X_a_norm = X_a / (patch_extent_x / 2)
        Y_a_norm = Y_a / (patch_extent_y / 2)
        
        # Randomly select texture, curvature, and angle types - each sheet can have different patterns
        # Select texture types independently for each sheet
        texture_type_1 = random.choice(texture_types)
        texture_type_2 = random.choice(texture_types)
        
        # Select curvature types independently for each sheet
        # Prioritize spherical curvatures (75% chance for each sheet)
        spherical_curvatures = [
            "convex", "concave", "mixed", 
            "concave_diff_radius", "convex_diff_radius",
            "strong_convex", "strong_concave",
            "weak_convex", "weak_concave",
            "opposite_sphere"
        ]
        
        if random.random() < 0.75:
            curvature_type_1 = random.choice(spherical_curvatures)
            # 35% chance for no texture with spherical curvature
            if random.random() < 0.35:
                texture_type_1 = "none"
        else:
            curvature_type_1 = "none"
            if texture_type_1 == "none":
                texture_type_1 = random.choice(["random", "wave", "noise", "grid", "spots"])
        
        if random.random() < 0.75:
            curvature_type_2 = random.choice(spherical_curvatures)
            # 35% chance for no texture with spherical curvature
            if random.random() < 0.35:
                texture_type_2 = "none"
        else:
            curvature_type_2 = "none"
            if texture_type_2 == "none":
                texture_type_2 = random.choice(["random", "wave", "noise", "grid", "spots"])
        
        angle_type = random.choice(angle_types)
        
        # Generate texture patterns (scaled to mm - moderate, visible but not extreme)
        # Use moderate values: 200-800mm variations (0.025-0.1% of base depth ~800k mm)
        # Each sheet can have different texture patterns
        def generate_texture(texture_type, X_norm, Y_norm):
            """Generate texture pattern for a single sheet."""
            if texture_type == "none" or texture_type == "no_texture":
                return np.zeros((patch_size, patch_size))
            elif texture_type == "random":
                # Random noise texture - scale to mm (200-500mm variation)
                return np.random.normal(0, random.uniform(200, 500), (patch_size, patch_size))
            elif texture_type == "wave":
                # Wave pattern - scale to mm (300-800mm amplitude)
                freq = random.uniform(3, 6)
                amplitude = random.uniform(300, 800)
                return amplitude * np.sin(freq * X_norm) * np.cos(freq * Y_norm)
            elif texture_type == "noise":
                # Perlin-like noise - scale to mm (200-500mm variation)
                return np.random.normal(0, random.uniform(200, 500), (patch_size, patch_size))
            elif texture_type == "grid":
                # Grid pattern - scale to mm (300-800mm amplitude)
                grid_size = random.randint(6, 10)
                amplitude = random.uniform(300, 800)
                return amplitude * (np.sin(grid_size * X_norm) + np.sin(grid_size * Y_norm))
            elif texture_type == "spots":
                # Spot pattern - scale to mm (500-1500mm intensity)
                texture = np.zeros((patch_size, patch_size))
                num_spots = random.randint(2, 5)
                for _ in range(num_spots):
                    cx, cy = random.randint(0, patch_size-1), random.randint(0, patch_size-1)
                    radius = random.randint(3, 6)
                    intensity = random.uniform(-1500, 1500)  # mm
                    y_ind, x_ind = np.ogrid[:patch_size, :patch_size]
                    mask = (x_ind - cx)**2 + (y_ind - cy)**2 <= radius**2
                    texture[mask] += intensity
                return texture
            else:
                return np.random.normal(0, random.uniform(200, 500), (patch_size, patch_size))
        
        # Generate textures independently for each sheet
        texture1 = generate_texture(texture_type_1, X_a_norm, Y_a_norm)
        texture2 = generate_texture(texture_type_2, X_a_norm, Y_a_norm)
        
        # Generate curvature patterns (scaled to mm - use normalized coordinates for proper scaling)
        # Curvature should be visible but not extreme - use normalized coords and scale appropriately
        # Each sheet can have different curvature patterns
        def generate_curvature(curvature_type, X_norm, Y_norm):
            """Generate curvature pattern for a single sheet."""
            if curvature_type == "none":
                return np.zeros((patch_size, patch_size))
            elif curvature_type == "convex":
                radius_norm = random.uniform(0.5, 2.0)  # Normalized radius
                amplitude = random.uniform(1500, 2500)  # mm - further reduced max from 3500
                return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "concave":
                radius_norm = random.uniform(0.5, 2.0)  # Normalized radius
                amplitude = random.uniform(1500, 2500)  # mm - further reduced max from 3500
                return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "mixed":
                # For mixed, randomly choose convex or concave for this sheet
                if random.random() > 0.5:
                    radius_norm = random.uniform(0.5, 2.0)
                    amplitude = random.uniform(1500, 2500)
                    return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
                else:
                    radius_norm = random.uniform(0.5, 2.0)
                    amplitude = random.uniform(1500, 2500)
                    return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "concave_diff_radius":
                radius_norm = random.uniform(0.5, 2.0)
                amplitude = random.uniform(1500, 2500)  # mm - further reduced max from 3500
                return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "convex_diff_radius":
                radius_norm = random.uniform(0.5, 2.0)
                amplitude = random.uniform(1500, 2500)  # mm - further reduced max from 3500
                return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "strong_convex":
                radius_norm = random.uniform(0.3, 1.0)  # Tighter radius
                amplitude = random.uniform(3000, 4000)  # mm - further reduced max from 6000
                return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "strong_concave":
                radius_norm = random.uniform(0.3, 1.0)  # Tighter radius
                amplitude = random.uniform(3000, 4000)  # mm - further reduced max from 6000
                return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "weak_convex":
                radius_norm = random.uniform(1.5, 3.0)  # Larger radius
                amplitude = random.uniform(800, 1800)  # mm - further reduced max from 2500
                return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "weak_concave":
                radius_norm = random.uniform(1.5, 3.0)  # Larger radius
                amplitude = random.uniform(800, 1800)  # mm - further reduced max from 2500
                return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            elif curvature_type == "opposite_sphere":
                # For opposite_sphere, randomly choose convex or concave for this sheet
                radius_norm = random.uniform(0.5, 1.5)
                amplitude = random.uniform(2000, 3500)  # mm - further reduced max from 5000
                if random.random() > 0.5:
                    return amplitude * (X_norm**2 + Y_norm**2) / radius_norm
                else:
                    return -amplitude * (X_norm**2 + Y_norm**2) / radius_norm
            else:
                return np.zeros((patch_size, patch_size))
        
        # Generate curvatures independently for each sheet
        curvature1 = generate_curvature(curvature_type_1, X_a_norm, Y_a_norm)
        curvature2 = generate_curvature(curvature_type_2, X_a_norm, Y_a_norm)
        
        # Decide whether to apply slight bend augmentation (90% chance - very common augmentation)
        use_slight_bend = random.random() < 0.9
        
        # Generate angle variations - sheets should face each other (max 8 degrees between sheets)
        max_angle_between_sheets = 8.0  # Maximum angle between sheets in degrees (reduced for closer to parallel)
        base_angle_range = 5.0  # Reduced base angle range so sheets are more aligned
        
        if use_slight_bend:
            # Apply slight bend augmentation
            # Choose which sheet(s) to bend (both can be bent, or just one)
            bend_sheet_1 = random.random() < 0.9  # 90% chance sheet 1 is bent
            bend_sheet_2 = random.random() < 0.9  # 90% chance sheet 2 is bent
            
            # Choose bend axis and direction
            bend_axis = random.choice(['x', 'y'])
            bend_from_left = random.random() > 0.5
            
            # Slight bend: 20-40% of sheet is bent (much less than bad patches which use 70%+)
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
        else:
            # No bend - use uniform masks
            transition_mask_1 = np.ones_like(X_a_norm)
            transition_mask_2 = np.ones_like(X_a_norm)
            bend_sheet_1 = False
            bend_sheet_2 = False
            bend_axis = None
        
        if angle_type == "random":
            angle1_x = random.uniform(-base_angle_range, base_angle_range)
            angle1_y = random.uniform(-base_angle_range, base_angle_range)
            angle1_z = random.uniform(-base_angle_range, base_angle_range)
            angle_diff_x = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_diff_y = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_diff_z = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_x = angle1_x
            angle_y = angle1_y
            angle_z = angle1_z
            angle2_x = angle1_x + angle_diff_x
            angle2_y = angle1_y + angle_diff_y
            angle2_z = angle1_z + angle_diff_z
        elif angle_type == "tilt_x":
            angle1_x = random.uniform(-base_angle_range, base_angle_range)
            angle_diff_x = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_diff_y = 0  # No Y tilt difference
            angle_x = angle1_x
            angle2_x = angle1_x + angle_diff_x
            angle_y = 0
            angle_z = 0
            angle2_y = 0
            angle2_z = 0
        elif angle_type == "tilt_y":
            angle_x = 0
            angle_z = 0
            angle2_x = 0
            angle2_z = 0
            angle1_y = random.uniform(-base_angle_range, base_angle_range)
            angle_diff_x = 0  # No X tilt difference
            angle_diff_y = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_y = angle1_y
            angle2_y = angle1_y + angle_diff_y
        elif angle_type == "tilt_both":
            angle1_x = random.uniform(-base_angle_range, base_angle_range)
            angle1_y = random.uniform(-base_angle_range, base_angle_range)
            angle_diff_x = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_diff_y = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_x = angle1_x
            angle_y = angle1_y
            angle2_x = angle1_x + angle_diff_x
            angle2_y = angle1_y + angle_diff_y
            angle_z = 0
            angle2_z = 0
        elif angle_type == "rotation":
            angle_x = 0
            angle_y = 0
            angle2_x = 0
            angle2_y = 0
            angle_diff_x = 0  # No X tilt difference
            angle_diff_y = 0  # No Y tilt difference
            angle1_z = random.uniform(-base_angle_range, base_angle_range)
            angle_diff_z = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_z = angle1_z
            angle2_z = angle1_z + angle_diff_z
        else:
            angle1_x = random.uniform(-base_angle_range, base_angle_range)
            angle1_y = random.uniform(-base_angle_range, base_angle_range)
            angle1_z = random.uniform(-base_angle_range, base_angle_range)
            angle_diff_x = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_diff_y = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_diff_z = random.uniform(0, max_angle_between_sheets) * random.choice([-1, 1])
            angle_x = angle1_x
            angle_y = angle1_y
            angle_z = angle1_z
            angle2_x = angle1_x + angle_diff_x
            angle2_y = angle1_y + angle_diff_y
            angle2_z = angle1_z + angle_diff_z
        
        # Ensure the geometric angle between sheets is ≤ 8 degrees
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
        
        # Convert to radians for calculation
        angle_x_rad_temp = np.radians(angle_x)
        angle_y_rad_temp = np.radians(angle_y)
        angle2_x_rad_temp = np.radians(angle2_x)
        angle2_y_rad_temp = np.radians(angle2_y)
        
        # Calculate normal vectors and geometric angle
        normal1 = calculate_plane_normal(angle_x_rad_temp, angle_y_rad_temp)
        normal2 = calculate_plane_normal(angle2_x_rad_temp, angle2_y_rad_temp)
        dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
        angle_between_planes_rad = np.arccos(dot_product)
        angle_between_planes_deg = np.degrees(angle_between_planes_rad)
        
        # If geometric angle is too large, scale down the angle differences
        if angle_between_planes_deg > max_angle_between_sheets:
            scale_factor = max_angle_between_sheets / angle_between_planes_deg
            angle_diff_x = angle_diff_x * scale_factor
            angle_diff_y = angle_diff_y * scale_factor
            
            # Recalculate angles
            angle2_x = angle1_x + angle_diff_x
            angle2_y = angle1_y + angle_diff_y
            angle2_x_rad_temp = np.radians(angle2_x)
            angle2_y_rad_temp = np.radians(angle2_y)
            
            # Recalculate to verify
            normal2 = calculate_plane_normal(angle2_x_rad_temp, angle2_y_rad_temp)
            dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
            angle_between_planes_rad = np.arccos(dot_product)
            angle_between_planes_deg = np.degrees(angle_between_planes_rad)
        
        # Convert angles to radians
        angle_x_rad = np.radians(angle_x)
        angle_y_rad = np.radians(angle_y)
        angle_z_rad = np.radians(angle_z)
        angle2_x_rad = np.radians(angle2_x)
        angle2_y_rad = np.radians(angle2_y)
        angle2_z_rad = np.radians(angle2_z)
        
        # Apply slight bend augmentation if enabled
        if use_slight_bend and bend_sheet_1:
            # Slight bend: add 10-30 degrees (much less than bad patches which use 70-90 degrees)
            bent_angle_x = angle_x + random.choice([-1, 1]) * random.uniform(10, 30)
            bent_angle_y = angle_y + random.choice([-1, 1]) * random.uniform(0, 15)
            bent_angle_x_rad = np.radians(bent_angle_x)
            bent_angle_y_rad = np.radians(bent_angle_y)
            
            # Interpolate angles based on transition mask (gradual transition)
            angle1_x_map = angle_x + (bent_angle_x - angle_x) * (1.0 - transition_mask_1)
            angle1_y_map = angle_y + (bent_angle_y - angle_y) * (1.0 - transition_mask_1)
            angle1_x_rad = np.radians(angle1_x_map)
            angle1_y_rad = np.radians(angle1_y_map)
        else:
            angle1_x_rad = angle_x_rad
            angle1_y_rad = angle_y_rad
        
        if use_slight_bend and bend_sheet_2:
            # Slight bend for sheet 2
            bent_angle2_x = angle2_x + random.choice([-1, 1]) * random.uniform(10, 30)
            bent_angle2_y = angle2_y + random.choice([-1, 1]) * random.uniform(0, 15)
            bent_angle2_x_rad = np.radians(bent_angle2_x)
            bent_angle2_y_rad = np.radians(bent_angle2_y)
            
            # Interpolate angles based on transition mask
            angle2_x_map = angle2_x + (bent_angle2_x - angle2_x) * (1.0 - transition_mask_2)
            angle2_y_map = angle2_y + (bent_angle2_y - angle2_y) * (1.0 - transition_mask_2)
            angle2_x_rad = np.radians(angle2_x_map)
            angle2_y_rad = np.radians(angle2_y_map)
        else:
            angle2_x_rad = angle2_x_rad
            angle2_y_rad = angle2_y_rad
        
        # Apply tilt effects (scaled to mm, tilt_amplifier for visibility)
        # Reduced amplifier for less tilt
        tilt_amplifier = 2.0  # Reduced from 3.0 for less tilt
        
        # Patch A: Apply tilt, texture, and curvature
        if angle_type == "tilt_x":
            tilt_factor1 = np.tan(angle1_x_rad) * Y_a * tilt_amplifier
        elif angle_type == "tilt_y":
            tilt_factor1 = np.tan(angle1_y_rad) * X_a * tilt_amplifier
        elif angle_type == "tilt_both":
            tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
        else:
            tilt_factor1 = (np.tan(angle1_x_rad) * Y_a + np.tan(angle1_y_rad) * X_a) * tilt_amplifier
        
        # Create base Z coordinates for patch A (relative to center)
        z_a_base = tilt_factor1 + texture1 + curvature1
        
        # Apply rotation around Z-axis if needed
        if angle_z != 0:
            cos_z = np.cos(angle_z_rad)
            sin_z = np.sin(angle_z_rad)
            x_a_rot = X_a * cos_z - Y_a * sin_z
            y_a_rot = X_a * sin_z + Y_a * cos_z
        else:
            x_a_rot = X_a
            y_a_rot = Y_a
        
        # Final coordinates for patch A (add center offsets)
        x_a_final = x_a_rot + center_a_x
        y_a_final = y_a_rot + center_a_y
        z_a_final = z_a_base + center_a_z
        
        # Patch B: Apply tilt, texture, and curvature (with angle differences)
        if angle_type == "tilt_x":
            tilt_factor2 = np.tan(angle2_x_rad) * Y_a * tilt_amplifier
        elif angle_type == "tilt_y":
            tilt_factor2 = np.tan(angle2_y_rad) * X_a * tilt_amplifier
        elif angle_type == "tilt_both":
            tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
        else:
            tilt_factor2 = (np.tan(angle2_x_rad) * Y_a + np.tan(angle2_y_rad) * X_a) * tilt_amplifier
        
        # Create base Z coordinates for patch B (relative to center)
        z_b_base = tilt_factor2 + texture2 + curvature2
        
        # Apply rotation around Z-axis if needed
        if angle2_z != 0:
            cos_z2 = np.cos(angle2_z_rad)
            sin_z2 = np.sin(angle2_z_rad)
            x_b_rot = X_a * cos_z2 - Y_a * sin_z2
            y_b_rot = X_a * sin_z2 + Y_a * cos_z2
        else:
            x_b_rot = X_a
            y_b_rot = Y_a
        
        # Final coordinates for patch B (add center offsets)
        x_b_final = x_b_rot + center_b_x
        y_b_final = y_b_rot + center_b_y
        z_b_final = z_b_base + center_b_z
        
        # Create patch A and B XYZ arrays
        xyz_a = np.stack([x_a_final.flatten(), y_a_final.flatten(), z_a_final.flatten()], axis=1)
        xyz_b = np.stack([x_b_final.flatten(), y_b_final.flatten(), z_b_final.flatten()], axis=1)
        
        # Pivot both sheets together as a unit to add variation
        xyz_a, xyz_b = pivot_sheets_together(xyz_a, xyz_b)
        
        # Centralize the centroid of the pair to origin
        xyz_a, xyz_b = centralize_pair_centroid(xyz_a, xyz_b)
        
        # Save using the same format as save_xyz_sheets()
        prefix = f"good_sheet{pair_idx:05d}"
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f"{prefix}_sheet1_xyz.npy"), xyz_a)
        np.save(os.path.join(output_dir, f"{prefix}_sheet2_xyz.npy"), xyz_b)
        
        # Save as text files (CSV format, matching save_xyz_sheets format)
        np.savetxt(os.path.join(output_dir, f"{prefix}_sheet1_xyz.txt"), xyz_a, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        np.savetxt(os.path.join(output_dir, f"{prefix}_sheet2_xyz.txt"), xyz_b, delimiter=',', 
                   header='X,Y,Z', comments='', fmt='%.6f')
        
        # Save as PLY point cloud files for visualization
        save_ply(xyz_a, os.path.join(output_dir, f"{prefix}_sheet1.ply"))
        save_ply(xyz_b, os.path.join(output_dir, f"{prefix}_sheet2.ply"))
        
        if pair_idx % 5 == 0 or pair_idx == 1:
            print(f"Generated pair {pair_idx}/{num_pairs}:")
            print(f"  Patch A center: [{center_a_x:.1f}, {center_a_y:.1f}, {center_a_z:.1f}] mm")
            print(f"  Patch B center: [{center_b_x:.1f}, {center_b_y:.1f}, {center_b_z:.1f}] mm")
            print(f"  Separation: X={x_sep:.1f}, Y={y_sep:.1f}, Z={z_sep:.1f} mm")
            print(f"  Texture A: {texture_type_1}, Texture B: {texture_type_2}")
            print(f"  Curvature A: {curvature_type_1}, Curvature B: {curvature_type_2}, Angle: {angle_type}")
            if use_slight_bend:
                bend_info = f"Bent: Sheet1={bend_sheet_1}, Sheet2={bend_sheet_2}, Axis={bend_axis if use_slight_bend else 'N/A'}"
                print(f"  {bend_info}")
            print(f"  Saved: {output_dir}/{prefix}_sheet1_xyz.npy, {output_dir}/{prefix}_sheet2_xyz.npy")
            print(f"         {output_dir}/{prefix}_sheet1_xyz.txt, {output_dir}/{prefix}_sheet2_xyz.txt")
            print(f"         {output_dir}/{prefix}_sheet1.ply, {output_dir}/{prefix}_sheet2.ply")
            print()
    
    print(f"Successfully generated {num_pairs} good XYZ sheet pairs")
    print(f"Each sheet has {num_points} points (32x32 grid)")
    print(f"Coordinates are in millimeters (mm)")
    print(f"Features: textures, curvatures, tilts, and slight bends (90% chance - very common augmentation) applied")
    print(f"\nFiles saved in '{output_dir}' directory:")
    print(f"  - good_sheet{{i:05d}}_sheet1_xyz.npy and good_sheet{{i:05d}}_sheet2_xyz.npy (NumPy arrays)")
    print(f"  - good_sheet{{i:05d}}_sheet1_xyz.txt and good_sheet{{i:05d}}_sheet2_xyz.txt (CSV text files)")
    print(f"  - good_sheet{{i:05d}}_sheet1.ply and good_sheet{{i:05d}}_sheet2.ply (PLY point clouds)")

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
    parser.add_argument('--good-scale', action='store_true',
                       help='Create good XYZ sheet pairs at the scale matching existing good patches (millimeters)')
    parser.add_argument('--num-pairs', type=int, default=20,
                       help='Number of patch pairs to generate (default: 20)')
    parser.add_argument('--separation', type=float, default=2.0,
                       help='Separation between planes in cm (default: 2.0)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization (only create depth maps)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.good_scale:
        # Create good XYZ sheets at the correct scale
        create_good_xyz_sheets_at_scale(num_pairs=args.num_pairs)
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
        print(f"Creating XYZ sheets for parallel planes {args.separation}cm apart...")
        
        # Create depth maps with random rotations (now returns XYZ coordinates)
        # Add small X/Y offset for good pairs (0.5-3cm range)
        xy_offset_range = (0.0005, 0.0015)  # 0.5cm to 1.5cm
        xyz1, xyz2, depth1, depth2, X, Y = create_depth_maps(plane_separation=sep_m, xy_offset_range=xy_offset_range)
        
        # Print rotation information
        print("\n" + "="*60)
        print("XYZ SHEET INFORMATION")
        print("="*60)
        print("Both sheets are rotated around different axes:")
        print("  - X-axis rotation: Tilts the sheet up/down")
        print("  - Y-axis rotation: Tilts the sheet left/right") 
        print("  - Z-axis rotation: Rotates the sheet in-plane")
        print("="*60)
        print("This creates more realistic, non-parallel sheets")
        print("that better represent real-world objects and surfaces.")
        print("="*60)
        
        # Print statistics
        print_depth_stats(depth1, depth2)
        
        # Print XYZ statistics
        print("\nXYZ Sheet Statistics:")
        print("=" * 50)
        print(f"Sheet 1: {xyz1.shape[0]} points")
        print(f"  X range: [{xyz1[:, 0].min():.4f}, {xyz1[:, 0].max():.4f}] m")
        print(f"  Y range: [{xyz1[:, 1].min():.4f}, {xyz1[:, 1].max():.4f}] m")
        print(f"  Z range: [{xyz1[:, 2].min():.4f}, {xyz1[:, 2].max():.4f}] m")
        print(f"\nSheet 2: {xyz2.shape[0]} points")
        print(f"  X range: [{xyz2[:, 0].min():.4f}, {xyz2[:, 0].max():.4f}] m")
        print(f"  Y range: [{xyz2[:, 1].min():.4f}, {xyz2[:, 1].max():.4f}] m")
        print(f"  Z range: [{xyz2[:, 2].min():.4f}, {xyz2[:, 2].max():.4f}] m")
        
        if not args.no_viz:
            # Visualize
            print("\nCreating visualization...")
            fig = visualize_depth_maps(depth1, depth2)
            
            # Save plots
            plt.savefig('depth_maps_visualization.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as 'depth_maps_visualization.png'")
            
            # Show plot
            plt.show()
        
        # Save XYZ sheets
        save_xyz_sheets(xyz1, xyz2)
        
        # Also save depth maps for backwards compatibility
        save_depth_maps(depth1, depth2)

if __name__ == "__main__":
    main()
