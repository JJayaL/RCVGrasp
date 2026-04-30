#!/usr/bin/env python3
"""
Visualize two numpy patches as 3D point clouds.
"""

import numpy as np
# Try to import sklearn for PCA
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Note: scikit-learn not available. Install with: pip install scikit-learn for PCA reorientation")
import matplotlib
# Use interactive backend for zooming
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# Try to import Open3D for better point cloud visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Note: Open3D not available. Install with: pip install open3d for better point cloud visualization")


def load_patch(filepath):
    """Load a numpy patch file. If it's a combined preprocessed pair, split it into two patches."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Patch file not found: {filepath}")
    
    patch = np.load(filepath)
    print(f"Loaded {filepath}: shape={patch.shape}, dtype={patch.dtype}")
    
    # Check if this is a combined preprocessed pair (2048 points = 1024 + 1024)
    filename = os.path.basename(filepath)
    if '_preprocessed' in filename and patch.shape[0] == 2048:
        # Split into two patches
        patch1 = patch[:1024]
        patch2 = patch[1024:]
        print(f"  Detected combined preprocessed pair, splitting into two patches (1024 points each)")
        return [patch1, patch2]
    
    return patch


def print_patch_stats(patch, name):
    """Print statistics about a patch."""
    print(f"\n{name} Statistics:")
    print(f"  Shape: {patch.shape}")
    print(f"  Min: {np.min(patch, axis=0)}")
    print(f"  Max: {np.max(patch, axis=0)}")
    print(f"  Mean: {np.mean(patch, axis=0)}")
    print(f"  Std: {np.std(patch, axis=0)}")
    print(f"  Range (X): {np.min(patch[:, 0]):.2f} to {np.max(patch[:, 0]):.2f}")
    print(f"  Range (Y): {np.min(patch[:, 1]):.2f} to {np.max(patch[:, 1]):.2f}")
    print(f"  Range (Z): {np.min(patch[:, 2]):.2f} to {np.max(patch[:, 2]):.2f}")


def reorient_patch_pair_with_pca(patch_a, patch_b, target_plane='xy'):
    """
    Reorient a patch pair using PCA so both patches have the same orientation.
    
    Computes PCA on the combined points of both patches, then rotates both patches
    so that the principal components align with the coordinate axes. This ensures
    both patches in a pair have the same orientation.
    
    Args:
        patch_a: First patch, shape (N, 3)
        patch_b: Second patch, shape (M, 3)
        target_plane: Target plane orientation ('xy', 'xz', or 'yz'). 
                     'xy' means patches will be oriented parallel to xy plane
                     (first PC -> x-axis, second PC -> y-axis, third PC -> z-axis)
    
    Returns:
        patch_a_rotated: Rotated first patch
        patch_b_rotated: Rotated second patch
        rotation_matrix: The rotation matrix applied (3x3)
        pca_components: The PCA components (3x3, each row is a component)
        explained_variance: Explained variance ratio for each component
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for PCA reorientation. Install with: pip install scikit-learn")
    # Combine both patches for PCA
    combined_points = np.vstack([patch_a, patch_b])
    
    # Center the combined points
    centroid = np.mean(combined_points, axis=0)
    centered_points = combined_points - centroid
    
    # Compute PCA
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    # Get principal components (each row is a component vector)
    # Components are ordered by explained variance (largest first)
    components = pca.components_  # Shape: (3, 3)
    explained_variance = pca.explained_variance_ratio_
    
    # Create rotation matrix to align PCA components with coordinate axes
    # We want: first PC -> x-axis, second PC -> y-axis, third PC -> z-axis
    # The rotation matrix R satisfies: R @ components.T = I
    # So: R = components (since components are orthonormal)
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
    
    return patch_a_rotated, patch_b_rotated, rotation_matrix, components, explained_variance


def visualize_patches_open3d(patches, labels=None, point_size=1.0, save_ply=None):
    """
    Visualize multiple numpy patches as 3D point clouds using Open3D.
    
    Args:
        patches: List of patch arrays, each of shape (N, 3)
        labels: List of labels for each patch (optional)
        point_size: Size of points in the visualization
        save_ply: Optional prefix to save PLY files
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required for this visualization. Install with: pip install open3d")
    
    num_patches = len(patches)
    if labels is None:
        labels = [f"Patch {i+1}" for i in range(num_patches)]
    
    # Define colors for each patch: blue, red, green, yellow
    patch_colors = [
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
    ]
    
    all_points_list = []
    all_colors_list = []
    pcd_list = []
    
    # Create point clouds for each patch
    for i, patch in enumerate(patches):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(patch.astype(np.float64))
        
        # Use solid color for each patch
        color = patch_colors[i % len(patch_colors)]
        colors = np.array([color] * len(patch))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        all_points_list.append(patch)
        all_colors_list.append(colors)
        pcd_list.append(pcd)
        
        # Save PLY files if requested
        if save_ply:
            ply_path = f"{save_ply}_patch{i+1}.ply"
            o3d.io.write_point_cloud(ply_path, pcd)
            if i == 0:
                print(f"\nSaved point clouds to:")
            print(f"  {ply_path}")
    
    # Create combined point cloud for visualization
    pcd_combined = o3d.geometry.PointCloud()
    all_points = np.vstack(all_points_list)
    all_colors = np.vstack(all_colors_list)
    pcd_combined.points = o3d.utility.Vector3dVector(all_points.astype(np.float64))
    pcd_combined.colors = o3d.utility.Vector3dVector(all_colors)
    
    # Create window title
    color_names = ["blue", "red", "green", "yellow"]
    title_parts = [f"{labels[i]} ({color_names[i % len(color_names)]})" for i in range(num_patches)]
    window_title = "Point Cloud: " + ", ".join(title_parts)
    
    # Set up visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1920, height=1080)
    vis.add_geometry(pcd_combined)
    
    # Configure render options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background
    
    # Set up view control
    view_control = vis.get_view_control()
    view_control.set_front([0.0, 0.0, -1.0])
    view_control.set_lookat(all_points.mean(axis=0))
    view_control.set_up([0.0, -1.0, 0.0])
    view_control.set_zoom(0.7)
    
    print("\n" + "="*70)
    print("Open3D Point Cloud Viewer")
    print("="*70)
    print("Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Shift + Mouse drag: Pan")
    print("  - Close window to exit")
    print("="*70 + "\n")
    
    vis.run()
    vis.destroy_window()


def visualize_patches(patches, labels=None, save_path=None, show=True, 
                      use_open3d=None, use_matplotlib=False, point_size=1.0, save_ply=None,
                      scales=None, use_pca_reorientation=False, centralize_centroid=True):
    """
    Visualize multiple numpy patches as 3D point clouds.
    
    Args:
        patches: List of patch arrays, each of shape (N, 3)
        labels: List of labels for each patch (optional)
        save_path: Optional path to save the figure (matplotlib only)
        show: Whether to display the figure
        use_open3d: Use Open3D for interactive visualization (None = auto-detect, True = force Open3D, False = force matplotlib)
        use_matplotlib: Force use of matplotlib (overrides use_open3d)
        point_size: Size of points (Open3D only)
        save_ply: Optional prefix to save PLY files (Open3D only)
        scales: Optional list of scale factors to apply to each patch (e.g., [1.0, 1.0, 0.001] to scale third patch by 0.001)
        use_pca_reorientation: If True, use PCA to reorient patch pairs so both patches in each pair have the same orientation
    """
    num_patches = len(patches)
    if labels is None:
        labels = [f"Patch {i+1}" for i in range(num_patches)]
    
    # Apply scaling if provided
    if scales is not None:
        if len(scales) != num_patches:
            raise ValueError(f"Number of scale factors ({len(scales)}) must match number of patches ({num_patches})")
        patches = [patch * scale for patch, scale in zip(patches, scales)]
        print("\nApplied scaling factors:")
        for i, (label, scale) in enumerate(zip(labels, scales)):
            print(f"  {label}: {scale}")
    
    # Apply PCA reorientation if requested
    if use_pca_reorientation:
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA reorientation. Install with: pip install scikit-learn")
        
        if num_patches >= 2 and num_patches % 2 == 0:
            print("\nApplying PCA reorientation to patch pairs:")
            for pair_idx in range(0, num_patches, 2):
                patch_a = patches[pair_idx]
                patch_b = patches[pair_idx + 1]
                
                patch_a_rot, patch_b_rot, rot_matrix, components, explained_var = \
                    reorient_patch_pair_with_pca(patch_a, patch_b)
                
                patches[pair_idx] = patch_a_rot
                patches[pair_idx + 1] = patch_b_rot
                
                print(f"  Pair {pair_idx//2 + 1} ({labels[pair_idx]}, {labels[pair_idx + 1]}):")
                print(f"    Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}, PC3={explained_var[2]:.3f}")
                print(f"    Patches now have same orientation (aligned to coordinate axes)")
        else:
            print(f"\nWarning: PCA reorientation requires even number of patches (got {num_patches}). Skipping.")
    
    # Center patch pairs at (0,0,0) if requested
    # Assuming patches are paired: 0-1, 2-3, etc. Single patch: center that patch.
    if centralize_centroid:
        if num_patches == 1:
            centroid = np.mean(patches[0], axis=0)
            patches[0] = patches[0] - centroid
            print(f"\nCentering single patch at (0,0,0): centroid was at ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), now at (0.00, 0.00, 0.00)")
        elif num_patches >= 2 and num_patches % 2 == 0:
            print("\nCentering patch pairs at (0,0,0):")
            for pair_idx in range(0, num_patches, 2):
                patch_a = patches[pair_idx]
                patch_b = patches[pair_idx + 1]
                
                # Calculate combined centroid of the pair
                combined_points = np.vstack([patch_a, patch_b])
                centroid = np.mean(combined_points, axis=0)
                
                # Translate both patches so centroid is at (0,0,0)
                patches[pair_idx] = patch_a - centroid
                patches[pair_idx + 1] = patch_b - centroid
                
                print(f"  Pair {pair_idx//2 + 1} ({labels[pair_idx]}, {labels[pair_idx + 1]}): "
                      f"centroid was at ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
                      f"now at (0.00, 0.00, 0.00)")
        elif num_patches >= 2:
            print(f"\nWarning: Odd number of patches ({num_patches}). Centering all patches together at (0,0,0):")
            all_points = np.vstack(patches)
            centroid = np.mean(all_points, axis=0)
            patches = [patch - centroid for patch in patches]
            print(f"  Combined centroid was at ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), "
                  f"now at (0.00, 0.00, 0.00)")
    else:
        print("\nCentroid centralization disabled - patches shown in original coordinates")
    
    # Determine which visualization to use
    # Priority: use_matplotlib > use_open3d > auto-detect (Open3D if available)
    if use_matplotlib:
        use_open3d_actual = False
    elif use_open3d is None:
        # Auto-detect: use Open3D if available
        use_open3d_actual = OPEN3D_AVAILABLE
    else:
        use_open3d_actual = use_open3d and OPEN3D_AVAILABLE
    
    if use_open3d_actual:
        visualize_patches_open3d(patches, labels, point_size, save_ply)
        return
    
    if use_open3d and not OPEN3D_AVAILABLE:
        print("Warning: Open3D requested but not available. Falling back to matplotlib.")
    
    # Create figure with single combined view
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for each patch: blue, red, green, yellow
    patch_colors = ['blue', 'red', 'green', 'yellow']
    color_names = ['blue', 'red', 'green', 'yellow']
    
    # Plot all patches in combined view
    for i, patch in enumerate(patches):
        color = patch_colors[i % len(patch_colors)]
        ax.scatter(patch[:, 0], patch[:, 1], patch[:, 2], 
                  c=color, s=10, alpha=0.6, label=labels[i],
                  edgecolors='none')
    
    # Remove axes labels and hide axes
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Hide the axes panes (background planes)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Hide the axis lines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # Remove grid
    ax.grid(False)
    
    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Create title
    title_parts = [f"{labels[i]} ({color_names[i % len(color_names)]})" for i in range(num_patches)]
    ax.set_title('Combined Point Cloud: ' + ', '.join(title_parts))
    ax.legend()
    
    # Set equal aspect ratio
    all_points = np.vstack(patches)
    x_range = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
    y_range = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
    z_range = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]
    
    # Set equal aspect ratio
    max_range = np.array([x_range[1] - x_range[0],
                         y_range[1] - y_range[0],
                         z_range[1] - z_range[0]]).max() / 2.0
    mid_x = (x_range[0] + x_range[1]) * 0.5
    mid_y = (y_range[0] + y_range[1]) * 0.5
    mid_z = (z_range[0] + z_range[1]) * 0.5
    
    # Store initial limits for zoom reset
    initial_xlim = (mid_x - max_range, mid_x + max_range)
    initial_ylim = (mid_y - max_range, mid_y + max_range)
    initial_zlim = (mid_z - max_range, mid_z + max_range)
    
    ax.set_xlim(initial_xlim)
    ax.set_ylim(initial_ylim)
    ax.set_zlim(initial_zlim)
    
    # Add mouse wheel zoom support
    def on_scroll(event):
        """Handle mouse wheel zoom."""
        if event.inaxes != ax:
            return
        
        # Get current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        
        # Calculate zoom factor (scroll up = zoom in, scroll down = zoom out)
        zoom_factor = 1.1 if event.button == 'up' else 0.9
        
        # Calculate centers
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        # Calculate new ranges
        x_range_new = (xlim[1] - xlim[0]) * zoom_factor
        y_range_new = (ylim[1] - ylim[0]) * zoom_factor
        z_range_new = (zlim[1] - zlim[0]) * zoom_factor
        
        # Set new limits
        ax.set_xlim(x_center - x_range_new/2, x_center + x_range_new/2)
        ax.set_ylim(y_center - y_range_new/2, y_center + y_range_new/2)
        ax.set_zlim(z_center - z_range_new/2, z_center + z_range_new/2)
        
        fig.canvas.draw()
    
    # Connect scroll event
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
    
    if show:
        print("\n" + "="*70)
        print("Matplotlib 3D Point Cloud Viewer")
        print("="*70)
        print("Controls:")
        print("  - Mouse drag: Rotate view")
        print("  - Mouse wheel: Zoom in/out")
        print("  - Right-click + drag: Pan")
        print("  - Close window to exit")
        print("="*70 + "\n")
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize multiple numpy patches as 3D point clouds')
    parser.add_argument('patches', type=str, nargs='+', help='Paths to numpy patch files (1-4 patches)')
    parser.add_argument('--labels', type=str, nargs='+', default=None, help='Labels for each patch (optional)')
    parser.add_argument('--save', type=str, default=None, help='Path to save the visualization (matplotlib only)')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    parser.add_argument('--open3d', action='store_true', help='Force use of Open3D for interactive point cloud visualization (default if available)')
    parser.add_argument('--matplotlib', action='store_true', help='Force use of matplotlib instead of Open3D')
    parser.add_argument('--point-size', type=float, default=2.0, help='Point size for Open3D visualization')
    parser.add_argument('--save-ply', type=str, default=None, help='Save point clouds as PLY files (prefix, Open3D only)')
    parser.add_argument('--scale', type=float, nargs='+', default=None, 
                       help='Scale factors for each patch (e.g., --scale 1.0 1.0 0.001 0.001 to scale patches 3 and 4 by 0.001)')
    parser.add_argument('--pca-reorient', action='store_true', 
                       help='Use PCA to reorient patch pairs so both patches in each pair have the same orientation (aligned to coordinate axes)')
    parser.add_argument('--no-centralize', action='store_true',
                       help='Disable centroid centralization (patches shown in original coordinates)')
    
    args = parser.parse_args()
    
    # Load patches (may split combined preprocessed pairs)
    patches = []
    labels_from_files = []
    for patch_path in args.patches:
        loaded = load_patch(patch_path)
        # If load_patch returned a list (split preprocessed pair), extend patches list
        if isinstance(loaded, list):
            patches.extend(loaded)
            # Generate labels for split patches
            base_name = os.path.basename(patch_path).replace('.npy', '')
            labels_from_files.append(f"{base_name}_A")
            labels_from_files.append(f"{base_name}_B")
        else:
            patches.append(loaded)
            labels_from_files.append(os.path.basename(patch_path).replace('.npy', ''))
    
    # Validate number of patches
    if len(patches) < 1 or len(patches) > 4:
        raise ValueError(f"Please provide 1-4 patch files (got {len(patches)} after loading)")
    
    # Validate shapes
    for i, patch in enumerate(patches):
        if patch.shape[1] != 3:
            raise ValueError(f"Patch {i+1} must have shape (N, 3) for XYZ coordinates, got {patch.shape}")
    
    # Determine labels
    if args.labels:
        if len(args.labels) != len(patches):
            raise ValueError(f"Number of labels ({len(args.labels)}) must match number of patches ({len(patches)})")
        labels = args.labels
    else:
        # Use generated labels (accounting for split patches)
        labels = labels_from_files
    
    # Print statistics
    for i, (patch, label) in enumerate(zip(patches, labels)):
        print_patch_stats(patch, label)
    
    # Visualize
    visualize_patches(patches, 
                     labels=labels,
                     save_path=args.save,
                     show=not args.no_show,
                     use_open3d=args.open3d if args.open3d else None,
                     use_matplotlib=args.matplotlib,
                     point_size=args.point_size,
                     save_ply=args.save_ply,
                     scales=args.scale,
                     use_pca_reorientation=args.pca_reorient,
                     centralize_centroid=not args.no_centralize)


if __name__ == '__main__':
    main()
