#!/usr/bin/env python3
"""
Given a PLY (or OBJ) point cloud and a predictions JSON, pick the best "good" pair:
  - Primary: highest quality score (P(Good)).
  - Tie-break: if multiple good predictions have the same highest P(Good), pick the pair
    whose individual patch centroids are closest to the object centroid (minimize
    distance(patch1_center, centroid) + distance(patch2_center, centroid)).

Usage:
  python pick_best_pair_centroid.py --ply ape.ply --predictions predictions_ape.json [--patch_dir ape_patches]
  python pick_best_pair_centroid.py --ply ape.ply --predictions predictions_ape.json --visualize
"""

import os
import json
import argparse
import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def load_ply_xyz(filepath):
    """Load PLY file (ASCII) and return points as Nx3 array."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header_end = 0
    num_vertices = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
        elif line == 'end_header':
            header_end = i + 1
            break
    points = []
    for line in lines[header_end : header_end + num_vertices]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(points, dtype=np.float64)


def load_obj_xyz(filepath):
    """Load OBJ file and return vertices as Nx3 array."""
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(points, dtype=np.float64) if points else np.empty((0, 3), dtype=np.float64)


def load_pointcloud_xyz(filepath):
    """Load point cloud from PLY or OBJ; returns Nx3 array."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Point cloud not found: {filepath}")
    path_lower = filepath.lower()
    if path_lower.endswith('.obj'):
        return load_obj_xyz(filepath)
    if path_lower.endswith('.ply'):
        return load_ply_xyz(filepath)
    with open(filepath, 'r') as f:
        first = f.readline().strip()
    if first == 'ply':
        return load_ply_xyz(filepath)
    return load_obj_xyz(filepath)


def visualize_grasp_on_ply(points, center1, center2, ee_position=None, point_size=2.0):
    """
    Visualize the picked grasp on the point cloud:
    gray point cloud, red sphere at patch 1 center, blue sphere at patch 2 center,
    green line between patches, and (optionally) a green sphere at the end-effector
    position with an orange approach line from the end-effector to the midpoint.
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D is required for visualization. Install with: pip install open3d")
        return
    if 'XDG_SESSION_TYPE' not in os.environ or os.environ.get('XDG_SESSION_TYPE') == 'wayland':
        os.environ['XDG_SESSION_TYPE'] = 'x11'
    # Scale sphere radius to point cloud extent so spheres are visible
    extent = float(np.ptp(points, axis=0).max())
    radius = max(extent * 0.015, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    # Estimate normals for shading (optional)
    pcd.estimate_normals()

    red_pt = np.asarray(center1, dtype=np.float64).reshape(3)
    blue_pt = np.asarray(center2, dtype=np.float64).reshape(3)

    red_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    red_sphere.translate(red_pt)
    red_sphere.paint_uniform_color([1, 0, 0])

    blue_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    blue_sphere.translate(blue_pt)
    blue_sphere.paint_uniform_color([0, 0, 1])

    midpoint = (red_pt + blue_pt) * 0.5

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array([red_pt, blue_pt]))
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

    geometries = [pcd, red_sphere, blue_sphere, line_set]

    if ee_position is not None:
        ee_pt = np.asarray(ee_position, dtype=np.float64).reshape(3)

        ee_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        ee_sphere.translate(ee_pt)
        ee_sphere.paint_uniform_color([0, 0.8, 0])

        approach_line = o3d.geometry.LineSet()
        approach_line.points = o3d.utility.Vector3dVector(np.array([ee_pt, midpoint]))
        approach_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        approach_line.colors = o3d.utility.Vector3dVector([[1, 0.5, 0]])

        geometries += [ee_sphere, approach_line]
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Best grasp (centroid)", width=1200, height=800)
    for geom in geometries:
        vis.add_geometry(geom)
    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = np.array([1, 1, 1])
    print("\nControls: Left=rotate, Right=pan, Wheel=zoom, R=reset, Q=quit")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Pick the good prediction pair; tie-break by sum of patch-center distances to object centroid"
    )
    parser.add_argument("--ply", type=str, required=True,
                        help="Path to object point cloud (PLY or OBJ)")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON (e.g. predictions_ape.json)")
    parser.add_argument("--patch_dir", type=str, default=None,
                        help="Directory containing patch_metadata.json (default: from predictions JSON)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: save best pair info to this JSON file")
    parser.add_argument("--visualize", action="store_true",
                        help="Show the picked grasp on the PLY (Open3D: red/blue spheres, green line)")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Point size for visualization (default: 2.0)")
    args = parser.parse_args()

    # Load point cloud and compute centroid (center of mass)
    points = load_pointcloud_xyz(args.ply)
    centroid = np.mean(points, axis=0)
    ply_scale = float(np.ptp(points, axis=0).max())
    print(f"Loaded {args.ply}: {len(points)} points")
    print(f"Object centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")

    # Load predictions
    with open(args.predictions, "r") as f:
        data = json.load(f)
    pairs = data.get("pairs", [])
    good_pairs = [p for p in pairs if p.get("prediction", 1) == 0]
    if not good_pairs:
        print("No good predictions in the JSON. Exiting.")
        return

    print(f"Found {len(good_pairs)} good prediction(s)")

    # Resolve patch_dir and load patch centers
    pred_dir = os.path.dirname(os.path.abspath(args.predictions))
    patch_dir = args.patch_dir or data.get("patch_dir") or ""
    if not os.path.isabs(patch_dir):
        patch_dir = os.path.join(pred_dir, patch_dir)
    meta_path = os.path.join(patch_dir, "patch_metadata.json")
    if not os.path.isfile(meta_path):
        # Try common alternative: {object_id}_patches next to predictions
        object_id = data.get("object_id", "")
        if object_id:
            alt_patch_dir = os.path.join(pred_dir, f"{object_id}_patches")
            alt_meta = os.path.join(alt_patch_dir, "patch_metadata.json")
            if os.path.isfile(alt_meta):
                patch_dir = alt_patch_dir
                meta_path = alt_meta
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Patch metadata not found: {meta_path} (use --patch_dir e.g. obj_000001_patches)")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    center_coords = metadata["center_coordinates"]
    n_centers = len(center_coords)
    # Only keep good pairs whose patch indices exist in this metadata
    def valid_indices(p):
        i1, i2 = p.get("patch_idx1"), p.get("patch_idx2")
        return (i1 is not None and i2 is not None
                and 0 <= i1 < n_centers and 0 <= i2 < n_centers)
    n_before = len(good_pairs)
    good_pairs = [p for p in good_pairs if valid_indices(p)]
    if n_before > len(good_pairs):
        print(f"Skipped {n_before - len(good_pairs)} good pair(s) with patch indices out of range [0, {n_centers}).")
    if not good_pairs:
        print("No good pairs with valid patch indices in this patch metadata.")
        return

    # Warn if PLY and patch metadata are in different coordinate systems (e.g. different objects or units)
    sample_center = np.array(center_coords[0])
    patch_scale = float(np.ptp(np.array(center_coords), axis=0).max())
    if ply_scale > 1e-6 and patch_scale > 1e-6:
        ratio = max(ply_scale / patch_scale, patch_scale / ply_scale)
        if ratio > 5:
            print("\n*** WARNING: PLY and patch centers use very different scales (e.g. different objects or units).")
            print(f"    PLY extent ~ {ply_scale:.2f}, patch coords extent ~ {patch_scale:.2f}")
            print("    Use the PLY that matches the predictions (same object and coordinate system as patch_dir).")
            print("***\n")

    # Primary: highest quality (prob_good). Tie-break: among ties, pick pair whose individual patch centroids are closest to object centroid (min d1+d2).
    max_prob = max(p.get("prob_good", 0) for p in good_pairs)
    ties = [p for p in good_pairs if p.get("prob_good", 0) == max_prob]
    best_pair = None
    best_score = float("inf")
    best_centers = None

    for p in ties:
        idx1, idx2 = p["patch_idx1"], p["patch_idx2"]
        c1 = np.array(center_coords[idx1])
        c2 = np.array(center_coords[idx2])
        d1 = np.linalg.norm(c1 - centroid)
        d2 = np.linalg.norm(c2 - centroid)
        score = d1 + d2
        if score < best_score:
            best_score = score
            best_pair = p
            best_centers = (c1, c2)

    if best_pair is None:
        print("No pair selected.")
        return

    c1, c2 = best_centers
    d1 = np.linalg.norm(c1 - centroid)
    d2 = np.linalg.norm(c2 - centroid)

    # ------------------------------------------------------------------ #
    # End-effector position: perpendicular bisector of c1-c2, outside mesh
    # ------------------------------------------------------------------ #
    midpoint = (c1 + c2) * 0.5
    jaw_axis = c2 - c1
    jaw_len = np.linalg.norm(jaw_axis)
    if jaw_len > 1e-9:
        jaw_axis = jaw_axis / jaw_len

    # Approach direction: perpendicular to jaw axis, pointing away from centroid.
    # Project (midpoint - centroid) onto the plane perpendicular to jaw_axis.
    outward = midpoint - centroid
    outward = outward - np.dot(outward, jaw_axis) * jaw_axis
    outward_norm = np.linalg.norm(outward)
    if outward_norm < 1e-9:
        # Fallback: any vector perpendicular to jaw_axis
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, jaw_axis)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        outward = fallback - np.dot(fallback, jaw_axis) * jaw_axis
        outward_norm = np.linalg.norm(outward)
    approach_dir = outward / outward_norm

    # Find how far the object extends from midpoint along approach_dir,
    # then place the end-effector just beyond that surface.
    projections = (points - midpoint) @ approach_dir
    max_proj = float(projections.max())
    margin = float(np.ptp(points, axis=0).max()) * 0.05  # 5% of object extent
    ee_position = midpoint + (max_proj + margin) * approach_dir

    print("\n" + "=" * 60)
    print("Best good pair (highest P(Good); tie-break: individual patch centroids closest to object centroid)")
    print("=" * 60)
    if len(ties) > 1:
        print(f"  (Tie-break among {len(ties)} pairs with P(Good) = {max_prob:.6f})")
    print(f"  Patch indices:  {best_pair['patch_idx1']}, {best_pair['patch_idx2']}")
    print(f"  P(Good):        {best_pair.get('prob_good', 0):.6f}")
    print(f"  Patch 1 center: [{c1[0]:.4f}, {c1[1]:.4f}, {c1[2]:.4f}]  dist to obj centroid: {d1:.4f}")
    print(f"  Patch 2 center: [{c2[0]:.4f}, {c2[1]:.4f}, {c2[2]:.4f}]  dist to obj centroid: {d2:.4f}")
    print(f"  Sum of distances to centroid: {best_score:.4f}")
    print(f"  Midpoint:       [{midpoint[0]:.4f}, {midpoint[1]:.4f}, {midpoint[2]:.4f}]")
    print(f"  Approach dir:   [{approach_dir[0]:.4f}, {approach_dir[1]:.4f}, {approach_dir[2]:.4f}]")
    print(f"  EE position:    [{ee_position[0]:.4f}, {ee_position[1]:.4f}, {ee_position[2]:.4f}]")
    print("=" * 60)

    if args.output:
        out = {
            "ply": args.ply,
            "predictions": args.predictions,
            "object_centroid": centroid.tolist(),
            "best_pair": {
                "patch_idx1": best_pair["patch_idx1"],
                "patch_idx2": best_pair["patch_idx2"],
                "prob_good": best_pair.get("prob_good"),
                "patch1_center": c1.tolist(),
                "patch2_center": c2.tolist(),
                "dist_patch1_to_centroid": float(d1),
                "dist_patch2_to_centroid": float(d2),
                "sum_dist_to_centroid": float(best_score),
                "midpoint": midpoint.tolist(),
                "approach_direction": approach_dir.tolist(),
                "ee_position": ee_position.tolist(),
            },
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved best pair info to {args.output}")

    if args.visualize:
        visualize_grasp_on_ply(points, c1, c2, ee_position=ee_position, point_size=args.point_size)


if __name__ == "__main__":
    main()
