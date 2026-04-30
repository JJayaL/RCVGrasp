# RCVGrasp - Zero-Shot Grasping from Local Surface Geometry

This repository contains code for **zero-shot robotic grasping** using point-cloud patch pairs. The system learns to classify pairs of 3D patches as **graspable** (good) or **non-graspable** (bad) using a PointNet++-based binary classifier trained on synthetic data. At inference time, patches from a novel scene can be paired and scored to select the best grasp.
## Overview
- **Input:** Pairs of XYZ point-cloud patches (e.g., 32×32 = 1024 points per patch), representing two local regions (e.g., from two camera views or two parallel planes).
- **Output:** Binary label (good / bad) or score for each pair; “good” pairs correspond to promising grasp configurations.
- **Model:** PointNet2 (Single-Scale Grouping, XYZ only) for binary classification, implemented via the [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) codebase.
The pipeline has three main stages:
1. **Synthetic data generation** — depth maps and XYZ point clouds for graspable “good” and non-graspable “bad” sheet/plane configurations.
2. **Patch extraction** — 32×32 patches from point clouds, saved as `.npy` files and organized into `good_patches/` and `bad_patches/`.
3. **Training and inference** — train the PointNet2 classifier on patch pairs, then run inference on new patches (e.g., from a real robot scene) to pick the best pair.
---
## Repository Structure
| File / Folder | Description |
|---------------|-------------|
| `create_depth_graspable_xyz.py` | Generates synthetic “good” depth maps and XYZ point clouds (e.g., parallel planes 3 cm apart, camera-aligned). |
| `create_depth_nongraspable_xyz.py` | Generates synthetic “bad” depth/XYZ data (configurations that should be classified as not graspable). |
| `create_patch_ascii.py` | Extracts 32×32 patches from point clouds (PLY or ASCII), saves as `patch_*.npy`. |
| `dataloader.py` | Data loading for good/bad patch pairs from `good_patches/` and `bad_patches/`; augmentations (rotation, point dropout, PCA alignment) and train/test split. |
| `train.py` | Training script for PointNet2 binary classification (good vs bad pairs). |
| `test_patch_pairs.py` | Loads a trained model, runs inference on patch pairs (e.g., from `ape_patches/`), and visualizes the best “good” prediction. |
| `visualize_patches.py` | Visualizes one or more `.npy` patches as 3D point clouds (matplotlib/Open3D). |
| `visualize_pairs_from_coords.py` | Uses a PLY and a coordinates file to extract and visualize patch pairs (red/blue points) on the full point cloud. |
| `best_model_bce.pth` | Example trained checkpoint (BCE loss); used by `test_patch_pairs.py` if no other checkpoint is given. |
| `generate_syn_dataset` | (Script or folder for generating the full synthetic dataset; adjust to your layout.) |
| `tests/` | Unit or integration tests. |
---
## Dependencies
- **Python 3** (tested with 3.10)
- **PyTorch** (with CUDA if available)
- **Pointnet_Pointnet2_pytorch** — clone or add as submodule so that `Pointnet_Pointnet2_pytorch/models/pointnet2_cls_ssg.py` is on the path (see below).
- **NumPy**, **OpenCV** (`cv2`), **Matplotlib**, **scikit-learn** (for PCA in dataloader), **tqdm**, **seaborn**
- **Open3D** (optional, for 3D visualization in `test_patch_pairs.py` and `visualize_patches.py`)
- **SciPy** (optional, for `visualize_pairs_from_coords.py`)
Example (after cloning Pointnet_Pointnet2_pytorch):
```bash
pip install torch numpy opencv-python matplotlib scikit-learn tqdm seaborn
pip install open3d scipy  # optional, for visualization
```
### Pointnet_Pointnet2_pytorch
Training and testing assume the PointNet2 implementation is available as:
```
<repo_root>/Pointnet_Pointnet2_pytorch/
  models/
    pointnet2_cls_ssg.py
    pointnet2_utils.py
    ...
```
Clone it into the project root if needed:
```bash
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
```

---
## Data Layout
- **Synthetic generation:**  
  `create_depth_graspable_xyz.py` and `create_depth_nongraspable_xyz.py` produce depth/XYZ outputs (and may use `camera.json` for intrinsics). Organize outputs so that patch extraction can run on them.
- **Patch directories for training:**  
  The dataloader expects a base directory containing:
  - `good_patches/` — pair files named like `good_sheet_*_*.npy` (or the naming convention used by `find_xyz_pairs` in `dataloader.py`).
  - `bad_patches/` — pair files named like `bad_sheet_*_*.npy`.
  Default base path in code is set to a NAS path; override with the `dataset_dir` argument where supported (e.g. in `get_pointnet_dataloaders` in `dataloader.py`).
- **Patch extraction:**  
  `create_patch_ascii.py` reads a point cloud (PLY or ASCII), samples points, extracts 32×32 patches around them, and saves each as `patch_*.npy` in an output folder. Use this to build `good_patches/`, `bad_patches/`, or a directory like `ape_patches/` for inference.
---
## Usage
### 1. Generate synthetic data (good and bad)
Run the depth/XYZ generation scripts (optionally driven by `generate_syn_dataset` if you use it):
```bash
cd gen_synthetic_data
python create_depth_graspable_xyz.py --good-scale --num-pairs 5000   # outputs graspable (good) depth/XYZ
python create_depth_nongraspable_xyz.py --bad --num-pairs 1000   # outputs non-graspable (bad) depth/XYZ
```
**Note:**  
The code organizes outputs into separate folders for each augmentation type. This is intentional to help with understanding and visualizing the generated data, and to support tuning of hyperparameters.
Before using the data for training, you may want to modify the code to remove this folder-wise separation and consolidate the dataset.

Adjust any paths, camera intrinsics (`camera.json`), and output directories inside the scripts as needed.
### 2. Train the model
From the repo root (with `Pointnet_Pointnet2_pytorch` in place):
```bash
python3 train.py --use_pca_alignment --pca_target_plane xy --loss_type bce
or
python train.py --epochs 200 --batch_size 32 --loss_type bce
```
Important options:
- `--dataset_dir` — base path containing `good_patches/` and `bad_patches/` (if not using default).
- `--use_rotation` — enable random Y-axis rotation augmentation.
- `--use_point_dropout` — enable point dropout augmentation.
- `--use_pca_alignment` — align patch pairs with PCA.
- `--loss_type bce` — binary cross-entropy (alternatively `nll`).
- `--save_dir`, `--log_dir` — where to write checkpoints and logs.
Checkpoints and best model (e.g. `best_model.pth`) are written under `save_dir`; logs and plots (loss curves, confusion matrix) under `log_dir`.
### 3. Run inference and visualize best pair
To score all pairs from a directory of patches (e.g. `ape_patches/`) and visualize the best “good” pair:
```bash
python test_patch_pairs.py --patch_dir ape_patches --checkpoint checkpoints/pt_1/best_model.pth
```
If you use the provided `best_model_bce.pth`:
```bash
python test_patch_pairs.py --patch_dir ape_patches --checkpoint best_model_bce.pth
```
### 4. Visualize patches or pairs
- **Single or multiple patches (e.g. preprocessed pair):**

  ```bash
  python visualize_patches.py patch_0.npy [patch_1.npy ...]
  ```

- **Pairs from a PLY and a coordinates file (red/blue points):**
  ```bash
  python visualize_pairs_from_coords.py scene.ply top_3_coordinates.txt
  ```

---

## Model and Training Details

- **Architecture:** PointNet2 (SSG), 2-class (good=0, bad=1), XYZ only (no normals).
- **Input:** Concatenated patch pair of shape `(B, N, 3)` with `N = 2048` (1024 + 1024) after preprocessing; passed to the model as `(B, 3, N)`.
- **Preprocessing:** Centering, unit-sphere normalization, optional PCA alignment, optional random Y rotation and point dropout (training only).
- **Loss:** BCE or NLL; evaluation uses balanced accuracy (average of per-class precision).
- **Outputs:** Checkpoints, loss curves, confusion matrices, and TensorBoard logs (if available).

---

## Tests

The `tests/` directory is reserved for unit or integration tests. Run with your preferred test runner (e.g. `pytest tests/`).
### Extract Patches from an Object Mesh

From each point cloud (or combined good/bad point clouds), extract 32×32 patches:

```bash
python create_patch_ascii.py <pointcloud_file> <output_dir>
```

Then feed in this <output_dir> to test_patch_pairs.py to find the best graspable patch pairs for the given object.


## Citation and License
Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. 2017. PointNet++: deep hierarchical feature learning on point sets in a metric space. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). Curran Associates Inc., Red Hook, NY, USA, 5105–5114.

Pointnet_Pointnet2_pytorch project: https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git

---

## Summary

This repo implements a **synthetic-data → patch-pair → PointNet2** pipeline for zero-shot grasp selection: generate good/bad depth and XYZ, extract 32×32 patches, train a binary classifier on patch pairs, then run `test_patch_pairs.py` on new patches to get the best pair and visualize it. `best_model_bce.pth` is a pretrained checkpoint you can use for inference without retraining.

Access required

You don't have permission to use GitLab Duo Agent Platform in this project. Learn more.
Get access to GitLab Duo
Who can grant you access
Contact a project Maintainer or group Owner
