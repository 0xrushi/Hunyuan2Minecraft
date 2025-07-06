# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

# Apply torchvision compatibility fix before other imports

import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')


try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


import os
import random
import shutil
import subprocess
import time
import multiprocessing
from glob import glob
from pathlib import Path

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uuid
import numpy as np
import PIL.Image

# Import voxelization functions - now using built-in implementation
VOXELIZATION_AVAILABLE = True
try:
    import torch
    import torch.nn.functional as F
    TORCH_VOXEL_AVAILABLE = True
    voxel_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] PyTorch available for voxelization - Using device: {voxel_device}")
except ImportError:
    print("[WARNING] PyTorch not available for advanced voxelization")
    TORCH_VOXEL_AVAILABLE = False
    voxel_device = 'cpu'

# Import PyVista for interactive 3D visualization
try:
    import pyvista as pv
    # Set PyVista to work in headless mode for server environments
    pv.set_plot_theme("document")
    pv.OFF_SCREEN = True  # Enable off-screen rendering for server
    PYVISTA_AVAILABLE = True
    print("[INFO] PyVista available for interactive 3D visualization")
except ImportError:
    print("[WARNING] PyVista not available. Install with: pip install pyvista")
    PYVISTA_AVAILABLE = False

# Import Minecraft Pi for voxel building
try:
    from mcpi.minecraft import Minecraft
    from mcpi import block
    MINECRAFT_AVAILABLE = True
    print("[INFO] Minecraft Pi API available for voxel building")
except ImportError:
    print("[WARNING] Minecraft Pi API not available. Install with: pip install mcpi")
    MINECRAFT_AVAILABLE = False

from hy3dshape.utils import logger
from hy3dpaint.convert_utils import create_glb_with_pbr_materials

# Fixed voxelization functions for proper pyramid generation

import torch
import numpy as np
import trimesh
import time

def ray_triangle_intersection_vectorized(ray_origins, ray_dirs, v0, v1, v2):
    """
    Vectorized ray-triangle intersection using Möller-Trumbore algorithm
    """
    # Compute triangle edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Compute determinant
    h = torch.cross(ray_dirs, edge2, dim=1)
    a = torch.sum(edge1 * h, dim=1)
    
    # Rays parallel to triangle
    parallel_mask = torch.abs(a) < 1e-8
    
    f = 1.0 / (a + 1e-8)  # Add small epsilon for numerical stability
    s = ray_origins - v0.unsqueeze(0)
    u = f * torch.sum(s * h, dim=1)
    
    # Check u bounds
    valid_u = (u >= 0.0) & (u <= 1.0)
    
    q = torch.cross(s, edge1, dim=1)
    v = f * torch.sum(ray_dirs * q, dim=1)
    
    # Check v bounds and u+v bounds
    valid_v = (v >= 0.0) & (u + v <= 1.0)
    
    # Compute t (distance along ray)
    t = f * torch.sum(edge2.unsqueeze(0) * q, dim=1)
    
    # Valid intersection: not parallel, valid barycentric coords, positive t
    valid_intersection = ~parallel_mask & valid_u & valid_v & (t > 1e-8)
    
    return valid_intersection, t


def voxel_triangle_intersection_improved(voxel_centers, voxel_size, v0, v1, v2):
    """
    Improved voxel-triangle intersection test using multiple rays per voxel
    """
    batch_size = voxel_centers.shape[0]
    device = voxel_centers.device
    
    # Generate multiple ray directions (6 axis-aligned + some diagonal)
    ray_directions = torch.tensor([
        [1, 0, 0], [-1, 0, 0],  # X-axis
        [0, 1, 0], [0, -1, 0],  # Y-axis  
        [0, 0, 1], [0, 0, -1],  # Z-axis
        [1, 1, 0], [-1, -1, 0], # XY diagonal
        [1, 0, 1], [-1, 0, -1], # XZ diagonal
        [0, 1, 1], [0, -1, -1], # YZ diagonal
    ], dtype=torch.float32, device=device)
    
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)
    
    # Test intersections for each ray direction
    intersection_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for ray_dir in ray_directions:
        # Ray origins at voxel centers
        ray_origins = voxel_centers
        ray_dirs = ray_dir.unsqueeze(0).expand(batch_size, -1)
        
        # Test intersection
        intersects, t_values = ray_triangle_intersection_vectorized(
            ray_origins, ray_dirs, v0, v1, v2
        )
        
        # Check if intersection point is within voxel bounds
        half_size = voxel_size * 0.5
        intersection_points = ray_origins + ray_dirs * t_values.unsqueeze(1)
        
        # Check bounds
        within_bounds = torch.all(
            torch.abs(intersection_points - voxel_centers) <= half_size, dim=1
        )
        
        valid_intersections = intersects & within_bounds
        intersection_found |= valid_intersections
        
        # Early termination if we found intersections for all voxels
        if intersection_found.all():
            break
    
    return intersection_found


def pytorch_voxelize_mesh_fixed(vertices, faces, resolution, device='cuda'):
    """
    Fixed mesh voxelization with proper geometric intersection testing
    """
    print(f"[INFO] Fixed PyTorch voxelization on {device}")
    
    # Convert to PyTorch tensors
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    # Compute mesh bounds and normalize
    min_bound = torch.min(vertices, dim=0)[0]
    max_bound = torch.max(vertices, dim=0)[0]
    size = max_bound - min_bound
    max_size = torch.max(size)
    
    # Add small padding to ensure mesh fits within voxel grid
    padding = max_size * 0.05
    min_bound -= padding
    max_size += 2 * padding
    
    # Normalize vertices to [0, resolution]
    vertices_norm = (vertices - min_bound) / max_size * resolution
    
    # Create voxel grid
    voxel_grid = torch.zeros((resolution, resolution, resolution), dtype=torch.bool, device=device)
    
    # Generate voxel centers
    coords = torch.arange(resolution, dtype=torch.float32, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    voxel_centers = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    voxel_size = 1.0  # Size of each voxel in normalized coordinates
    
    print(f"[INFO] Processing {faces.shape[0]} triangles...")
    
    # Process triangles in batches for memory efficiency
    batch_size = min(500, faces.shape[0])  # Smaller batches for better stability
    
    for batch_start in range(0, faces.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, faces.shape[0])
        batch_faces = faces[batch_start:batch_end]
        
        print(f"[INFO] Processing triangles {batch_start} to {batch_end-1}")
        
        # Get triangle vertices for this batch
        v0 = vertices_norm[batch_faces[:, 0]]  # (batch_size, 3)
        v1 = vertices_norm[batch_faces[:, 1]]
        v2 = vertices_norm[batch_faces[:, 2]]
        
        # Compute triangle bounding boxes
        tri_min = torch.min(torch.min(v0, v1), v2)  # (batch_size, 3)
        tri_max = torch.max(torch.max(v0, v1), v2)
        
        # For each triangle in the batch
        for i in range(batch_faces.shape[0]):
            tv0, tv1, tv2 = v0[i], v1[i], v2[i]
            tmin, tmax = tri_min[i], tri_max[i]
            
            # Expand bounding box slightly to ensure we don't miss edge cases
            expand = voxel_size * 0.5
            tmin -= expand
            tmax += expand
            
            # Find candidate voxels using expanded bounding box
            candidate_mask = (
                (voxel_centers[:, 0] >= tmin[0]) & (voxel_centers[:, 0] <= tmax[0]) &
                (voxel_centers[:, 1] >= tmin[1]) & (voxel_centers[:, 1] <= tmax[1]) &
                (voxel_centers[:, 2] >= tmin[2]) & (voxel_centers[:, 2] <= tmax[2])
            )
            
            if not candidate_mask.any():
                continue
                
            candidate_centers = voxel_centers[candidate_mask]
            
            # Improved triangle-voxel intersection test
            intersections = voxel_triangle_intersection_improved(
                candidate_centers, voxel_size, tv0, tv1, tv2
            )
            
            # Mark intersecting voxels as occupied
            if intersections.any():
                intersecting_centers = candidate_centers[intersections]
                
                # Convert to integer indices and clamp
                indices = torch.round(intersecting_centers).long()
                indices = torch.clamp(indices, 0, resolution - 1)
                
                # Set voxels to occupied
                voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    
    return voxel_grid.cpu().numpy()


def pytorch_voxelize_solid_fill(vertices, faces, resolution, device='cuda'):
    """
    Solid fill voxelization using ray casting to determine inside/outside
    This is better for closed meshes like pyramids
    """
    print(f"[INFO] Solid fill voxelization on {device}")
    
    # Convert to PyTorch tensors
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    # Normalize mesh
    min_bound = torch.min(vertices, dim=0)[0]
    max_bound = torch.max(vertices, dim=0)[0]
    size = max_bound - min_bound
    max_size = torch.max(size)
    
    # Add padding
    padding = max_size * 0.05
    min_bound -= padding
    max_size += 2 * padding
    
    vertices_norm = (vertices - min_bound) / max_size * (resolution - 1)
    
    # Create voxel grid
    voxel_grid = torch.zeros((resolution, resolution, resolution), dtype=torch.bool, device=device)
    
    # Generate voxel centers
    coords = torch.arange(resolution, dtype=torch.float32, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    voxel_centers = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    # Ray direction (along X-axis)
    ray_dir = torch.tensor([1.0, 0.0, 0.0], device=device)
    
    print(f"[INFO] Ray casting for {voxel_centers.shape[0]} voxels...")
    
    # Process voxels in batches
    voxel_batch_size = 10000
    
    for batch_start in range(0, voxel_centers.shape[0], voxel_batch_size):
        batch_end = min(batch_start + voxel_batch_size, voxel_centers.shape[0])
        batch_centers = voxel_centers[batch_start:batch_end]
        batch_size = batch_centers.shape[0]
        
        # Count intersections for each voxel
        intersection_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
        
        # Ray directions for this batch
        ray_dirs = ray_dir.unsqueeze(0).expand(batch_size, -1)
        
        # Test against all triangles
        for face_idx in range(faces.shape[0]):
            face = faces[face_idx]
            v0, v1, v2 = vertices_norm[face[0]], vertices_norm[face[1]], vertices_norm[face[2]]
            
            # Ray-triangle intersection
            intersects, t_values = ray_triangle_intersection_vectorized(
                batch_centers, ray_dirs, v0, v1, v2
            )
            
            # Only count intersections in positive ray direction
            valid_intersections = intersects & (t_values > 0)
            intersection_counts += valid_intersections.int()
        
        # Odd number of intersections = inside mesh
        inside_mask = (intersection_counts % 2) == 1
        
        # Set corresponding voxels
        if inside_mask.any():
            inside_centers = batch_centers[inside_mask]
            indices = torch.round(inside_centers).long()
            indices = torch.clamp(indices, 0, resolution - 1)
            voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    
    return voxel_grid.cpu().numpy()


def pytorch_mesh_to_voxels_fixed(mesh_path: str, resolution: int = 32, 
                                method: str = 'solid', use_gpu: bool = True) -> np.ndarray:
    """
    Fixed mesh to voxel conversion with multiple methods
    
    Args:
        mesh_path: Path to your 3D file
        resolution: Number of voxels along each axis
        method: 'surface' (surface only), 'solid' (filled interior), 'fast' (conservative)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        A boolean array of shape (resolution, resolution, resolution)
    """
    start_time = time.time()
    
    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print("[INFO] Using CPU for voxelization")
    
    # Load mesh
    print(f"[INFO] Loading mesh: {mesh_path}")
    
    try:
        raw = trimesh.load(mesh_path)
        if isinstance(raw, trimesh.Scene):
            if not raw.geometry:
                raise ValueError(f"No geometry found in scene: {mesh_path}")
            mesh = trimesh.util.concatenate(tuple(raw.geometry.values()))
        else:
            mesh = raw
    except Exception as e:
        raise ValueError(f"Failed to load mesh {mesh_path}: {e}")
    
    print(f"[INFO] Mesh loaded - {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    print(f"[INFO] Mesh bounds: {mesh.bounds}")
    print(f"[INFO] Mesh is watertight: {mesh.is_watertight}")
    
    # Pre-processing for solid fill method
    if method == 'solid' and not mesh.is_watertight:
        print("[WARNING] Mesh is not watertight. Attempting to fix...")
        try:
            mesh.fill_holes()
            mesh.fix_normals()
            print(f"[INFO] Fixed mesh. New watertight status: {mesh.is_watertight}")
        except:
            print("[WARNING] Could not fix mesh. Results may be incomplete.")
    
    # Choose voxelization method
    try:
        if method == 'solid':
            voxel_grid = pytorch_voxelize_solid_fill(mesh.vertices, mesh.faces, resolution, device)
        elif method == 'surface':
            voxel_grid = pytorch_voxelize_mesh_fixed(mesh.vertices, mesh.faces, resolution, device)
        else:  # fast method - use original implementation
            from your_original_code import pytorch_voxelize_fast  # Replace with actual import
            voxel_grid = pytorch_voxelize_fast(mesh.vertices, mesh.faces, resolution, device)
    except Exception as e:
        print(f"[WARNING] PyTorch voxelization failed: {e}, falling back to trimesh")
        # Fallback to trimesh
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        pitch = size.max() / (resolution - 1)
        mesh.apply_translation(-bounds[0])
        vox = mesh.voxelized(pitch)
        voxel_grid = vox.matrix.astype(bool)
    
    # Post-processing: center the voxel grid
    dims = np.array(voxel_grid.shape)
    if np.any(dims > resolution):
        voxel_grid = voxel_grid[
            :min(dims[0], resolution),
            :min(dims[1], resolution),
            :min(dims[2], resolution)
        ]
    
    # Center the voxel block
    final_grid = np.zeros((resolution,)*3, dtype=bool)
    shape = np.array(voxel_grid.shape)
    offset = ((resolution - shape) // 2).astype(int)
    x0, y0, z0 = offset
    final_grid[x0:x0+shape[0], y0:y0+shape[1], z0:z0+shape[2]] = voxel_grid
    
    total_time = time.time() - start_time
    print(f"[INFO] Total processing time: {total_time:.3f}s")
    print(f"[INFO] Voxel grid: {final_grid.shape}, occupied: {final_grid.sum()} voxels")
    
    return final_grid


# Advanced PyTorch-based voxelization functions
def point_in_triangle_vectorized(points, v0, v1, v2):
    """
    Vectorized point-in-triangle test using PyTorch
    points: (N, 3) tensor of points
    v0, v1, v2: (3,) tensors representing triangle vertices
    Returns: (N,) boolean tensor
    """
    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = points - v0.unsqueeze(0)
    
    # Compute dot products
    dot00 = torch.dot(v0v2, v0v2)
    dot01 = torch.dot(v0v2, v0v1)
    dot02 = torch.sum(v0v2.unsqueeze(0) * v0p, dim=1)
    dot11 = torch.dot(v0v1, v0v1)
    dot12 = torch.sum(v0v1.unsqueeze(0) * v0p, dim=1)
    
    # Compute barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-8)  # Add small epsilon for stability
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Check if point is in triangle
    return (u >= 0) & (v >= 0) & (u + v <= 1)


def triangle_aabb_intersection(tri_min, tri_max, voxel_centers, voxel_size):
    """
    Fast AABB (Axis-Aligned Bounding Box) intersection test
    """
    half_size = voxel_size * 0.5
    voxel_min = voxel_centers - half_size
    voxel_max = voxel_centers + half_size
    
    # Check for overlap in all three dimensions
    overlap_x = (voxel_min[:, 0] <= tri_max[0]) & (voxel_max[:, 0] >= tri_min[0])
    overlap_y = (voxel_min[:, 1] <= tri_max[1]) & (voxel_max[:, 1] >= tri_min[1])
    overlap_z = (voxel_min[:, 2] <= tri_max[2]) & (voxel_max[:, 2] >= tri_min[2])
    
    return overlap_x & overlap_y & overlap_z


def pytorch_voxelize_mesh(vertices, faces, resolution, device='cuda'):
    """
    Fast GPU-based mesh voxelization using PyTorch with proper triangle intersection
    """
    print(f"[INFO] PyTorch accurate voxelization on {device}")
    
    # Convert to PyTorch tensors
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    # Compute mesh bounds
    min_bound = torch.min(vertices, dim=0)[0]
    max_bound = torch.max(vertices, dim=0)[0]
    size = max_bound - min_bound
    max_size = torch.max(size)
    
    # Normalize vertices
    vertices_norm = (vertices - min_bound) / max_size * (resolution - 1)
    
    # Create voxel grid
    voxel_grid = torch.zeros((resolution, resolution, resolution), dtype=torch.bool, device=device)
    
    # Generate all voxel centers
    coords = torch.arange(resolution, dtype=torch.float32, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    voxel_centers = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    
    # Process triangles in batches for memory efficiency
    batch_size = min(1000, faces.shape[0])
    
    for batch_start in range(0, faces.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, faces.shape[0])
        batch_faces = faces[batch_start:batch_end]
        
        # Get triangle vertices for this batch
        v0 = vertices_norm[batch_faces[:, 0]]  # (batch_size, 3)
        v1 = vertices_norm[batch_faces[:, 1]]
        v2 = vertices_norm[batch_faces[:, 2]]
        
        # Compute triangle bounding boxes
        tri_min = torch.min(torch.min(v0, v1), v2)  # (batch_size, 3)
        tri_max = torch.max(torch.max(v0, v1), v2)
        
        # For each triangle in the batch
        for i in range(batch_faces.shape[0]):
            # Get triangle vertices
            tv0, tv1, tv2 = v0[i], v1[i], v2[i]
            tmin, tmax = tri_min[i], tri_max[i]
            
            # Find voxels that might intersect with triangle AABB
            candidates = triangle_aabb_intersection(
                tmin, tmax, voxel_centers, 1.0
            )
            
            if torch.sum(candidates) == 0:
                continue
                
            candidate_centers = voxel_centers[candidates]
            
            # More precise triangle-voxel intersection test
            # Use triangle normal to determine dominant projection plane
            normal = torch.cross(tv1 - tv0, tv2 - tv0)
            abs_normal = torch.abs(normal)
            
            if abs_normal[2] >= abs_normal[0] and abs_normal[2] >= abs_normal[1]:
                # Project to XY plane
                inside = point_in_triangle_vectorized(
                    candidate_centers[:, :2], tv0[:2], tv1[:2], tv2[:2]
                )
            elif abs_normal[1] >= abs_normal[0]:
                # Project to XZ plane
                inside = point_in_triangle_vectorized(
                    candidate_centers[:, [0, 2]], tv0[[0, 2]], tv1[[0, 2]], tv2[[0, 2]]
                )
            else:
                # Project to YZ plane
                inside = point_in_triangle_vectorized(
                    candidate_centers[:, [1, 2]], tv0[[1, 2]], tv1[[1, 2]], tv2[[1, 2]]
                )
            
            # Mark voxels as occupied
            if torch.sum(inside) > 0:
                inside_centers = candidate_centers[inside]
                indices = inside_centers.long()
                # Clamp indices to valid range
                indices = torch.clamp(indices, 0, resolution - 1)
                voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    
    return voxel_grid.cpu().numpy()


def pytorch_voxelize_fast(vertices, faces, resolution, device='cuda'):
    """
    Fast approximate voxelization using conservative rasterization approach
    """
    print(f"[INFO] PyTorch fast voxelization on {device}")
    
    # Convert to PyTorch tensors
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    # Normalize vertices to [0, resolution-1]
    min_bound = torch.min(vertices, dim=0)[0]
    max_bound = torch.max(vertices, dim=0)[0]
    size = max_bound - min_bound
    max_size = torch.max(size)
    
    vertices_norm = (vertices - min_bound) / max_size * (resolution - 1)
    
    # Create voxel grid
    voxel_grid = torch.zeros((resolution, resolution, resolution), dtype=torch.bool, device=device)
    
    # Get triangle vertices
    v0 = vertices_norm[faces[:, 0]]
    v1 = vertices_norm[faces[:, 1]]
    v2 = vertices_norm[faces[:, 2]]
    
    # Compute triangle bounding boxes
    tri_min = torch.floor(torch.min(torch.min(v0, v1), v2)).long()
    tri_max = torch.ceil(torch.max(torch.max(v0, v1), v2)).long()
    
    # Clamp to grid bounds
    tri_min = torch.clamp(tri_min, 0, resolution - 1)
    tri_max = torch.clamp(tri_max, 0, resolution - 1)
    
    # For each triangle, fill its bounding box (conservative approximation)
    for i in range(faces.shape[0]):
        x_min, y_min, z_min = tri_min[i]
        x_max, y_max, z_max = tri_max[i]
        
        # Fill bounding box (conservative voxelization)
        voxel_grid[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = True
    
    return voxel_grid.cpu().numpy()


def plot_all_height_slices(voxels: np.ndarray, title: str = "Voxel Grid", 
                          max_cols: int = 8, save_path: str = None):
    """
    Plot 2D slices for every height level (Y-axis) from y=0 to y=height
    """
    height = voxels.shape[1]  # Y dimension
    
    # Filter out empty slices to save space
    non_empty_slices = []
    for y in range(height):
        if voxels[:, y, :].sum() > 0:  # XZ slice at height y
            non_empty_slices.append(y)
    
    if not non_empty_slices:
        print("[WARNING] No occupied voxels found in any height slice")
        return None
    
    print(f"[INFO] Plotting {len(non_empty_slices)} non-empty height slices out of {height} total")
    
    # Calculate subplot grid dimensions
    n_slices = len(non_empty_slices)
    n_cols = min(max_cols, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols
    
    # Create figure with appropriate size
    fig_width = min(20, n_cols * 2.5)
    fig_height = min(24, n_rows * 2.5)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        fig.suptitle(f'{title} - Height Slices (Y=0 to Y={height-1})', fontsize=16)
        
        # Handle single subplot case
        if n_slices == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot each non-empty slice
        for idx, y in enumerate(non_empty_slices):
            ax = axes[idx]
            
            # Get XZ slice at height y
            slice_data = voxels[:, y, :]  # Shape: (X, Z)
            
            # Plot the slice
            im = ax.imshow(slice_data.T, origin='lower', cmap='Blues', 
                          interpolation='nearest', aspect='equal')
            
            ax.set_title(f'Y={y} ({slice_data.sum()} voxels)')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            
            # Add grid for better visualization
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(non_empty_slices), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save to numpy array
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to numpy array
        import PIL.Image
        img = PIL.Image.open(buf)
        return np.array(img)
        
    except Exception as e:
        print(f"Error creating height slices plot: {e}")
        return None


def plot_height_slices_detailed(voxels: np.ndarray, title: str = "Voxel Grid"):
    """
    Plot detailed height slices with analysis
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        height = voxels.shape[1]
        
        # Get non-empty slices
        non_empty_data = []
        for y in range(height):
            slice_data = voxels[:, y, :]
            if slice_data.sum() > 0:
                non_empty_data.append((y, slice_data))
        
        if not non_empty_data:
            print(f"[WARNING] No occupied voxels found")
            return None
        
        print(f"[INFO] Creating detailed analysis for {len(non_empty_data)} height slices")
        
        # Create summary statistics
        heights = [y for y, _ in non_empty_data]
        voxel_counts = [slice_data.sum() for _, slice_data in non_empty_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot voxel count vs height
        ax1.plot(heights, voxel_counts, 'b-o', markersize=4)
        ax1.set_xlabel('Height (Y)')
        ax1.set_ylabel('Number of Occupied Voxels')
        ax1.set_title('Voxel Count vs Height')
        ax1.grid(True, alpha=0.3)
        
        # Plot center slice
        center_y = height // 2
        center_slice = voxels[:, center_y, :]
        im = ax2.imshow(center_slice.T, origin='lower', cmap='viridis')
        ax2.set_title(f'Center Slice (Y={center_y})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        
        # Save to numpy array
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to numpy array
        import PIL.Image
        img = PIL.Image.open(buf)
        return np.array(img)
        
    except Exception as e:
        print(f"Error creating detailed plot: {e}")
        return None


MAX_SEED = 1e7
ENV = "Local" # "Huggingface"
if ENV == 'Huggingface':
    """
    Setup environment for running on Huggingface platform.

    This block performs the following:
    - Changes directory to the differentiable renderer folder and runs a shell 
        script to compile the mesh painter.
    - Installs a custom rasterizer wheel package via pip.

    Note:
        This setup assumes the script is running in the Huggingface environment 
        with the specified directory structure.
    """
    import os, spaces, subprocess, sys, shlex
    print("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    os.system("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    print('install custom')
    subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"),
                   check=True)
else:
    """
    Define a dummy `spaces` module with a GPU decorator class for local environment.

    The GPU decorator is a no-op that simply returns the decorated function unchanged.
    This allows code that uses the `spaces.GPU` decorator to run without modification locally.
    """
    class spaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration
            def __call__(self, func):
                return func 

def get_example_img_list():
    """
    Load and return a sorted list of example image file paths.

    Searches recursively for PNG images under the './assets/example_images/' directory.

    Returns:
        list[str]: Sorted list of file paths to example PNG images.
    """
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list():
    """
    Load and return a list of example text prompts.

    Reads lines from the './assets/example_prompts.txt' file, stripping whitespace.

    Returns:
        list[str]: List of example text prompts.
    """
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def find_latest_mesh_file(save_dir):
    """
    Find the latest generated mesh file (OBJ or STL) in the save directory.
    
    Args:
        save_dir: Directory to search for mesh files
        
    Returns:
        str: Path to the latest mesh file, or None if not found
    """
    if not os.path.exists(save_dir):
        return None
    
    # Look for mesh files in subdirectories
    mesh_files = []
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            if file.endswith(('.obj', '.stl')):
                file_path = os.path.join(root, file)
                # Get file modification time
                mtime = os.path.getmtime(file_path)
                mesh_files.append((file_path, mtime))
    
    if not mesh_files:
        return None
    
    # Sort by modification time (newest first)
    mesh_files.sort(key=lambda x: x[1], reverse=True)
    return mesh_files[0][0]


def gen_save_folder(max_size=200):
    """
    Generate a new save folder inside SAVE_DIR, maintaining a maximum number of folders.

    If the number of existing folders in SAVE_DIR exceeds `max_size`, the oldest folder is removed.

    Args:
        max_size (int, optional): Maximum number of folders to keep in SAVE_DIR. Defaults to 200.

    Returns:
        str: Path to the newly created save folder.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")
    return new_folder


# Removed complex PBR conversion functions - using simple trimesh-based conversion
def export_mesh(mesh, save_folder, textured=False, type='glb'):
    """
    Export a mesh to a file in the specified folder, optionally including textures.

    Args:
        mesh (trimesh.Trimesh): The mesh object to export.
        save_folder (str): Directory path where the mesh file will be saved.
        textured (bool, optional): Whether to include textures/normals in the export. Defaults to False.
        type (str, optional): File format to export ('glb' or 'obj' supported). Defaults to 'glb'.

    Returns:
        str: The full path to the exported mesh file.
    """
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path




def quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> bool:
    # 执行转换
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)
            


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', f'{related_path}/')
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" \
height="{height}" width="100%" frameborder="0"></iframe>'
    print(f'Find html file {output_html_path}, \
{os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

@spaces.GPU(duration=60)
def _gen_shape(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    if not MV_MODE and image is None and caption is None:
        raise gr.Error("Please provide either a caption or an image.")
    if MV_MODE:
        if mv_image_front is None and mv_image_back is None \
            and mv_image_left is None and mv_image_right is None:
            raise gr.Error("Please provide at least one view image.")
        image = {}
        if mv_image_front:
            image['front'] = mv_image_front
        if mv_image_back:
            image['back'] = mv_image_back
        if mv_image_left:
            image['left'] = mv_image_left
        if mv_image_right:
            image['right'] = mv_image_right

    seed = int(randomize_seed_fn(seed, randomize_seed))

    octree_resolution = int(octree_resolution)
    if caption: print('prompt is', caption)
    save_folder = gen_save_folder()
    stats = {
        'model': {
            'shapegen': f'{args.model_path}/{args.subfolder}',
            'texgen': f'{args.texgen_model_path}',
        },
        'params': {
            'caption': caption,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
        }
    }
    time_meta = {}

    if image is None:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise gr.Error(f"Text to 3D is disable. \
            Please enable it by `python gradio_app.py --enable_t23d`.")
        time_meta['text2image'] = time.time() - start_time

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'input.png'))
    if MV_MODE:
        start_time = time.time()
        for k, v in image.items():
            if check_box_rembg or v.mode == "RGB":
                img = rmbg_worker(v.convert('RGB'))
                image[k] = img
        time_meta['remove background'] = time.time() - start_time
    else:
        if check_box_rembg or image.mode == "RGB":
            start_time = time.time()
            image = rmbg_worker(image.convert('RGB'))
            time_meta['remove background'] = time.time() - start_time

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'rembg.png'))

    # image to white model
    start_time = time.time()

    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    outputs = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh'
    )
    time_meta['shape generation'] = time.time() - start_time
    logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))

    tmp_start = time.time()
    mesh = export_to_trimesh(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    stats['time'] = time_meta
    main_image = image if not MV_MODE else image['front']
    return mesh, main_image, save_folder, stats, seed

@spaces.GPU(duration=60)
def generation_all(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption,
        image,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
    )
    path = export_mesh(mesh, save_folder, textured=False)
    

    print(path)
    print('='*40)

    # tmp_time = time.time()
    # mesh = floater_remove_worker(mesh)
    # mesh = degenerate_face_remove_worker(mesh)
    # logger.info("---Postprocessing takes %s seconds ---" % (time.time() - tmp_time))
    # stats['time']['postprocessing'] = time.time() - tmp_time

    tmp_time = time.time()
    mesh = face_reduce_worker(mesh)

    # path = export_mesh(mesh, save_folder, textured=False, type='glb')
    path = export_mesh(mesh, save_folder, textured=False, type='obj') # 这样操作也会 core dump

    logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['face reduction'] = time.time() - tmp_time

    tmp_time = time.time()

    text_path = os.path.join(save_folder, f'textured_mesh.obj')
    path_textured = tex_pipeline(mesh_path=path, image_path=image, output_mesh_path=text_path, save_glb=False)
        
    logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['texture generation'] = time.time() - tmp_time

    tmp_time = time.time()
    # Convert textured OBJ to GLB using obj2gltf with PBR support
    glb_path_textured = os.path.join(save_folder, 'textured_mesh.glb')
    conversion_success = quick_convert_with_obj2gltf(path_textured, glb_path_textured)

    logger.info("---Convert textured OBJ to GLB takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['convert textured OBJ to GLB'] = time.time() - tmp_time
    stats['time']['total'] = time.time() - start_time_0
    model_viewer_html_textured = build_model_viewer_html(save_folder, 
                                                         height=HTML_HEIGHT, 
                                                         width=HTML_WIDTH, textured=True)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return (
        gr.update(value=path),
        gr.update(value=glb_path_textured),
        model_viewer_html_textured,
        stats,
        seed,
    )

@spaces.GPU(duration=60)
def shape_generation(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed = _gen_shape(
        caption,
        image,
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
    )
    stats['time']['total'] = time.time() - start_time_0
    mesh.metadata['extras'] = stats

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    return (
        gr.update(value=path),
        model_viewer_html,
        stats,
        seed,
    )


def create_standalone_html_voxel_plot(voxels):
    """
    Create a standalone HTML file for local viewing without web server.
    
    Args:
        voxels: 3D boolean array of voxels
    
    Returns:
        str: Path to standalone HTML file or None if failed
    """
    if not PYVISTA_AVAILABLE:
        return None
    
    try:
        if not voxels.any():
            return None
        
        print(f"[INFO] Creating standalone HTML voxel plot for {voxels.shape} grid")
        
        # Downsample if needed for better performance
        original_shape = voxels.shape
        max_resolution = 64  # Smaller for better HTML performance
        
        if any(dim > max_resolution for dim in voxels.shape):
            downsample_factor = max(voxels.shape) // max_resolution
            if downsample_factor > 1:
                downsampled_voxels = voxels[::downsample_factor, ::downsample_factor, ::downsample_factor]
                print(f"[INFO] Downsampled from {original_shape} to {downsampled_voxels.shape} for HTML")
                voxels_to_plot = downsampled_voxels
            else:
                voxels_to_plot = voxels
        else:
            voxels_to_plot = voxels
        
        # Create PyVista grid
        dims = np.array(voxels_to_plot.shape) + 1
        grid = pv.ImageData(dimensions=dims)
        grid.cell_data["voxels"] = voxels_to_plot.flatten(order="F")
        occupied = grid.threshold(0.5, scalars="voxels")
        
        if occupied.n_cells == 0:
            return None
        
        # Add height coloring
        occupied["height"] = occupied.points[:, 1]
        
        # Create plotter for HTML export
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        plotter.add_mesh(
            occupied, 
            scalars="height",
            cmap="viridis",
            show_edges=False,
            opacity=0.9,
            scalar_bar_args={"title": "Height", "color": "black"}
        )
        plotter.camera_position = 'iso'
        plotter.add_text(f"Interactive Voxels: {voxels_to_plot.sum()}", 
                        position='upper_left', font_size=12, color='black')
        
        # Generate HTML file in downloads directory
        import tempfile
        import os
        import time
        
        # Create a more accessible temporary file
        timestamp = int(time.time())
        temp_dir = tempfile.gettempdir()
        html_filename = f"voxel_interactive_{timestamp}.html"
        html_path = os.path.join(temp_dir, html_filename)
        
        # Export HTML with all dependencies embedded
        try:
            plotter.export_html(html_path)
            plotter.close()
            
            if os.path.exists(html_path) and os.path.getsize(html_path) > 0:
                # Add browser compatibility instructions to the HTML
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Add instructions at the top
                browser_instructions = """
                <!-- 
                BROWSER SETUP FOR LOCAL VIEWING:
                
                Firefox:
                1. Type 'about:config' in the URL bar
                2. Search for 'privacy.file_unique_origin'
                3. Change value from 'true' to 'false'
                4. Restart Firefox and open this file
                
                Chrome:
                1. Close all Chrome windows
                2. Start Chrome with: chrome --allow-file-access-from-files --enable-webgl
                3. Or use Chrome flags: chrome://flags/ and enable "Allow invalid certificates for resources loaded from localhost"
                
                Edge:
                1. Similar to Chrome: edge --allow-file-access-from-files --enable-webgl
                
                Alternative: Serve with simple HTTP server:
                - Python 3: python -m http.server 8000
                - Python 2: python -m SimpleHTTPServer 8000
                - Node.js: npx http-server
                -->
                """
                
                # Insert instructions after the opening HTML tag
                html_content = html_content.replace('<html>', f'<html>\n{browser_instructions}')
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                print(f"[INFO] Standalone HTML saved to: {html_path}")
                return html_path
            else:
                plotter.close()
                return None
                
        except Exception as e:
            print(f"[ERROR] HTML export failed: {e}")
            plotter.close()
            return None
            
    except Exception as e:
        print(f"[ERROR] Standalone HTML creation failed: {e}")
        return None


def create_pyvista_voxel_plot(voxels, save_as_html=False):
    """
    Create a 3D voxel plot using PyVista (screenshot only to avoid threading issues).
    
    Args:
        voxels: 3D boolean array of voxels
        save_as_html: Not used (kept for compatibility)
    
    Returns:
        numpy.ndarray: Image array or None if failed
    """
    if not PYVISTA_AVAILABLE:
        print("[WARNING] PyVista not available, falling back to matplotlib")
        return None
    
    try:
        if not voxels.any():
            print("[WARNING] No occupied voxels found")
            return None
        
        # Create PyVista structured grid from voxel data
        print(f"[INFO] Creating PyVista voxel plot for {voxels.shape} grid")
        
        # For performance, downsample if resolution is too high
        original_shape = voxels.shape
        max_resolution = 128  # Lower threshold for PyVista
        
        if any(dim > max_resolution for dim in voxels.shape):
            downsample_factor = max(voxels.shape) // max_resolution
            if downsample_factor > 1:
                downsampled_voxels = voxels[::downsample_factor, ::downsample_factor, ::downsample_factor]
                print(f"[INFO] Downsampled voxel grid from {original_shape} to {downsampled_voxels.shape} for PyVista")
                voxels_to_plot = downsampled_voxels
            else:
                voxels_to_plot = voxels
        else:
            voxels_to_plot = voxels
        
        # For PyVista ImageData, we need to create a grid with dimensions+1 for points
        # because cells are defined between points
        dims = np.array(voxels_to_plot.shape) + 1  # Add 1 to each dimension for points
        grid = pv.ImageData(dimensions=dims)
        
        # Add voxel data as cell data (cells are between points, so original voxel shape)
        grid.cell_data["voxels"] = voxels_to_plot.flatten(order="F")
        
        # Extract only the occupied voxels
        occupied = grid.threshold(0.5, scalars="voxels")
        
        if occupied.n_cells == 0:
            print("[WARNING] No occupied cells after thresholding")
            return None
        
        # Force off-screen rendering to avoid threading issues
        pv.OFF_SCREEN = True
        
        # Create plotter with minimal configuration
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        
        # Add the voxel mesh with height-based coloring
        if occupied.n_points > 0:
            # Add height-based coloring (Y-coordinate)
            occupied["height"] = occupied.points[:, 1]
            
            # Add the mesh to plotter
            plotter.add_mesh(
                occupied, 
                scalars="height",
                cmap="viridis",
                show_edges=False,  # Disable edges for better performance
                opacity=0.9,
                scalar_bar_args={
                    "title": "Height",
                    "color": "black"
                }
            )
            
            # Set up camera
            plotter.camera_position = 'iso'
            
            # Add minimal text
            total_voxels = voxels.sum()
            shown_voxels = voxels_to_plot.sum()
            plotter.add_text(f"PyVista Voxels: {shown_voxels}", position='upper_left', font_size=12, color='black')
            
            # Always use screenshot to avoid threading issues with HTML export
            try:
                print("[INFO] Taking PyVista screenshot (avoiding HTML export due to threading)")
                screenshot = plotter.screenshot()
                plotter.close()
                return screenshot
            except Exception as screenshot_error:
                print(f"[ERROR] PyVista screenshot failed: {screenshot_error}")
                plotter.close()
                return None
        
        plotter.close()
        return None
        
    except Exception as e:
        print(f"[ERROR] PyVista voxel plot failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_3d_voxel_plot(voxels, elev=20, azim=45):
    """
    Create a true 3D voxel plot showing actual voxel cubes using matplotlib.
    
    Args:
        voxels: 3D boolean array of voxels
        elev: Elevation angle for 3D view (degrees)
        azim: Azimuth angle for 3D view (degrees)
    
    Returns:
        numpy array: Image of the 3D voxel plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.colors as mcolors
        
        if not voxels.any():
            # No occupied voxels
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No occupied voxels found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Create 3D voxel plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # For performance, downsample if resolution is too high
            original_shape = voxels.shape
            max_resolution = 64  # Maximum resolution for good performance
            
            if any(dim > max_resolution for dim in voxels.shape):
                # Downsample the voxel grid
                downsample_factor = max(voxels.shape) // max_resolution
                if downsample_factor > 1:
                    # Simple downsampling by taking every nth voxel
                    downsampled_voxels = voxels[::downsample_factor, ::downsample_factor, ::downsample_factor]
                    print(f"[INFO] Downsampled voxel grid from {original_shape} to {downsampled_voxels.shape} for 3D visualization")
                    voxels_to_plot = downsampled_voxels
                else:
                    voxels_to_plot = voxels
            else:
                voxels_to_plot = voxels
            
            # Create colors for the voxels - gradient based on height (Y-axis)
            colors = np.empty(voxels_to_plot.shape + (4,), dtype=float)
            
            # Get occupied voxel coordinates for coloring
            occupied_coords = np.where(voxels_to_plot)
            if len(occupied_coords[0]) > 0:
                # Create a colormap based on height (Y coordinate)
                y_coords = occupied_coords[1]
                y_min, y_max = y_coords.min(), y_coords.max()
                
                # Normalize Y coordinates to [0, 1] for colormap
                if y_max > y_min:
                    y_normalized = (y_coords - y_min) / (y_max - y_min)
                else:
                    y_normalized = np.ones_like(y_coords) * 0.5
                
                # Use a colormap (viridis)
                cmap = plt.cm.viridis
                
                # Set colors for all voxels to transparent first
                colors[:] = [0, 0, 0, 0]  # Transparent
                
                # Set colors for occupied voxels
                for i, (x, y, z) in enumerate(zip(*occupied_coords)):
                    color = cmap(y_normalized[i])
                    colors[x, y, z] = color
            
            # Plot the voxels
            ax.voxels(voxels_to_plot, facecolors=colors, alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            total_voxels = voxels.sum()
            shown_voxels = voxels_to_plot.sum()
            if original_shape != voxels_to_plot.shape:
                ax.set_title(f'3D Voxel Visualization\n({shown_voxels} voxels shown, {total_voxels} total)\nDownsampled for performance')
            else:
                ax.set_title(f'3D Voxel Visualization\n({total_voxels} voxels)')
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Rotate view for better visualization
            ax.view_init(elev=elev, azim=azim)
            
            # Add a colorbar
            if len(occupied_coords[0]) > 0:
                # Create a dummy mappable for the colorbar
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                         norm=plt.Normalize(vmin=0, vmax=voxels_to_plot.shape[1]-1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
                cbar.set_label('Height (Y)')
        
        plt.tight_layout()
        
        # Save to numpy array
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to numpy array
        import PIL.Image
        img = PIL.Image.open(buf)
        return np.array(img)
        
    except Exception as e:
        print(f"Error creating 3D voxel plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def start_pyvista_external_viewer(voxels):
    """
    Start PyVista in an external window with full interactivity.
    This runs in a separate process to avoid threading issues.
    """
    if not PYVISTA_AVAILABLE or voxels is None:
        return
    
    try:
        print(f"[INFO] Starting external PyVista viewer for {voxels.shape} voxel grid")
        
        # Downsample if needed for performance
        original_shape = voxels.shape
        max_resolution = 128
        
        if any(dim > max_resolution for dim in voxels.shape):
            downsample_factor = max(voxels.shape) // max_resolution
            if downsample_factor > 1:
                downsampled_voxels = voxels[::downsample_factor, ::downsample_factor, ::downsample_factor]
                print(f"[INFO] Downsampled from {original_shape} to {downsampled_voxels.shape} for external viewer")
                voxels_to_plot = downsampled_voxels
            else:
                voxels_to_plot = voxels
        else:
            voxels_to_plot = voxels
        
        # Create PyVista grid
        dims = np.array(voxels_to_plot.shape) + 1
        grid = pv.ImageData(dimensions=dims)
        grid.cell_data["voxels"] = voxels_to_plot.flatten(order="F")
        occupied = grid.threshold(0.5, scalars="voxels")
        
        if occupied.n_cells == 0:
            print("[WARNING] No occupied voxels to display")
            return
        
        # Add height-based coloring
        occupied["height"] = occupied.points[:, 1]
        
        # Configure PyVista for external display
        pv.OFF_SCREEN = False  # Enable GUI for external window
        
        # Create and configure plotter
        plotter = pv.Plotter(
            window_size=[1000, 800],
            title=f"Interactive Voxel Viewer - {voxels_to_plot.sum()} voxels"
        )
        
        # Add the mesh with enhanced visualization
        plotter.add_mesh(
            occupied,
            scalars="height",
            cmap="viridis",
            show_edges=False,
            opacity=0.8,
            scalar_bar_args={
                "title": "Height (Y)",
                "color": "black",
                "label_font_size": 14,
                "title_font_size": 16,
                "width": 0.6,
                "height": 0.8
            }
        )
        
        # Set camera and add text
        plotter.camera_position = 'iso'
        plotter.add_text(
            f"Voxel Grid: {original_shape}\nShown: {voxels_to_plot.sum()} voxels\n\nControls:\n• Left Click + Drag: Rotate\n• Right Click + Drag: Pan\n• Scroll: Zoom",
            position='upper_left',
            font_size=12,
            color='black'
        )
        
        # Add axes
        plotter.show_axes()
        plotter.add_axes_at_origin(xlabel='X', ylabel='Y', zlabel='Z')
        
        # Show the plot (this will block until window is closed)
        plotter.show()
        
    except Exception as e:
        print(f"[ERROR] External PyVista viewer failed: {e}")
        import traceback
        traceback.print_exc()


def launch_external_voxel_viewer(voxels):
    """
    Launch PyVista viewer in a separate process to avoid blocking Gradio.
    """
    if voxels is None:
        return "❌ No voxel data available. Please generate voxels first."
    
    try:
        # Start PyVista in a separate process
        process = multiprocessing.Process(
            target=start_pyvista_external_viewer,
            args=(voxels,),
            daemon=True  # Dies when main process exits
        )
        process.start()
        
        voxel_count = voxels.sum()
        return f"🚀 External PyVista viewer launched with {voxel_count:,} voxels!\n\n🎮 Interactive controls:\n• Left Click + Drag: Rotate view\n• Right Click + Drag: Pan\n• Mouse Wheel: Zoom in/out\n• 'r' key: Reset view\n• 'q' key: Quit viewer"
        
    except Exception as e:
        return f"❌ Failed to launch external viewer: {str(e)}"


import time
import numpy as np
from mcpi.minecraft import Minecraft
from mcpi import block

def build_voxels_in_minecraft_fixed(voxels, scale=1, block_type='STONE', delay=0.1, 
                                   max_blocks_per_batch=100, max_total_blocks=10000):
    """
    Build voxels in Minecraft layer by layer with safety checks and error handling.
    
    Args:
        voxels: 3D boolean array of voxels
        scale: Scale factor for the build (1 = 1:1, 2 = 2x larger, etc.)
        block_type: Type of block to use ('STONE', 'WOOD', 'BRICK', etc.)
        delay: Delay between placing each layer (seconds)
        max_blocks_per_batch: Maximum blocks to place in one batch
        max_total_blocks: Maximum total blocks to prevent crashes
    
    Returns:
        str: Status message
    """
    if voxels is None:
        return "❌ No voxel data available. Please generate voxels first."
    
    try:
        # Safety check: limit total blocks to prevent crashes
        total_voxels = int(voxels.sum())
        total_blocks_scaled = total_voxels * (scale ** 3)
        
        if total_blocks_scaled > max_total_blocks:
            return f"❌ Too many blocks ({total_blocks_scaled:,}). Maximum allowed: {max_total_blocks:,}. Try reducing scale or voxel resolution."
        
        print(f"[MINECRAFT] Attempting to connect to Minecraft...")
        
        # Connect to Minecraft with error handling
        try:
            mc = Minecraft.create(address="10.0.0.175")  # Default localhost connection
            print(f"[MINECRAFT] Connected successfully")
        except Exception as conn_error:
            print(f"[MINECRAFT] Failed to connect to localhost, trying specific IP...")
            try:
                mc = Minecraft.create(address="127.0.0.1")  # Explicit localhost
                print(f"[MINECRAFT] Connected to 127.0.0.1")
            except Exception as conn_error2:
                return f"❌ Failed to connect to Minecraft: {conn_error2}. Make sure Minecraft is running with RaspberryJam mod."
        
        # Test connection with a simple command
        try:
            test_pos = mc.player.getPos()
            print(f"[MINECRAFT] Player position: {test_pos}")
        except Exception as test_error:
            return f"❌ Minecraft connection test failed: {test_error}"
        
        # Get player position and rotation with error handling
        try:
            pos = mc.player.getPos()
            rot = mc.player.getRotation()
        except Exception as pos_error:
            return f"❌ Failed to get player position: {pos_error}"
        
        # Snap to nearest cardinal direction
        rot = rot % 360
        if 45 <= rot < 135:
            direction = "west"
            fx, fz = -1, 0
        elif 135 <= rot < 225:
            direction = "north"
            fx, fz = 0, -1
        elif 225 <= rot < 315:
            direction = "east"
            fx, fz = 1, 0
        else:
            direction = "south"
            fx, fz = 0, 1
        
        try:
            mc.postToChat(f"Building {total_voxels} voxels facing: {direction}")
        except Exception as chat_error:
            print(f"[WARNING] Failed to post chat message: {chat_error}")
        
        # Origin point (player position)
        origin = (float(pos.x), float(pos.y), float(pos.z))
        forward = (float(fx), 0.0, float(fz))
        
        # Coordinate transformation function
        def local_to_world(origin, forward, lx, ly, lz):
            ox, oy, oz = origin
            fx, _, fz = forward
            wx = ox + lx * fz + ly * fx
            wy = oy + lz
            wz = oz + lx * -fx + ly * fz
            return int(round(wx)), int(round(wy)), int(round(wz))
        
        # Get block type safely
        try:
            block_id = getattr(block, block_type, block.STONE).id
        except AttributeError:
            print(f"[WARNING] Unknown block type '{block_type}', using STONE")
            block_id = block.STONE.id
        
        # Get voxel dimensions
        vx_size, vy_size, vz_size = voxels.shape
        
        # Center the build around the player
        offset_x = vx_size // 2
        offset_z = vz_size // 2
        
        blocks_placed = 0
        blocks_in_current_batch = 0
        
        try:
            mc.postToChat(f"Starting build: {total_voxels} voxels, {total_blocks_scaled} blocks total")
        except:
            pass
        
        # Build layer by layer from bottom to top (Y axis)
        for y in range(vy_size):
            layer_blocks = 0
            layer_start_time = time.time()
            
            # Collect all blocks for this layer first
            layer_block_positions = []
            
            for x in range(vx_size):
                for z in range(vz_size):
                    if voxels[x, y, z]:  # If voxel is occupied
                        # Scale the coordinates
                        for sx in range(scale):
                            for sy in range(scale):
                                for sz in range(scale):
                                    # Calculate local coordinates (centered)
                                    lx = (x - offset_x) * scale + sx
                                    ly = y * scale + sy + 1  # Start 1 block above ground
                                    lz = (z - offset_z) * scale + sz
                                    
                                    # Convert to world coordinates
                                    try:
                                        wx, wy, wz = local_to_world(origin, forward, lx, ly, lz)
                                        layer_block_positions.append((wx, wy, wz))
                                    except Exception as coord_error:
                                        print(f"[WARNING] Coordinate conversion failed: {coord_error}")
                                        continue
            
            # Place blocks in batches to avoid overwhelming Minecraft
            for i in range(0, len(layer_block_positions), max_blocks_per_batch):
                batch_positions = layer_block_positions[i:i+max_blocks_per_batch]
                
                for wx, wy, wz in batch_positions:
                    try:
                        # Safety check: don't place blocks too far from player
                        distance_from_origin = abs(wx - origin[0]) + abs(wz - origin[2])
                        if distance_from_origin > 100:  # Limit to 100 blocks from player
                            continue
                        
                        # Safety check: don't place blocks too high or too low
                        if wy < -64 or wy > 320:  # Minecraft world height limits
                            continue
                        
                        mc.setBlock(wx, wy, wz, block_id)
                        blocks_placed += 1
                        layer_blocks += 1
                        blocks_in_current_batch += 1
                        
                    except Exception as block_error:
                        print(f"[WARNING] Failed to place block at ({wx}, {wy}, {wz}): {block_error}")
                        continue
                
                # Small delay between batches to prevent overwhelming Minecraft
                if len(batch_positions) >= max_blocks_per_batch:
                    time.sleep(0.05)
            
            if layer_blocks > 0:
                layer_time = time.time() - layer_start_time
                try:
                    mc.postToChat(f"Layer {y+1}/{vy_size}: {layer_blocks} blocks ({layer_time:.1f}s)")
                except:
                    pass
                
                print(f"[MINECRAFT] Layer {y+1}/{vy_size}: {layer_blocks} blocks placed")
                
                # Add delay between layers for visual effect
                if delay > 0:
                    time.sleep(delay)
            
            # Safety check: if too many blocks placed, stop
            if blocks_placed >= max_total_blocks:
                try:
                    mc.postToChat("Build stopped: maximum block limit reached")
                except:
                    pass
                break
        
        final_message = f"✅ Build complete! Placed {blocks_placed:,} blocks from {total_voxels} voxels"
        
        try:
            mc.postToChat("Voxel build complete!")
        except:
            pass
        
        return final_message
        
    except Exception as e:
        error_msg = f"❌ Failed to build in Minecraft: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg


# Usage function that tries both methods
def build_voxels_in_minecraft_robust(voxels, scale=1, block_type='STONE', delay=0.1, base_plane='XZ'):
    """
    Robust builder that tries the full version first, then falls back to safe version.
    Uses the selected base plane orientation for building.
    """
    if voxels is None:
        return "❌ No voxel data available."
    
    print(f"[MINECRAFT] Building with base plane: {base_plane}")
    
    # Reorient voxels based on selected base plane
    oriented_voxels = reorient_voxels_for_plane(voxels, base_plane)
    
    print("[MINECRAFT] Attempting full-featured build...")
    
    # Try the full version first
    result = build_voxels_in_minecraft_fixed(
        oriented_voxels, scale, block_type, delay, 
        max_blocks_per_batch=50, 
        max_total_blocks=8000
    )
    
    # If it failed, return the error
    if result.startswith("❌"):
        print("[MINECRAFT] Full build failed")
        return result
    
    # Add plane information to the result
    plane_info = f"\n🎯 Built using {base_plane} plane as base"
    return result + plane_info

def get_plane_view(voxels, plane):
    """
    Get 2D slice view of voxels from specified plane.
    
    Args:
        voxels: 3D boolean array of voxels
        plane: Viewing plane ('XY', 'YZ', 'XZ')
    
    Returns:
        numpy array: 2D image of the plane view
    """
    if voxels is None:
        return None
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if plane == 'XY':
            # Top view (looking down Z-axis) - sum along Z
            view_data = np.sum(voxels, axis=2)  # Sum along Z
            ax.imshow(view_data.T, origin='lower', cmap='Blues', aspect='equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'XY Plane View (Top) - {np.sum(view_data)} voxels')
        elif plane == 'YZ':
            # Side view (looking down X-axis) - sum along X  
            view_data = np.sum(voxels, axis=0)  # Sum along X
            ax.imshow(view_data.T, origin='lower', cmap='Blues', aspect='equal')
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_title(f'YZ Plane View (Side) - {np.sum(view_data)} voxels')
        elif plane == 'XZ':
            # Front view (looking down Y-axis) - sum along Y
            view_data = np.sum(voxels, axis=1)  # Sum along Y
            ax.imshow(view_data.T, origin='lower', cmap='Blues', aspect='equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title(f'XZ Plane View (Front) - {np.sum(view_data)} voxels')
        else:
            # Default to XZ (front view)
            view_data = np.sum(voxels, axis=1)
            ax.imshow(view_data.T, origin='lower', cmap='Blues', aspect='equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title(f'XZ Plane View (Front) - {np.sum(view_data)} voxels')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to numpy array
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to numpy array
        import PIL.Image
        img = PIL.Image.open(buf)
        return np.array(img)
        
    except Exception as e:
        print(f"Error creating plane view: {e}")
        return None


def reorient_voxels_for_plane(voxels, base_plane):
    """
    Reorient voxel data based on the selected base plane for Minecraft building.
    
    Args:
        voxels: 3D boolean array of voxels
        base_plane: Base plane ('XY', 'YZ', 'XZ')
    
    Returns:
        numpy array: Reoriented voxels with the selected plane as the base (XY plane)
    """
    if voxels is None or base_plane == 'XY':
        return voxels  # XY is already the default orientation
    
    if base_plane == 'YZ':
        # Rotate so YZ plane becomes XY plane
        # Original: (X, Y, Z) -> New: (Y, Z, X)
        return np.transpose(voxels, (1, 2, 0))
    elif base_plane == 'XZ':
        # Rotate so XZ plane becomes XY plane  
        # Original: (X, Y, Z) -> New: (X, Z, Y)
        return np.transpose(voxels, (0, 2, 1))
    
    return voxels


def voxelize_mesh(mesh_path, resolution=64, method='solid', use_gpu=True):
    """
    Voxelize a mesh file and return visualization plots.
    
    Args:
        mesh_path: Path to the mesh file (OBJ or STL)
        resolution: Voxel grid resolution
        method: 'solid' (filled interior - best for pyramids), 'surface' (surface only), 'fast' (conservative)
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        tuple: (voxel_stats, height_slices_plot, detailed_plot, voxel_3d_plot)
    """
    if not VOXELIZATION_AVAILABLE:
        return "Voxelization not available. Please install required dependencies.", None, None
    
    try:
        # Use the fixed voxelization function
        voxels = pytorch_mesh_to_voxels_fixed(mesh_path, resolution, method, use_gpu)
        
        # Calculate statistics
        total_voxels = voxels.size
        occupied_voxels = voxels.sum()
        fill_ratio = (occupied_voxels / total_voxels) * 100
        
        stats = {
            "resolution": f"{resolution}³",
            "total_voxels": int(total_voxels),
            "occupied_voxels": int(occupied_voxels),
            "fill_ratio": f"{fill_ratio:.2f}%",
            "method": method,
            "device": "GPU" if use_gpu else "CPU"
        }
        
        # Create height slices plot
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create height slices visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Find non-empty slices
            non_empty_slices = []
            for y in range(voxels.shape[1]):
                if voxels[:, y, :].sum() > 0:
                    non_empty_slices.append(y)
            
            if non_empty_slices:
                # Plot a few representative slices
                sample_indices = np.linspace(0, len(non_empty_slices)-1, min(6, len(non_empty_slices))).astype(int)
                
                for i, idx in enumerate(sample_indices):
                    y = non_empty_slices[idx]
                    slice_data = voxels[:, y, :]
                    
                    ax_sub = plt.subplot(2, 3, i+1)
                    ax_sub.imshow(slice_data.T, origin='lower', cmap='Blues')
                    ax_sub.set_title(f'Y={y} ({slice_data.sum()} voxels)')
                    ax_sub.set_xticks([])
                    ax_sub.set_yticks([])
                
                plt.tight_layout()
                
                # Save to numpy array
                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                
                # Convert to numpy array
                import PIL.Image
                img = PIL.Image.open(buf)
                height_slices_plot = np.array(img)
            else:
                height_slices_plot = None
                
        except Exception as e:
            print(f"Error creating height slices plot: {e}")
            height_slices_plot = None
        
        # Create detailed analysis
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot voxel count vs height
            heights = []
            voxel_counts = []
            for y in range(voxels.shape[1]):
                count = voxels[:, y, :].sum()
                heights.append(y)
                voxel_counts.append(count)
            
            if any(count > 0 for count in voxel_counts):
                ax1.plot(heights, voxel_counts, 'b-o', markersize=4)
                ax1.set_xlabel('Height (Y)')
                ax1.set_ylabel('Number of Occupied Voxels')
                ax1.set_title(f'Voxel Count vs Height\nMethod: {method.title()}')
                ax1.grid(True, alpha=0.3)
                
                # Highlight pyramid shape characteristics
                max_count = max(voxel_counts)
                max_height_idx = voxel_counts.index(max_count)
                ax1.axvline(x=max_height_idx, color='red', linestyle='--', alpha=0.7, label=f'Max width at Y={max_height_idx}')
                ax1.legend()
                
                # Plot center slice
                center_y = voxels.shape[1] // 2
                center_slice = voxels[:, center_y, :]
                im = ax2.imshow(center_slice.T, origin='lower', cmap='viridis')
                ax2.set_title(f'Center Slice (Y={center_y})\nMethod: {method.title()}')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Z')
                plt.colorbar(im, ax=ax2)
                
                plt.tight_layout()
                
                # Save to numpy array
                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                
                # Convert to numpy array
                import PIL.Image
                img = PIL.Image.open(buf)
                detailed_plot = np.array(img)
            else:
                detailed_plot = None
                
        except Exception as e:
            print(f"Error creating detailed plot: {e}")
            detailed_plot = None
        
        # Create 3D voxel visualization
        voxel_3d_plot = None
        voxel_html_path = None
        
        try:
            if PYVISTA_AVAILABLE:
                # Try PyVista for high-quality static visualization
                pyvista_result = create_pyvista_voxel_plot(voxels, save_as_html=False)
                if pyvista_result is not None:
                    voxel_3d_plot = pyvista_result
                    print(f"[INFO] Created high-quality PyVista plot")
                else:
                    # Fallback to matplotlib
                    voxel_3d_plot = create_3d_voxel_plot(voxels, elev=20, azim=45)
                    print(f"[INFO] PyVista failed, using matplotlib fallback")
            else:
                # Use matplotlib if PyVista not available
                voxel_3d_plot = create_3d_voxel_plot(voxels, elev=20, azim=45)
                print(f"[INFO] Using matplotlib (PyVista not available)")
        except Exception as e:
            print(f"Error creating 3D voxel plot: {e}")
            # Final fallback to matplotlib
            try:
                voxel_3d_plot = create_3d_voxel_plot(voxels, elev=20, azim=45)
            except:
                voxel_3d_plot = None
        
        return stats, height_slices_plot, detailed_plot, voxel_3d_plot, voxels, voxel_html_path
        
    except Exception as e:
        return f"Error during voxelization: {str(e)}", None, None, None, None, None
    
def build_app():
    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if MV_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'

    title = 'Hunyuan-3D-2.1'
        
    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">

    {title}
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    """
    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1480px;
    }
    .mv-image button .wrap {
        font-size: 10px;
    }

    .mv-image .icon-wrap {
        width: 20px;
    }

    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.1', analytics_enabled=False, css=custom_css) as demo:
        gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt', visible=not MV_MODE) as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        caption = gr.State(None)
#                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
#                        caption = gr.Textbox(label='Text Prompt',
#                                             placeholder='HunyuanDiT will be used to generate image.',
#                                             info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('MultiView Prompt', visible=MV_MODE) as tab_mv:
                        # gr.Label('Please upload at least one front image.')
                        with gr.Row():
                            mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')

                with gr.Row():
                    btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                    btn_all = gr.Button(value='Gen Textured Shape',
                                        variant='primary',
                                        visible=HAS_TEXTUREGEN,
                                        min_width=100)

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                    with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(
                            label='Generation Mode',
                            choices=['Turbo', 'Fast', 'Standard'], 
                            value='Turbo')
                        gr.HTML("""
                        <div style="font-size: 11px; color: #666; margin-top: -10px; margin-bottom: 10px;">
                        <em>Recommendation: Turbo for most cases, Fast for very complex cases, Standard seldom use.</em>
                        </div>
                        """)
                        decode_mode = gr.Radio(
                            label='Decoding Mode',
                            choices=['Low', 'Standard', 'High'],
                            value='Standard')
                        gr.HTML("""
                        <div style="font-size: 11px; color: #666; margin-top: -10px; margin-bottom: 10px;">
                        <em>The resolution for exporting mesh from generated vectset</em>
                        </div>
                        """)
                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(
                                value=True, 
                                label='Remove Background', 
                                min_width=100)
                            randomize_seed = gr.Checkbox(
                                label="Randomize seed", 
                                value=True, 
                                min_width=100)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                        )
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps')
                            octree_resolution = gr.Slider(maximum=512, 
                                                          minimum=16, 
                                                          value=256, 
                                                          label='Octree Resolution')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100)
                    with gr.Tab("Export", id='tab_export'):
                        with gr.Row():
                            file_type = gr.Dropdown(label='File Type', 
                                                    choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100)
                            reduce_face = gr.Checkbox(label='Simplify Mesh', 
                                                      value=False, min_width=100)
                            export_texture = gr.Checkbox(label='Include Texture', value=False,
                                                         visible=False, min_width=100)
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000,
                                                    label='Target Face Number')
                        with gr.Row():
                            confirm_export = gr.Button(value="Transform", min_width=100)
                            file_export = gr.DownloadButton(label="Download", variant='primary',
                                                            interactive=False, min_width=100)

            with gr.Column(scale=6):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')
                    with gr.Tab('Voxelization', id='voxelization_panel', visible=VOXELIZATION_AVAILABLE):
                        pyvista_status = "✅ Available (External Viewer + High-Quality 3D)" if PYVISTA_AVAILABLE else "❌ Not Available (Basic 3D only)"
                        gr.HTML(f"""
                        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <h4>🎯 Voxelization Tool</h4>
                        <p>Convert 3D meshes into voxel grid representations. You can either:</p>
                        <ul>
                            <li><strong>📁 Upload your own mesh file</strong> (STL/OBJ) - Use any existing mesh</li>
                            <li><strong>🔗 Use exported mesh</strong> - From the "Export" tab after generating a mesh</li>
                        </ul>
                        <p>This tool creates:</p>
                        <ul>
                            <li><strong>Height Slices:</strong> 2D cross-sections at different heights</li>
                            <li><strong>Detailed Analysis:</strong> Voxel count distribution and center slice visualization</li>
                            <li><strong>High-Quality 3D Plot:</strong> PyVista-powered visualization with rotation controls</li>
                            <li><strong>🚀 External Viewer:</strong> Launch PyVista in separate window with full mouse interactivity</li>
                            <li><strong>🎮 Minecraft Building:</strong> Build your voxels in Minecraft layer by layer!</li>
                            <li><strong>Statistics:</strong> Resolution, occupied voxels, and fill ratio</li>
                        </ul>
                        <p><em>PyVista Status: {pyvista_status}</em></p>
                        <p><em>💡 Tip: Use the External Viewer for the best interactive experience!</em></p>
                        <p><em>Note: Requires PyTorch and GPU for optimal performance.</em></p>
                        </div>
                        """)
                        
                        # Status indicator
                        status_color = "#4CAF50" if VOXELIZATION_AVAILABLE else "#f44336"
                        status_text = "✅ Available" if VOXELIZATION_AVAILABLE else "❌ Not Available"
                        gr.HTML(f"""
                        <div style="background-color: {status_color}; color: white; padding: 5px 10px; border-radius: 3px; margin-bottom: 10px; text-align: center;">
                        <strong>Voxelization Status: {status_text}</strong>
                        </div>
                        """)
                        
                        # Minecraft building status indicator
                        minecraft_status_color = "#4CAF50" if MINECRAFT_AVAILABLE else "#f44336"
                        minecraft_status_text = "✅ Available" if MINECRAFT_AVAILABLE else "❌ Not Available"
                        gr.HTML(f"""
                        <div style="background-color: {minecraft_status_color}; color: white; padding: 5px 10px; border-radius: 3px; margin-bottom: 10px; text-align: center;">
                        <strong>Minecraft Building Status: {minecraft_status_text}</strong>
                        </div>
                        """)
                        with gr.Row():
                            with gr.Column(scale=1):
                                # Optional mesh file upload
                                gr.HTML("""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                                <h4>📁 Mesh Input (Optional)</h4>
                                <p><em>Upload your own mesh file, or leave empty to use the exported mesh from previous steps</em></p>
                                </div>
                                """)
                                
                                mesh_file_upload = gr.File(
                                    label="Upload Mesh File (STL/OBJ)",
                                    file_types=['.stl', '.obj'],
                                    file_count='single'
                                )
                                gr.HTML("""
                                <div style="font-size: 11px; color: #666; margin-top: -10px; margin-bottom: 15px;">
                                <em>Supported formats: STL, OBJ. If no file uploaded, uses exported mesh.</em>
                                </div>
                                """)
                                
                                voxel_resolution = gr.Slider(
                                    minimum=16, maximum=256, value=64, step=16,
                                    label='Voxel Resolution'
                                )
                                gr.HTML("""
                                <div style="font-size: 11px; color: #666; margin-top: -10px; margin-bottom: 10px;">
                                <em>Grid resolution for voxelization</em>
                                </div>
                                """)
                                # Replace the voxel method radio button section in your UI with this:
                                voxel_method = gr.Radio(
                                    choices=['solid', 'surface', 'fast'], 
                                    value='solid',
                                    label='Voxelization Method'
                                )
                                gr.HTML("""
                                <div style="font-size: 11px; color: #666; margin-top: -10px; margin-bottom: 10px;">
                                <em>Solid: filled interior (best for pyramids/closed shapes), Surface: surface only, Fast: conservative approximation</em>
                                </div>
                                """)
                                use_gpu_voxel = gr.Checkbox(
                                    value=True, label='Use GPU'
                                )
                                gr.HTML("""
                                <div style="font-size: 11px; color: #666; margin-top: -10px; margin-bottom: 10px;">
                                <em>Use GPU acceleration if available</em>
                                </div>
                                """)
                                voxelize_btn = gr.Button('Generate Voxels', variant='primary')
                                
                                # Minecraft building controls
                                gr.HTML("""
                                <div style="background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin-top: 15px;">
                                <h4>🎮 Minecraft Building</h4>
                                <p><em>Build generated voxels in Minecraft layer by layer</em></p>
                                </div>
                                """)
                                
                                minecraft_scale = gr.Slider(
                                    minimum=1, maximum=5, value=1, step=1,
                                    label='Build Scale'
                                )
                                gr.HTML("""
                                <div style="font-size: 11px; color: #666; margin-top: -10px; margin-bottom: 10px;">
                                <em>1 = 1:1 scale, 2 = 2x larger, etc.</em>
                                </div>
                                """)
                                
                                minecraft_block_type = gr.Dropdown(
                                    choices=['STONE', 'WOOD', 'BRICK', 'COBBLESTONE', 'GOLD_BLOCK', 'IRON_BLOCK', 'DIAMOND_BLOCK', 'WOOL'],
                                    value='STONE',
                                    label='Block Type'
                                )
                                
                                minecraft_delay = gr.Slider(
                                    minimum=0.0, maximum=2.0, value=0.1, step=0.1,
                                    label='Layer Delay (seconds)'
                                )
                                gr.HTML("""
                                <div style="font-size: 11px; color: #666; margin-top: -10px; margin-bottom: 10px;">
                                <em>Delay between building each layer</em>
                                </div>
                                """)
                                
                                minecraft_build_btn = gr.Button('🏗️ Build in Minecraft', variant='primary', visible=MINECRAFT_AVAILABLE)
                            
                            with gr.Column(scale=2):
                                voxel_stats = gr.Json({}, label='Voxel Statistics')
                                minecraft_build_status = gr.Textbox(
                                    label="Minecraft Build Status",
                                    visible=False,
                                    interactive=False
                                )
                        
                        with gr.Row():
                            height_slices_plot = gr.Image(label='Height Slices Visualization', visible=False)
                            detailed_plot = gr.Image(label='Detailed Analysis', visible=False)
                        
                        with gr.Row():
                            with gr.Column():
                                # Interactive PyVista plot (when available)
                                voxel_interactive_plot = gr.HTML(label='Interactive 3D Voxel Plot', visible=False)
                                # Fallback matplotlib plot  
                                voxel_3d_plot = gr.Image(label='3D Voxel Visualization (Static)', visible=False)
                                # Download interactive HTML file
                                download_html_btn = gr.DownloadButton(
                                    label="📁 Download Interactive HTML", 
                                    variant="secondary", 
                                    visible=False,
                                    size="sm"
                                )
                                # External PyVista viewer button
                                external_viewer_btn = gr.Button(
                                    "🚀 Launch External Viewer",
                                    variant="primary",
                                    visible=False,
                                    size="sm"
                                )
                                # Status display for external viewer
                                external_viewer_status = gr.Textbox(
                                    label="External Viewer Status",
                                    visible=False,
                                    interactive=False
                                )
                        
                        # Plane view controls for voxel visualization
                        with gr.Row(visible=False) as voxel_plane_controls:
                            gr.HTML("""
                            <div style="text-align: center; margin: 10px 0;">
                            <strong>📐 Plane Views</strong><br>
                            <em style="font-size: 12px; color: #666;">Select base plane for visualization and Minecraft building</em>
                            </div>
                            """)
                        
                        with gr.Row(visible=False) as voxel_plane_buttons:
                            plane_xy_btn = gr.Button("📋 XY Plane (Top)", size="sm", variant="secondary")
                            plane_yz_btn = gr.Button("📋 YZ Plane (Side)", size="sm", variant="secondary") 
                            plane_xz_btn = gr.Button("📋 XZ Plane (Front)", size="sm", variant="primary")
                        
                        # Hidden states to store current voxel data and selected plane
                        current_voxels = gr.State(None)
                        selected_plane = gr.State('XZ')  # Default to XZ (front view)

            with gr.Column(scale=3 if MV_MODE else 2):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Image to 3D Gallery', 
                                id='tab_img_gallery', 
                                visible=not MV_MODE) as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=18)

        tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
        #if HAS_T2I:
        #    tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value == 'Fast':
                return gr.update(value=10)
            else:
                return gr.update(value=30)

        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value == 'Low':
                return gr.update(value=196)
            elif value == 'Standard':
                return gr.update(value=256)
            else:
                return gr.update(value=384)

        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], 
                           outputs=[octree_resolution])

        def on_export_click(file_out, file_out2, file_type, 
                            reduce_face, export_texture, target_face_num):
            if file_out is None:
                raise gr.Error('Please generate a mesh first.')

            print(f'exporting {file_out}')
            print(f'reduce face to {target_face_num}')
            if export_texture:
                mesh = trimesh.load(file_out2)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=True, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=True)
                model_viewer_html = build_model_viewer_html(save_folder, 
                                                            height=HTML_HEIGHT, 
                                                            width=HTML_WIDTH,
                                                            textured=True)
            else:
                mesh = trimesh.load(file_out)
                mesh = floater_remove_worker(mesh)
                mesh = degenerate_face_remove_worker(mesh)
                if reduce_face:
                    mesh = face_reduce_worker(mesh, target_face_num)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=False, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=False)
                model_viewer_html = build_model_viewer_html(save_folder, 
                                                            height=HTML_HEIGHT, 
                                                            width=HTML_WIDTH,
                                                            textured=False)
            print(f'export to {path}')
            return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected='export_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )

        # Voxelization event handler
        def on_voxelize_click(mesh_file_upload, file_export, resolution, method, use_gpu):
            # Use uploaded mesh file if provided, otherwise use exported mesh
            if mesh_file_upload is not None:
                mesh_path = mesh_file_upload.name
                print(f'Voxelizing uploaded mesh: {mesh_path}')
            elif file_export is not None:
                mesh_path = file_export
                print(f'Voxelizing exported mesh: {mesh_path}')
            else:
                raise gr.Error('Please either upload a mesh file or export a mesh first using the "Export" tab.')
            
            stats, height_plot, detailed_plot, voxel_3d_plot, voxels, voxel_html_path = voxelize_mesh(mesh_path, resolution, method, use_gpu)
            
            # Update visibility of plots
            height_visible = height_plot is not None
            detailed_visible = detailed_plot is not None
            voxel_3d_visible = voxel_3d_plot is not None
            
            # No interactive HTML plot - just static visualization with rotation controls
            interactive_plot_html = ""
            interactive_visible = False
            
            # Show plane controls for static plots (both PyVista and matplotlib)
            plane_controls_visible = voxels is not None and voxel_3d_plot is not None
            
            # Show download button if HTML file is available
            download_visible = voxel_html_path is not None
            
            # Show external viewer button if PyVista is available and we have voxels
            external_viewer_visible = PYVISTA_AVAILABLE and voxels is not None
            
            return (
                stats,
                gr.update(value=height_plot, visible=height_visible),
                gr.update(value=detailed_plot, visible=detailed_visible),
                gr.update(value=interactive_plot_html, visible=interactive_visible),  # Interactive plot
                gr.update(value=voxel_3d_plot, visible=voxel_3d_visible and not interactive_visible),  # Static plot
                gr.update(value=voxel_html_path, visible=download_visible),  # Download button
                gr.update(visible=external_viewer_visible),  # External viewer button
                gr.update(visible=False, value=""),  # Reset external viewer status
                voxels,  # Store voxel data in state
                gr.update(visible=plane_controls_visible),  # Show plane controls 
                gr.update(visible=plane_controls_visible)   # Show plane buttons
            )

        voxelize_btn.click(
            lambda: gr.update(selected='voxelization_panel'),
            outputs=[tabs_output],
        ).then(
            on_voxelize_click,
            inputs=[mesh_file_upload, file_export, voxel_resolution, voxel_method, use_gpu_voxel],
            outputs=[voxel_stats, height_slices_plot, detailed_plot, voxel_interactive_plot, voxel_3d_plot, download_html_btn, external_viewer_btn, external_viewer_status, current_voxels, voxel_plane_controls, voxel_plane_buttons]
        )
        
        # Plane view button event handlers
        def on_plane_click(voxels, plane):
            if voxels is None:
                return gr.update(), gr.update()
            plane_view = get_plane_view(voxels, plane) 
            # Update button variants to show which is selected
            xy_variant = "primary" if plane == "XY" else "secondary"
            yz_variant = "primary" if plane == "YZ" else "secondary"
            xz_variant = "primary" if plane == "XZ" else "secondary"
            return (
                gr.update(value=plane_view) if plane_view is not None else gr.update(),
                plane  # Update the selected plane state
            )
        
        # Connect plane view buttons
        plane_xy_btn.click(
            lambda voxels: on_plane_click(voxels, 'XY'),
            inputs=[current_voxels],
            outputs=[voxel_3d_plot, selected_plane]
        ).then(
            lambda: (gr.update(variant="primary"), gr.update(variant="secondary"), gr.update(variant="secondary")),
            outputs=[plane_xy_btn, plane_yz_btn, plane_xz_btn]
        )
        
        plane_yz_btn.click(
            lambda voxels: on_plane_click(voxels, 'YZ'),
            inputs=[current_voxels],
            outputs=[voxel_3d_plot, selected_plane]
        ).then(
            lambda: (gr.update(variant="secondary"), gr.update(variant="primary"), gr.update(variant="secondary")),
            outputs=[plane_xy_btn, plane_yz_btn, plane_xz_btn]
        )
        
        plane_xz_btn.click(
            lambda voxels: on_plane_click(voxels, 'XZ'),
            inputs=[current_voxels],
            outputs=[voxel_3d_plot, selected_plane]
        ).then(
            lambda: (gr.update(variant="secondary"), gr.update(variant="secondary"), gr.update(variant="primary")),
            outputs=[plane_xy_btn, plane_yz_btn, plane_xz_btn]
        )
        
        # External viewer button event handler
        def on_external_viewer_click(voxels):
            status = launch_external_voxel_viewer(voxels)
            return gr.update(value=status, visible=True)
        
        external_viewer_btn.click(
            on_external_viewer_click,
            inputs=[current_voxels],
            outputs=[external_viewer_status]
        )
        
        # Minecraft build button event handler
        # Replace your minecraft_build_btn.click event handler with this:

        # Minecraft build button event handler
        def on_minecraft_build_click(voxels, scale, block_type, delay, base_plane):
            if not MINECRAFT_AVAILABLE:
                return gr.update(value="❌ Minecraft Pi API not available. Install with: pip install mcpi", visible=True)
            
            if voxels is None:
                return gr.update(value="❌ No voxel data available. Please generate voxels first.", visible=True)
            
            # Safety checks
            total_voxels = int(voxels.sum())
            if total_voxels == 0:
                return gr.update(value="❌ No occupied voxels found in the data.", visible=True)
            
            total_blocks = total_voxels * (scale ** 3)
            if total_blocks > 10000:
                return gr.update(
                    value=f"❌ Too many blocks ({total_blocks:,}). Maximum recommended: 10,000.\n"
                        f"Try reducing scale (current: {scale}) or voxel resolution.\n"
                        f"Current voxels: {total_voxels:,}", 
                    visible=True
                )
            
            # Use the robust builder with selected plane
            status = build_voxels_in_minecraft_robust(voxels, scale, block_type, delay, base_plane)
            return gr.update(value=status, visible=True)

        minecraft_build_btn.click(
            on_minecraft_build_click,
            inputs=[current_voxels, minecraft_scale, minecraft_block_type, minecraft_delay, selected_plane],
            outputs=[minecraft_build_status]
        )

    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='./save_dir')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    args = parser.parse_args()
    args.enable_flashvdm = False

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            # Apply torchvision fix before importing basicsr/RealESRGAN
            print("Applying torchvision compatibility fix for texture generation...")
            try:
                from torchvision_fix import apply_fix
                fix_result = apply_fix()
                if not fix_result:
                    print("Warning: Torchvision fix may not have been applied successfully")
            except Exception as fix_error:
                print(f"Warning: Failed to apply torchvision fix: {fix_error}")
            
            # from hy3dgen.texgen import Hunyuan3DPaintPipeline
            # texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            # if args.low_vram_mode:
            #     texgen_worker.enable_model_cpu_offload()

            from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
            conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            tex_pipeline = Hunyuan3DPaintPipeline(conf)
        
            # Not help much, ignore for now.
            # if args.compile:
            #     texgen_worker.models['delight_model'].pipeline.unet.compile()
            #     texgen_worker.models['delight_model'].pipeline.vae.compile()
            #     texgen_worker.models['multiview_model'].pipeline.unet.compile()
            #     texgen_worker.models['multiview_model'].pipeline.vae.compile()
            
            HAS_TEXTUREGEN = True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading texture generator: {e}")
            print("Failed to load texture generator.")
            print('Please try to install requirements by following README.md')
            HAS_TEXTUREGEN = False

    HAS_T2I = True
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline

        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True

    from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier, \
        Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=False,
        device=args.device,
    )
    if args.enable_flashvdm:
        mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
        i23d_worker.enable_flashvdm(mc_algo=mc_algo)
    if args.compile:
        i23d_worker.compile()

    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()

    # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
    # create a FastAPI app
    app = FastAPI()
    
    # create a static directory to store the static files
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)

    if args.low_vram_mode:
        torch.cuda.empty_cache()
    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host=args.host, port=args.port)
