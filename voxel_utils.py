"""
Voxel Utilities Module for Hunyuan3D-2.1

This module provides comprehensive voxelization functionality including:
- PyTorch GPU-accelerated voxelization
- 3D visualization with PyVista and matplotlib
- Minecraft integration for building voxels
- Height slice analysis and plotting
- Multiple voxelization methods (solid, surface, fast)

Author: Hunyuan3D Team
License: TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
"""

import os
import time
import random
import multiprocessing
import numpy as np
import trimesh
import tempfile
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import pickle
import io
import PIL.Image
import torch
import torch.nn.functional as F
TORCH_VOXEL_AVAILABLE = True
voxel_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] PyTorch available for voxelization - Using device: {voxel_device}")


# PyVista for interactive 3D visualization
import pyvista as pv
# Set PyVista to work in headless mode for server environments
pv.set_plot_theme("document")
pv.OFF_SCREEN = True  # Enable off-screen rendering for server
PYVISTA_AVAILABLE = True
print("[INFO] PyVista available for interactive 3D visualization")

# Minecraft Pi for voxel building
try:
    from mcpi.minecraft import Minecraft
    from mcpi import block
    MINECRAFT_AVAILABLE = True
    print("[INFO] Minecraft Pi API available for voxel building")
except ImportError:
    print("[WARNING] Minecraft Pi API not available. Install with: pip install mcpi")
    MINECRAFT_AVAILABLE = False


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


def pytorch_voxelize_mesh(vertices, faces, resolution, device='cuda'):
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
    voxel_batch_size = 10000000
    
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


def pytorch_mesh_to_voxels(mesh_path: str, resolution: int = 32, 
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
    
    device = voxel_device
    print(f"[INFO] Using {device} for voxelization")
    
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
            voxel_grid = pytorch_voxelize_mesh(mesh.vertices, mesh.faces, resolution, device)
        else:  # fast method - use conservative approach
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


def save_voxel_matrix_to_txt(voxels, mesh_path, resolution, method):
    """
    Save voxel matrix to a text file and pickle file for debugging purposes.
    
    Args:
        voxels: 3D boolean array of voxels
        mesh_path: Original mesh file path
        resolution: Voxel resolution used
        method: Voxelization method used
    """
    try:
        # Create filename based on mesh path and parameters
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        timestamp = int(time.time())
        txt_filename = f"voxel_matrix_{mesh_name}_{resolution}_{method}_{timestamp}.txt"
        pkl_filename = f"voxel_matrix_{mesh_name}_{resolution}_{method}_{timestamp}.pkl"
        
        # Save to the same directory as the mesh file
        save_dir = os.path.dirname(mesh_path)
        if not save_dir:
            save_dir = "."
        
        txt_filepath = os.path.join(save_dir, txt_filename)
        pkl_filepath = os.path.join(save_dir, pkl_filename)
        
        # Save as simple text format - one line per voxel with coordinates
        with open(txt_filepath, 'w') as f:
            # Write header with metadata
            f.write(f"# Voxel Matrix Export\n")
            f.write(f"# Original mesh: {mesh_path}\n")
            f.write(f"# Resolution: {resolution}³\n")
            f.write(f"# Method: {method}\n")
            f.write(f"# Shape: {voxels.shape}\n")
            f.write(f"# Total voxels: {voxels.size}\n")
            f.write(f"# Occupied voxels: {voxels.sum()}\n")
            f.write(f"# Export time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Format: x y z occupied\n")
            f.write(f"#\n")
            
            # Write voxel data
            for x in range(voxels.shape[0]):
                for y in range(voxels.shape[1]):
                    for z in range(voxels.shape[2]):
                        occupied = 1 if voxels[x, y, z] else 0
                        f.write(f"{x} {y} {z} {occupied}\n")
        
        metadata = {
            'original_mesh': mesh_path,
            'resolution': resolution,
            'method': method,
            'shape': voxels.shape,
            'total_voxels': voxels.size,
            'occupied_voxels': int(voxels.sum()),
            'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'voxel_matrix': voxels
        }
        
        with open(pkl_filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"[DEBUG] Voxel matrix saved to: {txt_filepath}")
        print(f"[DEBUG] Voxel matrix pickle saved to: {pkl_filepath}")
        return txt_filepath, pkl_filepath
        
    except Exception as e:
        print(f"[ERROR] Failed to save voxel matrix: {e}")
        return None, None


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
    try:
        voxels = pytorch_mesh_to_voxels(mesh_path, resolution, method, True)
        
        # Save voxel matrix to text file for debugging
        save_voxel_matrix_to_txt(voxels, mesh_path, resolution, method)
        
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
            "device": "GPU"
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
                
                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close()
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
                
                import io
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close()
                
                img = PIL.Image.open(buf)
                detailed_plot = np.array(img)
            else:
                detailed_plot = None
                
        except Exception as e:
            print(f"Error creating detailed plot: {e}")
            detailed_plot = None
        
        # Create 3D voxel visualization using PyVista
        voxel_3d_plot = None
        voxel_html_path = None
        
        try:
            # Use PyVista for high-quality visualization
            voxel_3d_plot = create_pyvista_voxel_plot(voxels, save_as_html=False)
            if voxel_3d_plot is not None:
                print(f"[INFO] Created PyVista plot")
        except Exception as e:
            print(f"Error creating 3D voxel plot: {e}")
            voxel_3d_plot = None
        
        return stats, height_slices_plot, detailed_plot, voxel_3d_plot, voxels, voxel_html_path
        
    except Exception as e:
        error_stats = {"error": f"Error during voxelization: {str(e)}"}
        return error_stats, None, None, None, None, None


def create_standalone_html_voxel_plot(voxels):
    """
    Create a standalone HTML file for local viewing without web server.
    
    Args:
        voxels: 3D boolean array of voxels
    
    Returns:
        str: Path to standalone HTML file or None if failed
    """
    
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





def get_plane_view(voxels, plane):
    """
    Get 2D slice view of voxels from specified plane.
    
    Args:
        voxels: 3D boolean array of voxels
        plane: Viewing plane ('XY', 'YZ', 'XZ')
    
    Returns:
        PIL.Image: 2D image of the plane view
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
        
        # Convert to PIL Image (which Gradio expects)
        img = PIL.Image.open(buf)
        return img
        
    except Exception as e:
        print(f"Error creating plane view: {e}")
        return None


def get_height_slices_for_plane(voxels, plane):
    """
    Generate height slices visualization for the specified plane orientation.
    
    Args:
        voxels: 3D boolean array of voxels
        plane: Viewing plane ('XY', 'YZ', 'XZ') - determines slicing direction
    
    Returns:
        PIL.Image: Height slices visualization oriented for the selected plane
    """
    if voxels is None:
        return None
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Reorient voxels based on the selected plane for consistent height slicing
        if plane == 'XY':
            # XY plane: slice along Z-axis (original orientation)
            oriented_voxels = voxels
            slice_axis = 2  # Z
            axis_labels = ('X', 'Y')
            slice_name = 'Z'
        elif plane == 'YZ':
            # YZ plane: slice along X-axis, so we need to permute axes
            # Original (X,Y,Z) -> New (Y,Z,X) so X becomes the slicing dimension
            oriented_voxels = np.transpose(voxels, (1, 2, 0))
            slice_axis = 2  # X (now at position 2)
            axis_labels = ('Y', 'Z')
            slice_name = 'X'
        elif plane == 'XZ':
            # XZ plane: slice along Y-axis, so we need to permute axes
            # Original (X,Y,Z) -> New (X,Z,Y) so Y becomes the slicing dimension
            oriented_voxels = np.transpose(voxels, (0, 2, 1))
            slice_axis = 2  # Y (now at position 2)
            axis_labels = ('X', 'Z')
            slice_name = 'Y'
        else:
            # Default to XZ
            oriented_voxels = np.transpose(voxels, (0, 2, 1))
            slice_axis = 2
            axis_labels = ('X', 'Z')
            slice_name = 'Y'
        
        # Find non-empty slices
        slice_dimension = oriented_voxels.shape[slice_axis]
        non_empty_slices = []
        
        for i in range(slice_dimension):
            if slice_axis == 0:
                slice_data = oriented_voxels[i, :, :]
            elif slice_axis == 1:
                slice_data = oriented_voxels[:, i, :]
            else:  # slice_axis == 2
                slice_data = oriented_voxels[:, :, i]
            
            if slice_data.sum() > 0:
                non_empty_slices.append(i)
        
        if not non_empty_slices:
            print(f"[WARNING] No occupied voxels found in any {slice_name} slice")
            return None
        
        print(f"[INFO] Creating height slices for {plane} plane: {len(non_empty_slices)} non-empty {slice_name} slices")
        
        # Calculate subplot grid
        n_slices = len(non_empty_slices)
        max_cols = 6
        n_cols = min(max_cols, n_slices)
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        # Create figure
        fig_width = min(15, n_cols * 2.5)
        fig_height = min(12, n_rows * 2.5)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        fig.suptitle(f'Height Slices - {plane} Plane View ({slice_name}-axis slicing)', fontsize=14)
        
        # Handle single subplot case
        if n_slices == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        # Plot slices
        for idx, slice_idx in enumerate(non_empty_slices[:len(axes)]):
            ax = axes[idx]
            
            if slice_axis == 0:
                slice_data = oriented_voxels[slice_idx, :, :]
            elif slice_axis == 1:
                slice_data = oriented_voxels[:, slice_idx, :]
            else:  # slice_axis == 2
                slice_data = oriented_voxels[:, :, slice_idx]
            
            ax.imshow(slice_data.T, origin='lower', cmap='Blues', 
                     interpolation='nearest', aspect='equal')
            ax.set_title(f'{slice_name}={slice_idx} ({slice_data.sum()} voxels)')
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.grid(True, alpha=0.3)
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(len(non_empty_slices), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Convert to PIL Image
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        img = PIL.Image.open(buf)
        return img
        
    except Exception as e:
        print(f"Error creating height slices for {plane} plane: {e}")
        return None


def reorient_voxels_for_plane(voxels, base_plane):
    """
    Reorient voxel data based on the selected base plane for visualization purposes.
    
    Args:
        voxels: 3D boolean array of voxels
        base_plane: Base plane ('XY', 'YZ', 'XZ')
    
    Returns:
        numpy array: Reoriented voxels for better visualization of the selected plane
    """
    if voxels is None:
        return voxels
    
    print(f"[REORIENT] Input: base_plane='{base_plane}', voxel_shape={voxels.shape}")
    
    if base_plane == 'XY':
        print(f"[REORIENT] XY plane: no reorientation needed (default)")
        return voxels  # XY is already the default orientation
    elif base_plane == 'YZ':
        # Rotate for YZ plane visualization
        reoriented = np.transpose(voxels, (1, 2, 0))
        print(f"[REORIENT] YZ plane: (X,Y,Z) -> (Y,Z,X), new_shape={reoriented.shape}")
        return reoriented
    elif base_plane == 'XZ':
        # Rotate for XZ plane visualization
        reoriented = np.transpose(voxels, (0, 2, 1))
        print(f"[REORIENT] XZ plane: (X,Y,Z) -> (X,Z,Y), new_shape={reoriented.shape}")
        return reoriented
    
    print(f"[REORIENT] Unknown plane '{base_plane}', no reorientation")
    return voxels 


def start_pyvista_external_viewer(voxels):
    """
    Start PyVista in an external window with full interactivity.
    This runs in a separate process to avoid threading issues.
    """
    if voxels is None:
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


def build_voxels_in_minecraft_robust(voxels, scale=1, block_type='STONE', delay=0.1, base_plane='XZ'):
    """
    Simple builder that reorients voxel matrix based on selected plane, then builds normally.
    """
    if voxels is None:
        return "❌ No voxel data available."
    
    print(f"[MINECRAFT] Building with base plane: {base_plane}")
    print(f"[MINECRAFT] Original voxel shape: {voxels.shape}, occupied: {voxels.sum()}")
    
    # Simply reorient the voxel matrix based on plane selection
    if base_plane == 'XY':
        # XY plane: Build Z layers (bottom to top) - DEFAULT
        working_voxels = voxels  # No change needed
        build_description = "bottom to top (Z layers)"
    elif base_plane == 'YZ':
        # YZ plane: Build X layers (left to right)
        # Reorient: (X,Y,Z) -> (Y,Z,X) so X becomes the layering axis
        working_voxels = np.transpose(voxels, (1, 2, 0))
        build_description = "left to right (X layers)"
    elif base_plane == 'XZ':
        # XZ plane: Build Y layers (front to back)  
        # Reorient: (X,Y,Z) -> (X,Z,Y) so Y becomes the layering axis
        working_voxels = np.transpose(voxels, (0, 2, 1))
        build_description = "front to back (Y layers)"
    else:
        # Default to XZ
        working_voxels = np.transpose(voxels, (0, 2, 1))
        build_description = "front to back (Y layers)"
    
    print(f"[MINECRAFT] Reoriented voxel shape: {working_voxels.shape}, occupied: {working_voxels.sum()}")
    print(f"[MINECRAFT] Building {build_description}")
    
    # Use simple vertical building on the reoriented matrix
    result = build_voxels_in_minecraft_simple(
        working_voxels, scale, block_type, delay, 
        max_blocks_per_batch=50, 
        max_total_blocks=900000
    )
    
    # Add plane information to the result
    plane_info = f"\n🎯 Built using {base_plane} plane - {build_description}"
    
    return result + plane_info


def build_voxels_in_minecraft_simple(voxels, scale=1, block_type='STONE', delay=0.1, 
                                     max_blocks_per_batch=50, max_total_blocks=900000):
    """
    Simple builder that builds layer by layer from bottom to top (Z direction).
    Works on any reoriented voxel matrix.
    """
    if voxels is None:
        return "❌ No voxel data available."
    
    try:
        # Safety check
        total_voxels = int(voxels.sum())
        total_blocks_scaled = total_voxels * (scale ** 3)
        
        if total_blocks_scaled > max_total_blocks:
            return f"❌ Too many blocks ({total_blocks_scaled:,}). Maximum allowed: {max_total_blocks:,}."
        
        print(f"[MINECRAFT] Attempting to connect to Minecraft...")
        
        # Connect to Minecraft
        try:
            mc = Minecraft.create(address="10.0.0.175")
            print(f"[MINECRAFT] Connected successfully")
        except Exception as conn_error:
            try:
                mc = Minecraft.create(address="127.0.0.1")
                print(f"[MINECRAFT] Connected to 127.0.0.1")
            except Exception as conn_error2:
                return f"❌ Failed to connect to Minecraft: {conn_error2}"
        
        # Get player position and setup
        pos = mc.player.getPos()
        rot = mc.player.getRotation()
        
        # Snap to cardinal direction
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
        
        # Find ground level
        ground_y = float(pos.y)
        for check_y in range(int(pos.y), int(pos.y) - 10, -1):
            try:
                block_below = mc.getBlock(int(pos.x), check_y - 1, int(pos.z))
                if block_below != 0:
                    ground_y = float(check_y)
                    break
            except:
                continue
        
        # Build origin (3 blocks in front of player)
        build_distance = 3
        origin_x = float(pos.x) + fx * build_distance
        origin_z = float(pos.z) + fz * build_distance
        
        # Find ground at build location
        build_ground_y = ground_y
        for check_y in range(int(ground_y + 5), int(ground_y - 10), -1):
            try:
                block_at_build = mc.getBlock(int(origin_x), check_y - 1, int(origin_z))
                if block_at_build != 0:
                    build_ground_y = float(check_y)
                    break
            except:
                continue
        
        origin = (origin_x, build_ground_y, origin_z)
        
        # Get block type
        try:
            block_id = getattr(block, block_type, block.STONE).id
        except AttributeError:
            block_id = block.STONE.id
        
        # Get voxel dimensions
        vx_size, vy_size, vz_size = voxels.shape
        
        # Center offsets
        offset_x = vx_size // 2
        offset_y = vy_size // 2
        offset_z = vz_size // 2
        
        # Chat messages
        try:
            mc.postToChat(f"Building {total_voxels} voxels")
            mc.postToChat(f"Voxel grid: {vx_size}x{vy_size}x{vz_size}")
            mc.postToChat(f"Building {vz_size} layers from bottom to top")
        except:
            pass
        
        # Build layer by layer from bottom to top (Z axis) - SIMPLE APPROACH
        for z in range(vz_size):
            layer_blocks = 0
            layer_start_time = time.time()
            layer_block_positions = []
            
            # Standard layer building (X,Y plane at height Z)
            for x in range(vx_size):
                for y in range(vy_size):
                    if voxels[x, y, z]:  # If voxel is occupied
                        for sx in range(scale):
                            for sy in range(scale):
                                for sz in range(scale):
                                    # Simple coordinate mapping: voxel coordinates → world coordinates
                                    wx = int(origin[0] + (x - offset_x) * scale + sx)
                                    wy = int(origin[1] + (z - offset_z) * scale + sz)  # Z becomes height
                                    wz = int(origin[2] + (y - offset_y) * scale + sy)  # Y becomes depth
                                    layer_block_positions.append((wx, wy, wz))
            
            # Place blocks in batches
            for i in range(0, len(layer_block_positions), max_blocks_per_batch):
                batch_positions = layer_block_positions[i:i+max_blocks_per_batch]
                
                for wx, wy, wz in batch_positions:
                    try:
                        # Safety checks
                        distance_from_origin = abs(wx - origin[0]) + abs(wz - origin[2])
                        if distance_from_origin > 100 or wy < -64 or wy > 320:
                            continue
                        
                        mc.setBlock(wx, wy, wz, block_id)
                        layer_blocks += 1
                        
                    except Exception as block_error:
                        continue
                
                # Small delay between batches
                if len(batch_positions) >= max_blocks_per_batch:
                    time.sleep(0.05)
            
            if layer_blocks > 0:
                layer_time = time.time() - layer_start_time
                try:
                    mc.postToChat(f"Layer {z+1}/{vz_size}: {layer_blocks} blocks ({layer_time:.1f}s)")
                except:
                    pass
                
                print(f"[MINECRAFT] Layer {z+1}/{vz_size}: {layer_blocks} blocks placed")
                
                if delay > 0:
                    time.sleep(delay)
        
        try:
            mc.postToChat(f"✅ Build complete! {total_blocks_scaled} blocks placed")
        except:
            pass
        
        return f"✅ Successfully built {total_voxels:,} voxels ({total_blocks_scaled:,} blocks) in Minecraft!\n\n📊 Build stats:\n• Layers: {vz_size}\n• Scale: {scale}x\n• Block type: {block_type}"
        
    except Exception as e:
        print(f"[MINECRAFT] Build error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Build failed: {str(e)}"

