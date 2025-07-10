# PyTorch GPU-Optimized Voxelization
# Install: pip install torch torchvision trimesh numpy matplotlib

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import time
from typing import Optional, Tuple
import PIL

# PyTorch imports
import torch
import torch.nn.functional as F

device = torch.device('cuda')
print(f"[INFO] Using device: {device}")
print(f"[INFO] GPU: {torch.cuda.get_device_name()}")
print(f"[INFO] CUDA version: {torch.version.cuda}")
print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
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
    Fast GPU-based mesh voxelization using PyTorch
    """
    print(f"[INFO] PyTorch voxelization on {device}")
    
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
    voxel_size = (resolution - 1) / max_size
    
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
    Ultra-fast approximate voxelization using rasterization-like approach
    """
    print(f"[INFO] Fast PyTorch voxelization on {device}")
    
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
    
    # For each triangle, fill its bounding box (fast approximation)
    for i in range(faces.shape[0]):
        x_min, y_min, z_min = tri_min[i]
        x_max, y_max, z_max = tri_max[i]
        
        # Fill bounding box (conservative voxelization)
        voxel_grid[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = True
    
    return voxel_grid.cpu().numpy()

def pytorch_mesh_to_voxels(mesh_path: str, resolution: int = 32, 
                          method: str = 'accurate', use_gpu: bool = True) -> np.ndarray:
    """
    PyTorch-accelerated mesh to voxel conversion.
    
    Args:
        mesh_path: Path to your 3D file
        resolution: Number of voxels along each axis
        method: 'accurate' or 'fast'
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        A boolean array of shape (resolution, resolution, resolution)
    """
    start_time = time.time()
    
    # Always use CUDA device
    device = 'cuda'
    print(f"[INFO] Using CUDA processing")
    
    # Load mesh
    print(f"[INFO] Loading mesh: {mesh_path}")
    load_start = time.time()
    
    raw = trimesh.load(mesh_path)
    if isinstance(raw, trimesh.Scene):
        if not raw.geometry:
            raise ValueError(f"No geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(raw.geometry.values()))
    else:
        mesh = raw
    
    load_time = time.time() - load_start
    print(f"[INFO] Mesh loaded in {load_time:.3f}s - {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    
    # Pre-processing
    if not mesh.is_watertight:
        filled = mesh.fill_holes()
        print(f"[DEBUG] Filled {filled} holes; watertight: {mesh.is_watertight}")
    
    # Voxelization
    voxel_start = time.time()
    
    if method == 'fast':
        voxel_grid = pytorch_voxelize_fast(mesh.vertices, mesh.faces, resolution, device)
    elif method == 'accurate':
        voxel_grid = pytorch_voxelize_mesh(mesh.vertices, mesh.faces, resolution, device)
    else:
        # Fallback to trimesh
        print("[INFO] Using trimesh fallback")
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        pitch = size.max() / (resolution - 1)
        mesh.apply_translation(-bounds[0])
        vox = mesh.voxelized(pitch)
        voxel_grid = vox.matrix.astype(bool)
    
    voxel_time = time.time() - voxel_start
    
    # Post-processing: center the voxel grid
    post_start = time.time()
    
    # Trim oversized dimensions
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
    
    post_time = time.time() - post_start
    total_time = time.time() - start_time
    
    print(f"[INFO] Voxelization completed in {voxel_time:.3f}s")
    print(f"[INFO] Post-processing completed in {post_time:.3f}s")
    print(f"[INFO] Total processing time: {total_time:.3f}s")
    print(f"[INFO] Voxel grid: {final_grid.shape}, occupied: {final_grid.sum()} voxels")
    
    return final_grid



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
        img = PIL.Image.open(buf)
        return np.array(img)
        
    except Exception as e:
        print(f"Error creating detailed plot: {e}")
        return None

def create_height_animation_frames(voxels: np.ndarray, output_dir: str = "height_frames"):
    """
    Create individual frames for each height slice (useful for creating animations)
    
    Args:
        voxels: 3D boolean array of voxels
        output_dir: Directory to save frame images
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    height = voxels.shape[1]
    
    print(f"[INFO] Creating {height} animation frames in '{output_dir}'")
    
    for y in range(height):
        slice_data = voxels[:, y, :]
        
        if slice_data.sum() == 0:
            continue  # Skip empty slices
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(slice_data.T, origin='lower', cmap='Blues',
                      interpolation='nearest', aspect='equal')
        
        ax.set_title(f'Height Y={y} ({slice_data.sum()} voxels)', fontsize=16)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Z', fontsize=14)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Save frame
        frame_path = os.path.join(output_dir, f'height_{y:03d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"[INFO] Animation frames saved to '{output_dir}'")
    print(f"[INFO] To create a GIF: ffmpeg -r 10 -i {output_dir}/height_%03d.png output.gif")

def benchmark_methods(mesh_path: str, resolution: int = 128):
    """
    Benchmark different voxelization methods
    """
    print("=== Benchmarking Voxelization Methods ===")
    methods = []
    times = []
    
    device = 'cuda'
    print(f"[INFO] Using CUDA for benchmarking")
    
    # GPU Fast method
    print("\n--- PyTorch GPU Fast ---")
    start = time.time()
    voxels_fast = pytorch_mesh_to_voxels(mesh_path, resolution, 'fast', True)
    time_fast = time.time() - start
    methods.append("PyTorch GPU Fast")
    times.append(time_fast)
    
    # GPU Accurate method
    print("\n--- PyTorch GPU Accurate ---")
    start = time.time()
    voxels_accurate = pytorch_mesh_to_voxels(mesh_path, resolution, 'accurate', True)
    time_accurate = time.time() - start
    methods.append("PyTorch GPU Accurate")
    times.append(time_accurate)
    
    # Print results
    print(f"\n=== Benchmark Results (Resolution: {resolution}³) ===")
    for method, time_taken in zip(methods, times):
        print(f"{method:20}: {time_taken:.3f}s")
    
    if len(times) > 1:
        speedup = max(times) / min(times)
        print(f"Best speedup: {speedup:.1f}x")
    
    # Return the accurate result for visualization
    return voxels_accurate

# Main execution
if __name__ == "__main__":
    mesh_file = 'Happy_Sitting_Planter.stl'
    resolution = 128
    
    print("=== PyTorch GPU-Optimized Voxelization ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    try:
        # Run benchmark
        voxels = benchmark_methods(mesh_file, resolution)
        plot_all_height_slices(voxels, "PyTorch GPU Voxelization")
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc() 