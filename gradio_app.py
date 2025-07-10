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

# Import voxelization utilities
try:
    from voxel_utils import (
        PYVISTA_AVAILABLE, MINECRAFT_AVAILABLE,
        TORCH_VOXEL_AVAILABLE, voxel_device,
        voxelize_mesh, create_pyvista_voxel_plot,
        create_standalone_html_voxel_plot, get_plane_view, get_height_slices_for_plane, reorient_voxels_for_plane,
        launch_external_voxel_viewer, build_voxels_in_minecraft_robust,
        save_voxel_matrix_to_txt
    )
    print("[INFO] Voxel utilities imported successfully")
except ImportError as e:
    print(f"[WARNING] Failed to import voxel utilities: {e}")
    PYVISTA_AVAILABLE = False
    MINECRAFT_AVAILABLE = False
    TORCH_VOXEL_AVAILABLE = False
    voxel_device = 'cpu'

from hy3dshape.utils import logger
from hy3dpaint.convert_utils import create_glb_with_pbr_materials

# Voxelization functions are now imported from voxel_utils module

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


def build_app():
    title = 'Hunyuan2Minecraft: Textured 3d assets generation to minecraft'
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
                        target_face_num = gr.Slider(minimum=10000, maximum=10000000, value=100000,
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
                    with gr.Tab('Voxelization', id='voxelization_panel', visible=True):
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
                            <li><strong>External Viewer:</strong> Launch PyVista in separate window with full mouse interactivity</li>
                            <li><strong>Minecraft Building:</strong> Build your voxels in Minecraft layer by layer!</li>
                            <li><strong>Statistics:</strong> Resolution, occupied voxels, and fill ratio</li>
                        </ul>
                        <p><em>💡 Tip: Use the External Viewer for the best interactive experience!</em></p>
                        <p><em>Note: Requires PyTorch and GPU for optimal performance.</em></p>
                        </div>
                        """)
                        
                        gr.HTML(f"""
                        <div style="background-color: green; color: white; padding: 5px 10px; border-radius: 3px; margin-bottom: 10px; text-align: center;">
                        <strong>Voxelization Status: Available</strong>
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
                                    value=True, label='Use GPU', visible=False
                                )
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
                                    visible=True,
                                    interactive=False
                                )
                        
                        with gr.Row():
                            height_slices_plot = gr.Image(label='Height Slices Visualization', visible=True)
                            detailed_plot = gr.Image(label='Detailed Analysis', visible=False)
                        
                        with gr.Row():
                            with gr.Column():
                                # Interactive PyVista plot (when available)
                                voxel_interactive_plot = gr.HTML(label='Interactive 3D Voxel Plot', visible=False)
                                # Fallback matplotlib plot  
                                voxel_3d_plot = gr.Image(label='3D Voxel Visualization (Static)', visible=True)                                
                                # External PyVista viewer button
                                external_viewer_btn = gr.Button(
                                    "🚀 Launch External Viewer",
                                    variant="primary",
                                    visible=True,
                                    size="sm"
                                )
                                # Status display for external viewer
                                external_viewer_status = gr.Textbox(
                                    label="External Viewer Status",
                                    visible=True,
                                    interactive=False
                                )
                        
                        # Plane view controls for voxel visualization
                        with gr.Row() as voxel_plane_controls:
                            gr.HTML("""
                            <div style="text-align: center; margin: 10px 0;">
                            <strong>📐 Plane Views</strong><br>
                            <em style="font-size: 12px; color: #666;">Select base plane for visualization and Minecraft building</em>
                            </div>
                            """)
                        
                        with gr.Row() as voxel_plane_buttons:
                            plane_xy_btn = gr.Button("📋 XY Plane (Top)", size="sm", variant="secondary", visible=True)
                            plane_yz_btn = gr.Button("📋 YZ Plane (Side)", size="sm", variant="secondary", visible=True) 
                            plane_xz_btn = gr.Button("📋 XZ Plane (Front)", size="sm", variant="primary", visible=True)
                        
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
                print(f'[GRADIO] Voxelizing uploaded mesh: {mesh_path}')
            elif file_export is not None:
                mesh_path = file_export
                print(f'[GRADIO] Voxelizing exported mesh: {mesh_path}')
            else:
                raise gr.Error('Please either upload a mesh file or export a mesh first using the "Export" tab.')
            
            # Check if mesh file exists
            if not os.path.exists(mesh_path):
                raise gr.Error(f'Mesh file not found: {mesh_path}')
            
            print(f'[GRADIO] Starting voxelization with resolution={resolution}, method={method}')
            
            try:
                stats, height_plot, detailed_plot, voxel_3d_plot, voxels, voxel_html_path = voxelize_mesh(mesh_path, resolution, method, use_gpu)
                
                # Check if voxelization was successful
                if voxels is None:
                    error_msg = stats.get('error', 'Unknown voxelization error') if isinstance(stats, dict) else 'Voxelization failed'
                    raise gr.Error(f'Voxelization failed: {error_msg}')
                
                print(f'[GRADIO] Voxelization completed successfully. Voxel shape: {voxels.shape}, occupied: {voxels.sum()}')
                
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
                
                print(f'[GRADIO] UI updates - height_visible: {height_visible}, detailed_visible: {detailed_visible}, voxel_3d_visible: {voxel_3d_visible}')
                print(f'[GRADIO] UI updates - plane_controls_visible: {plane_controls_visible}, external_viewer_visible: {external_viewer_visible}')
                
                return (
                    stats,  # → voxel_stats
                    gr.update(value=height_plot, visible=height_visible),  # → height_slices_plot
                    gr.update(value=detailed_plot, visible=detailed_visible),  # → detailed_plot
                    gr.update(value=interactive_plot_html, visible=interactive_visible),  # → voxel_interactive_plot
                    gr.update(value=voxel_3d_plot, visible=voxel_3d_visible and not interactive_visible),  # → voxel_3d_plot
                    gr.update(visible=external_viewer_visible),  # → external_viewer_btn (show/hide button)
                    gr.update(visible=False, value=""),  # → external_viewer_status (reset status)
                    voxels,  # → current_voxels (FIXED: store voxel data in state)
                    gr.update(visible=plane_controls_visible),  # → plane_xy_btn (show button)
                    gr.update(visible=plane_controls_visible),  # → plane_yz_btn (show button)
                    gr.update(visible=plane_controls_visible)   # → plane_xz_btn (show button)
                )
                
            except Exception as e:
                print(f'[GRADIO] Voxelization error: {e}')
                import traceback
                traceback.print_exc()
                raise gr.Error(f'Voxelization failed: {str(e)}')

        voxelize_btn.click(
            lambda: gr.update(selected='voxelization_panel'),
            outputs=[tabs_output],
        ).then(
            on_voxelize_click,
            inputs=[mesh_file_upload, file_export, voxel_resolution, voxel_method, use_gpu_voxel],
            outputs=[voxel_stats, height_slices_plot, detailed_plot, voxel_interactive_plot, voxel_3d_plot, external_viewer_btn, external_viewer_status, current_voxels, plane_xy_btn, plane_yz_btn, plane_xz_btn]
        )
        
        # Plane view button event handlers
        def on_plane_click(voxels, plane):
            print(f"[PLANE_VIEW] {plane} button clicked. Voxels type: {type(voxels)}, is None: {voxels is None}")
            if voxels is not None:
                print(f"[PLANE_VIEW] Voxel shape: {voxels.shape}, occupied: {voxels.sum()}")
            
            if voxels is None:
                print(f"[PLANE_VIEW] No voxel data, returning empty updates")
                return gr.update(), gr.update(), plane
                
            try:
                # Generate height slices for the selected plane orientation
                height_slices = get_height_slices_for_plane(voxels, plane)
                print(f"[PLANE_VIEW] Generated {plane} height slices. Result type: {type(height_slices)}, is None: {height_slices is None}")
                
                # Skip 3D plot regeneration during plane switching to avoid PyVista threading issues
                # The 3D plot will remain the same as the original voxelization
                print(f"[PLANE_VIEW] Skipping 3D plot regeneration for {plane} plane to avoid segfault")
                
                return (
                    gr.update(),  # Keep existing 3D voxel plot (don't regenerate)
                    gr.update(value=height_slices) if height_slices is not None else gr.update(),  # Update height slices
                    plane  # Update the selected plane state
                )
            except Exception as e:
                print(f"[PLANE_VIEW] Error generating {plane} plane view: {e}")
                import traceback
                traceback.print_exc()
                return gr.update(), gr.update(), plane
        
        # Connect plane view buttons
        plane_xy_btn.click(
            lambda voxels: on_plane_click(voxels, 'XY'),
            inputs=[current_voxels],
            outputs=[voxel_3d_plot, height_slices_plot, selected_plane]
        ).then(
            lambda: (gr.update(variant="primary"), gr.update(variant="secondary"), gr.update(variant="secondary")),
            outputs=[plane_xy_btn, plane_yz_btn, plane_xz_btn]
        )
        
        plane_yz_btn.click(
            lambda voxels: on_plane_click(voxels, 'YZ'),
            inputs=[current_voxels],
            outputs=[voxel_3d_plot, height_slices_plot, selected_plane]
        ).then(
            lambda: (gr.update(variant="secondary"), gr.update(variant="primary"), gr.update(variant="secondary")),
            outputs=[plane_xy_btn, plane_yz_btn, plane_xz_btn]
        )
        
        plane_xz_btn.click(
            lambda voxels: on_plane_click(voxels, 'XZ'),
            inputs=[current_voxels],
            outputs=[voxel_3d_plot, height_slices_plot, selected_plane]
        ).then(
            lambda: (gr.update(variant="secondary"), gr.update(variant="secondary"), gr.update(variant="primary")),
            outputs=[plane_xy_btn, plane_yz_btn, plane_xz_btn]
        )
        
        # External viewer button event handler
        def on_external_viewer_click(voxels):
            print(f"[EXTERNAL_VIEWER] Button clicked. Voxels type: {type(voxels)}, is None: {voxels is None}")
            if voxels is not None:
                print(f"[EXTERNAL_VIEWER] Voxel shape: {voxels.shape}, occupied: {voxels.sum()}")
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
            print(f"[MINECRAFT] Build button clicked. Voxels type: {type(voxels)}, is None: {voxels is None}")
            if voxels is not None:
                print(f"[MINECRAFT] Voxel shape: {voxels.shape}, occupied: {voxels.sum()}")
            
            if not MINECRAFT_AVAILABLE:
                return gr.update(value="❌ Minecraft Pi API not available. Install with: pip install mcpi", visible=True)
            
            if voxels is None:
                return gr.update(value="❌ No voxel data available.\n\n🔧 To fix this:\n1. Click 'Generate Voxels' button first\n2. Wait for voxelization to complete\n3. Make sure no errors occurred during voxelization\n4. Then try building in Minecraft again", visible=True)
            
            # Safety checks
            total_voxels = int(voxels.sum())
            if total_voxels == 0:
                return gr.update(value="❌ No occupied voxels found in the data.\n\n🔧 This might happen if:\n• The mesh file was empty\n• Voxel resolution is too low\n• The mesh is very small compared to the voxel grid", visible=True)
            
            total_blocks = total_voxels * (scale ** 3)
            if total_blocks > 10000000:
                return gr.update(
                    value=f"❌ Too many blocks ({total_blocks:,}). Maximum recommended: 10,000.\n"
                        f"Try reducing scale (current: {scale}) or voxel resolution.\n"
                        f"Current voxels: {total_voxels:,}", 
                    visible=True
                )
            
            # Use the robust builder with selected plane
            print(f"[MINECRAFT] Starting build with {total_voxels} voxels, scale {scale}, base plane {base_plane}")
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
