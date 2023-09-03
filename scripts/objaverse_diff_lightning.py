# isort: off
import blenderproc as bproc
from blenderproc.python.utility.SetupUtility import SetupUtility
from blenderproc.python.utility.Utility import Utility
import bpy

# isort: on
import argparse
import os
import json
from pathlib import Path
import math
import numpy as np
from mathutils import Vector
from scipy.spatial.transform import Rotation as R
import tempfile
import random


def disable_all_denoiser():
    """ Disables all denoiser.

    At the moment this includes the cycles and the intel denoiser.
    """
    # Disable cycles denoiser
    bpy.context.view_layer.cycles.use_denoising = False
    bpy.context.scene.cycles.use_denoising = False

    # Disable intel denoiser
    if bpy.context.scene.use_nodes:
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        # Go through all existing denoiser nodes
        for denoiser_node in Utility.get_nodes_with_type(nodes, 'CompositorNodeDenoise'):
            in_node = denoiser_node.inputs['Image']
            out_node = denoiser_node.outputs['Image']

            # If it is fully included into the node tree
            if in_node.is_linked and out_node.is_linked:
                # There is always only one input link
                in_link = in_node.links[0]
                # Connect from_socket of the incoming link with all to_sockets of the out going links
                for link in out_node.links:
                    links.new(in_link.from_socket, link.to_socket)

            # Finally remove the denoiser node
            nodes.remove(denoiser_node)

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # r, i, j, k = np.unbind(quaternions, -1)
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = np.linalg.norm(axis_angle, ord=2, axis = -1, keepdims = True)
    # angles = axis_angle.norm(p = 2, dim = -1, keepdim = True)
    # angles = np.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

THETA = np.arctan(32/2/35)
BETA = np.arctan(np.sqrt(2))

DELTA_AZI  = np.radians([30 + idx*60 for idx in range(6)])
DELTA_ELEV = np.radians([30, -30]*3)
FIXED_ELEV = np.radians([60, 110]*3)

# ---------------------------------------------------------------------------- #
# Arguments
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--object-path", type=str, required=True)
parser.add_argument("--use-gpu", type=int, default = 1)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--scale", type=float, default=1.)
parser.add_argument("--radius", type=float, default=1.)
parser.add_argument("--num-views", type=int, default=3)
parser.add_argument("--seed", type=int)
parser.add_argument("--engine", type=str, default="cycles")
parser.add_argument("--light-energy", type=float, default=10)
parser.add_argument("--no-depth", type=int, default=0)
parser.add_argument("--no-normal", type=int, default=1)
parser.add_argument("--random", type=int, default=0)
parser.add_argument("--random_angle", type=int, default=0)
parser.add_argument("--bbox_init", action="store_true")
parser.add_argument("--metadata_dir", type=str, default="/datasets-slow1/Objaverse/rawdata/hf-objaverse-v1/metadata")
parser.add_argument("--glbdata_dir", type=str, default="/datasets-slow1/Objaverse/rawdata/hf-objaverse-v1/glbs")
parser.add_argument("--objdata_dir", type=str, default="/objaverse-processed/blender_converted_objs")
parser.add_argument("--hdri_dir", type=str, default="/zhuoyang-fast-vol/zero123++_rendering/haven")

args = parser.parse_args()

n_threads = 16
uid = args.object_path.split("/")[-1]

args.output_dir = f'{args.output_dir}/{args.object_path}'

np.random.seed(args.seed)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------- #
# Initialize bproc
# ---------------------------------------------------------------------------- #
bproc.init() #compute_device = 'GPU')

# Renderer setting (following GET3D)
if args.engine == "cycles":
    #bproc.renderer.set_render_devices('GPU')
    if args.use_gpu:
        bpy.context.scene.cycles.device = "GPU"
        # bproc.renderer.set_render_devices(use_only_gpu=True)
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            if d["name"] == "Intel Xeon Platinum 8255C CPU @ 2.50GHz":
                d["use"] = 0
            elif d["name"] == "AMD EPYC 7502 32-Core Processor":
                d["use"] = 0
            else:
                d["use"] = 1
            print(d["name"], d["use"])
    else:
        #bproc.python.utility.Initializer.init() #compute_device='CPU') #, compute_device_type=None, use_experimental_features=False, clean_up_scene=True)
        bproc.renderer.set_render_devices(use_only_cpu=True)
        bproc.renderer.set_cpu_threads(n_threads)
    #bpy.context.preferences.addons["cycles"].preferences.get_devices()
    #bproc.renderer.set_denoiser("OPTIX")
    disable_all_denoiser()
    bpy.context.scene.use_nodes = False
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.view_layer.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.context.scene.cycles.filter_width = 1.0
    bpy.context.scene.cycles.denoising_prefilter = 'FAST'
    bpy.context.view_layer.use_pass_normal = False
    bpy.context.view_layer.use_pass_diffuse_color = False
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.set_light_bounces(
        diffuse_bounces=1,
        glossy_bounces=1,
        transmission_bounces=3,
        transparent_max_bounces=3,
    )
    bproc.renderer.set_max_amount_of_samples(32)
else:
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bproc.renderer.set_output_format(enable_transparency=True)

# ---------------------------------------------------------------------------- #
# Load model
# ---------------------------------------------------------------------------- #
fl = open(os.path.join(args.metadata_dir, args.object_path.split("/")[0] + ".json"), "r")
meta_load = json.load(fl)
animation_count = meta_load[uid]["animationCount"]
# print(animation_count)

if animation_count == 0:
    bpy.ops.import_scene.gltf(filepath = os.path.join(args.glbdata_dir, f"{args.object_path}.glb"), merge_vertices=True)
else:
    objs = bproc.loader.load_obj(
        os.path.join(args.objdata_dir, args.object_path, f"{uid}.obj"),
        use_legacy_obj_import=True,
        use_edges=False,
        use_smooth_groups=False,
        split_mode="OFF",
    )
    assert len(objs) == 1, len(objs)
    obj = objs[0]

# # Set the frame to the last frame of your animation
bpy.context.scene.frame_set(bpy.context.scene.frame_end)

bproc.utility.reset_keyframes()

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        # print("loaded obj:", obj)
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def get_center(l):
    return (max(l) + min(l)) / 2 if l else 0.0

def scene_sphere(single_obj=None, ignore_matrix=False):
    found = False
    points_co_global = []
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        mesh = obj.data
        for vertex in mesh.vertices:
            vertex_co = vertex.co
            if not ignore_matrix:
                vertex_co = obj.matrix_world @ vertex_co
            points_co_global.extend([vertex_co])
    if not found:
        raise RuntimeError("no objects in scene to compute bounding sphere for")
    x, y, z = [[point_co[i] for point_co in points_co_global] for i in range(3)]
    b_sphere_center = Vector([get_center(axis) for axis in [x, y, z]]) if (x and y and z) else None
    b_sphere_radius = max(((point - b_sphere_center) for point in points_co_global)) if b_sphere_center else None
    return b_sphere_center, b_sphere_radius.length

if not args.bbox_init:
    center, sphere_radius = scene_sphere()
    scale = 0.5 / sphere_radius
else:
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
for obj in scene_root_objects():
    obj.scale = obj.scale * scale * args.scale
# Apply scale to matrix_world.
bpy.context.view_layer.update()

if not args.bbox_init:
    new_center, _ = scene_sphere()
    offset = -new_center
else:
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
for obj in scene_root_objects():
    obj.matrix_world.translation += offset
bpy.ops.object.select_all(action="DESELECT")

# ---------------------------------------------------------------------------- #
# Render
# ---------------------------------------------------------------------------- #

# Set rendering parameters
fovy = np.arctan(32 / 2 / 35) * 2
bproc.camera.set_intrinsics_from_blender_params(fovy, lens_unit="FOV")
bproc.camera.set_resolution(args.resolution, args.resolution)

bbox_min, bbox_max = scene_bbox()
aabb = [np.array(bbox_min).tolist(), np.array(bbox_max).tolist()]

meta = {"uid": uid, "animationCount": animation_count, "scale": scale, "center": np.array(new_center).tolist(), "resolution": args.resolution, "fovy": fovy,  "bbox": aabb}

# Sample camera pose
if not args.no_depth:
    bproc.renderer.enable_depth_output(False, output_dir=str(output_dir), file_prefix="depth_")

if not args.no_normal:
    bproc.renderer.enable_normals_output(output_dir=str(output_dir))

poi = np.zeros(3)
for i in range(args.num_views):
    bproc.utility.reset_keyframes()
    meta_sample = {}
    frames = []

    if not args.bbox_init:
        # camera radius for bounding sphere normalization (sphere radius = 0.5)
        # args.radius = 0.5 / np.sin(THETA) * random.uniform(0.9, 1.1)
        args.radius = 0.5 / np.sin(THETA) * random.uniform(0.9, 1.3)
    else:
        # camera radius for bounding box normalization (box side length = 1)
        # args.radius = np.sqrt(3)/2 / np.sin(THETA) * random.uniform(0.9, 1.1)
        args.radius = np.sqrt(3)/2 / np.sin(THETA) * random.uniform(0.9, 1.3)
    
    # Set a random hdri from the given haven directory as background

    haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(args.hdri_dir)
    HDRI_id = haven_hdri_path.split('/')[-1].split('.')[0]
    rand_strength = np.random.uniform(1.5, 2)
    print("Using HDRI: ", HDRI_id, "with strength = ", rand_strength)
    # Rotate the HDRI by a random angle w.r.t. the upwards axis (y)
    rand_rotation_euler = [0, np.random.uniform(-np.pi, np.pi), 0]
    bproc.world.set_world_background_hdr_img(haven_hdri_path, 
                                             strength=rand_strength, 
                                             rotation_euler=rand_rotation_euler)

    hdri_info = dict(hdri_id=HDRI_id, strength=rand_strength, rotation_euler=rand_rotation_euler)

    # Sample random camera location above objects
    azimuth = np.random.uniform(0, 2 * np.pi)
    # elevation = np.arccos(np.random.uniform(-1, 1))
    elevation = np.random.uniform(90-45, 90+10) * np.pi/180

    x = np.cos(azimuth) * np.sin(elevation)
    y = np.sin(azimuth) * np.sin(elevation)
    z = np.cos(elevation)
    location = np.array([x, y, z]) * args.radius

    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

    # Add homogeneous cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix, frame=0)

    frame = dict(
        transform_matrix=cam2world_matrix.tolist(),
        azimuth=azimuth,
        elevation=elevation,
    )
    meta_sample["sample_frame"] = frame

    for delta_idx in range(6):
        azimuth_ = azimuth + DELTA_AZI[delta_idx]
        elevation_ = FIXED_ELEV[delta_idx]

        x = np.cos(azimuth_) * np.sin(elevation_)
        y = np.sin(azimuth_) * np.sin(elevation_)
        z = np.cos(elevation_)
        location = np.array([x, y, z]) * args.radius

        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

        # Add homogeneous cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        # bproc.camera.add_camera_pose(cam2world_matrix, frame=len(frames))

        bproc.camera.add_camera_pose(cam2world_matrix, frame=1+delta_idx)

        frame = dict(
            transform_matrix=cam2world_matrix.tolist(),
            azimuth=azimuth_,
            elevation=elevation_,
        )
        frames.append(frame)

    # Render RGB images
    bproc.renderer.set_cpu_threads(n_threads)
    # data = bproc.renderer.render()
    data = bproc.renderer.render(output_dir=str(output_dir), return_data=False)

    for index in range(7):
        source_depth_fn = os.path.join(output_dir, f"rgb_{index:04d}.png")
        if index == 0:
            target_depth_fn = os.path.join(output_dir, f"color_sample_{i:04d}.png")
        else:
            target_depth_fn = os.path.join(output_dir, f"color_sample_{i:04d}_view_{index-1:04d}.png")
        os.rename(source_depth_fn, target_depth_fn)
    for index in range(7):
        source_depth_fn = os.path.join(output_dir, f"depth_{index:04d}.exr")
        if index == 0:
            target_depth_fn = os.path.join(output_dir, f"depth_sample_{i:04d}.exr")
        else:
            target_depth_fn = os.path.join(output_dir, f"depth_sample_{i:04d}_view_{index-1:04d}.exr")
        os.rename(source_depth_fn, target_depth_fn)

    meta_sample["radius"] = args.radius
    meta_sample["hdri"] = hdri_info
    meta_sample["view_frames"] = frames
    meta[f"sample_{i}"] = meta_sample

dumpmeta_dir = os.path.join(output_dir, "meta.json")
with open(dumpmeta_dir, "w") as f:
    json.dump(meta, f, indent=4)
