"""
Snap Script for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)

"""
import pickle
import torch
import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
    BlendParams,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor)
from tqdm import tqdm
import pytorch3d
from PIL import Image
import numpy as np
import copy
import os
import random
from pytorch3d.structures import Pointclouds
from PIL import Image, ImageDraw, ImageFont
from .lookup import pcd2img_point_occlusion_aware
import plyfile
import numpy as np
import pandas as pd

def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
        return vertices, faces

def get3d_box_from_pcs(pc):
    """
    Given point clouds, return the width, length, and height dimensions of the box that contains the point clouds.
    """
    w = pc[:, 0].max() - pc[:, 0].min()
    l = pc[:, 1].max() - pc[:, 1].min()
    h = pc[:, 2].max() - pc[:, 2].min()
    return w, l, h

def lookat(center, target, up):
    """
    From: LAR-Look-Around-and-Refer
    https://github.com/eslambakr/LAR-Look-Around-and-Refer
    https://github.com/isl-org/Open3D/issues/2338
    https://stackoverflow.com/questions/54897009/look-at-function-returns-a-view-matrix-with-wrong-forward-position-python-im
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    https://www.youtube.com/watch?v=G6skrOtJtbM
    f: forward
    s: right
    u: up
    """
    f = target - center
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = -s
    m[1, :-1] = u
    m[2, :-1] = f
    m[-1, -1] = 1.0

    t = np.matmul(-m[:3, :3], center)
    m[:3, 3] = t

    return m

def get_rid_of_lip(mesh, scan_pc, remove_lip):
    """
    Given a mesh file in PyTorch3D, remove the lip with a predefined height, and return a mesh file in PyTorch3D
    """
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    texture_tensor = mesh.textures.verts_features_packed()
    a = verts
    b = faces
    z_max = scan_pc[:, 2].max()
    idx = a[:, 2] <= z_max - remove_lip

    # Crop the vertices and create an index map
    map_idx = torch.zeros_like(a[:, 0], dtype=torch.long).cuda() - 1
    map_idx[idx] = torch.arange(idx.sum()).cuda()
    a = a[idx]
    texture_tensor = texture_tensor[idx]

    # Crop the triangle surface and update the indices
    b = b[(idx[b[:, 0]] & idx[b[:, 1]] & idx[b[:, 2]])]
    final_b = map_idx[b]

    converted_texture = pytorch3d.renderer.mesh.textures.TexturesVertex(
        [texture_tensor]
    )
    cropped_mesh = pytorch3d.structures.Meshes(
        verts=[a], faces=[final_b], textures=converted_texture
    ).cuda()
    return cropped_mesh, idx

def intrinsic_calibration(point_cloud, pose, width, height):
    """
    Given a predefined pose and point cloud, calibrate the intrinsic matrix to encompass all points in the image
    """

    depth_intrinsic = np.array(
        [
            [577.590698, 0.000000, 318.905426, 0.000000],
            [0.000000, 578.729797, 242.683609, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ]
    )

    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]

    points = np.ones((point_cloud.shape[0], 4))
    points[:, :3] = point_cloud[:, :3]
    inv_pose = np.linalg.inv(np.transpose(pose))
    points_new = np.dot(points, inv_pose)

    point_projected = np.zeros((points_new.shape[0], 2))
    point_projected[:, 0] = points_new[:, 0] * fx / points_new[:, 2] + cx
    point_projected[:, 1] = points_new[:, 1] * fy / points_new[:, 2] + cy
    new_intrinsic = depth_intrinsic.copy()
    cx_new = new_intrinsic[0, 2] - point_projected[:, 0].min()
    cy_new = new_intrinsic[1, 2] - point_projected[:, 1].min()

    point_projected[:, 0] = points_new[:, 0] * fx / points_new[:, 2] + cx_new
    point_projected[:, 1] = points_new[:, 1] * fy / points_new[:, 2] + cy_new

    scale_1 = 1 / point_projected[:, 0].max() * width
    scale_2 = 1 / point_projected[:, 1].max() * height
    scale = scale_1 if scale_1 < scale_2 else scale_2
    fx_new = new_intrinsic[0, 0] * scale
    fy_new = new_intrinsic[1, 1] * scale
    cx_new = cx_new * scale
    cy_new = cy_new * scale

    point_projected_new = np.zeros((points_new.shape[0], 2))
    point_projected_new[:, 0] = points_new[:, 0] * fx_new / points_new[:, 2] + cx_new
    point_projected_new[:, 1] = points_new[:, 1] * fy_new / points_new[:, 2] + cy_new

    assert point_projected_new[:, 0].max() <= width + 1
    assert point_projected_new[:, 0].min() > -0.1
    assert point_projected_new[:, 1].max() <= height + 1
    assert point_projected_new[:, 1].min() > -0.1

    new_intrinsic = depth_intrinsic.copy()

    new_intrinsic[0, 0] = fx_new
    new_intrinsic[1, 1] = fy_new
    new_intrinsic[0, 2] = cx_new
    new_intrinsic[1, 2] = cy_new

    return new_intrinsic

def generate_camera_locations(center, width, length, height, num_split=5):
    """
    Generate camera positions for scene-level images by uniformly placing them on top of the scene
    """
    half_width, half_length, half_height = width / 2, length / 2, height / 2
    top_height = center[2] + half_height

    top_coord = np.linspace(center[0] - half_width, center[0] + half_width, num_split)
    ver_coord = np.linspace(center[1] - half_length, center[1] + half_length, num_split)

    camera_pos_from = []
    for x_coord in top_coord:
        camera_pos_from.append([x_coord, ver_coord[0], top_height])
        camera_pos_from.append([x_coord, ver_coord[-1], top_height])
    for y_coord in ver_coord[1:-1]:
        camera_pos_from.append([top_coord[0], y_coord, top_height])
        camera_pos_from.append([top_coord[-1], y_coord, top_height])
    return camera_pos_from

def render_mesh(pose, intrin_path, image_width, image_height, mesh, name):
    """
    Given the mesh in PyTorch3D format, render images with a defined pose, intrinsic matrix, and image dimensions
    """

    device = "cuda"
    background_color = (1.0, 1.0, 1.0)
    intrinsic_matrix = torch.zeros([4, 4])
    intrinsic_matrix[3, 3] = 1

    intrinsic_matrix_load = intrin_path
    intrinsic_matrix_load_torch = torch.from_numpy(intrinsic_matrix_load)
    intrinsic_matrix[:3, :3] = intrinsic_matrix_load_torch
    extrinsic_load = pose
    camera_to_world = torch.from_numpy(extrinsic_load)
    world_to_camera = torch.inverse(camera_to_world)
    fx, fy, cx, cy = (
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )
    width, height = image_width, image_height
    rotation_matrix = world_to_camera[:3, :3].permute(1, 0).unsqueeze(0)
    translation_vector = world_to_camera[:3, 3].reshape(-1, 1).permute(1, 0)
    focal_length = -torch.tensor([[fx, fy]])
    principal_point = torch.tensor([[cx, cy]])
    camera = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=rotation_matrix,
        T=translation_vector,
        image_size=torch.tensor([[height, width]]),
        in_ndc=False,
        device=device,
    )
    lights = AmbientLights(device=device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardPhongShader(
            blend_params=BlendParams(background_color=background_color),
            device=lights.device,
            cameras=camera,
            lights=lights,
        ),
    )
    rendered_image = renderer(mesh)
    rendered_image = rendered_image[0].cpu().numpy()
    color = rendered_image[..., :3]
    color_image = Image.fromarray((color * 255).astype(np.uint8))
    color_image.save(name)

def render_with_results(pose, intrin_path, image_width, image_height, scan_pc, mask2pxiel_map, detected_label, color_list_save, name):
    """
    Given the mesh in PyTorch3D format, render images with a defined pose, intrinsic matrix, and image dimensions
    """
    device = "cuda"
    point_cloud = scan_pc.cuda()
    intrinsic_matrix = torch.zeros([4, 4])
    intrinsic_matrix[3, 3] = 1
    intrinsic_matrix_load = intrin_path
    intrinsic_matrix_load_torch = torch.from_numpy(intrinsic_matrix_load)
    intrinsic_matrix[:3, :3] = intrinsic_matrix_load_torch
    extrinsic_load = pose

    xyz_rgb = point_cloud
    intrinsic_matrix = torch.zeros([4, 4])
    intrinsic_matrix[3, 3] = 1
    intrinsic_matrix_load = intrin_path
    intrinsic_matrix_load_torch = torch.from_numpy(intrinsic_matrix_load)
    intrinsic_matrix[:3, :3] = intrinsic_matrix_load_torch
    extrinsic_load = pose
    camera_to_world = torch.from_numpy(extrinsic_load)
    world_to_camera = torch.inverse(camera_to_world)
    fx, fy, cx, cy = (
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )
    width, height = image_width, image_height
    rotation_matrix = world_to_camera[:3, :3].permute(1, 0).unsqueeze(0)
    translation_vector = world_to_camera[:3, 3].reshape(-1, 1).permute(1, 0)
    focal_length = -torch.tensor([[fx, fy]])
    principal_point = torch.tensor([[cx, cy]])
    
    camera = PerspectiveCameras(focal_length=focal_length,
                                principal_point=principal_point,
                                R=rotation_matrix,
                                T=translation_vector,
                                image_size=torch.tensor([[height, width]]),
                                in_ndc=False,
                                device=device)
    lights = AmbientLights(device=device)
    
    raster_settings = PointsRasterizationSettings(
        image_size=(height, width), 
        radius = 0.007,
        points_per_pixel = 10
        )
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
    
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color = 255)
    )
    point_cloud = Pointclouds(points=[scan_pc[:,:3]], features=[scan_pc[:,3:6]]).cuda()

    rendered_image = renderer(point_cloud)
    rendered_image = rendered_image[0].cpu().numpy()
    color = rendered_image[..., :3]
    color_image = Image.fromarray((color).astype(np.uint8))
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 25, encoding="unic")
    draw = ImageDraw.Draw(color_image)  
    
    location = []
    mask_map_numpy = mask2pxiel_map.cpu().numpy()
    unique_indices = np.unique(mask_map_numpy)
    for i, idx in enumerate(unique_indices):
        if idx ==999:
            continue
        non_zero_indices = np.argwhere(mask_map_numpy == idx)
        # Calculate the mean of x-coordinates and y-coordinates
        center_x = np.mean(non_zero_indices[:, 1])
        center_y = np.mean(non_zero_indices[:, 0])
        location.append([center_x, center_y])


    for position, text, color in zip(location, detected_label,color_list_save):
        color_work = (color[0], color[1], color[2])
        bbox = draw.textbbox(position, text, font = font)
        draw.rectangle(bbox, fill="white")
        draw.text(position, text, fill=color_work, font=font, stroke_fill="black")

    color_image.save(name)

def render_pcd(pose, intrinsic, image_width, image_height, pcd, name):
    """
    Given the torch.tensor pcd, render images with a defined pose, intrinsic matrix, and image dimensions
    """

    device = "cuda"
    intrinsic_matrix = torch.zeros([4, 4])
    intrinsic_matrix[3, 3] = 1
    point_cloud = Pointclouds(points=[pcd[:,:3]], features=[pcd[:,3:6]]).cuda()
    intrinsic_matrix_torch = torch.from_numpy(intrinsic)
    intrinsic_matrix[:3, :3] = intrinsic_matrix_torch
    camera_to_world = torch.from_numpy(pose)
    world_to_camera = torch.inverse(camera_to_world)
    fx, fy, cx, cy = (
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )
    width, height = image_width, image_height
    rotation_matrix = world_to_camera[:3, :3].permute(1, 0).unsqueeze(0)
    translation_vector = world_to_camera[:3, 3].reshape(-1, 1).permute(1, 0)
    focal_length = -torch.tensor([[fx, fy]])
    principal_point = torch.tensor([[cx, cy]])
    camera = PerspectiveCameras(focal_length=focal_length,
                                    principal_point=principal_point,
                                    R=rotation_matrix,
                                    T=translation_vector,
                                    image_size=torch.tensor([[height, width]]),
                                    in_ndc=False,
                                    device=device)
        
    raster_settings = PointsRasterizationSettings(
            image_size=(height, width), 
            radius = 0.007,
            points_per_pixel = 10
            )

    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings) 
    renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color = 255)
        )
    rendered_image = renderer(point_cloud)
    rendered_image = rendered_image[0].cpu().numpy()
    color = rendered_image[..., :3]
    color_image = Image.fromarray((color).astype(np.uint8))
    color_image.save(name)

def image_generation_pcd(scan_pc, image_width, image_height, scene_name, folder_saved, adjust_camera=[1, 0.1, 0.3]):
 
    """
    Given a pcd numpy array, generate synthetic scene-level images
    """
    
    lift_cam, zoomout, remove_lip = adjust_camera
    
    # get rid of lip
    z_max = scan_pc[:, 2].max()
    idx_remained = scan_pc[:, 2] <= (z_max - remove_lip)
    scan_pc = scan_pc[idx_remained,:]

    w, l, h = get3d_box_from_pcs(scan_pc)
    scene_center = np.array([scan_pc[:, 0].max() - w / 2, scan_pc[:, 1].max() - l / 2, scan_pc[:, 2].max() - h / 2,])
    zoom_factor = 1 + zoomout
    w, l, h = w * zoom_factor, l * zoom_factor, h * zoom_factor
    camera_locations = generate_camera_locations(scene_center, w, l, h, 5)

    for i in tqdm(range(len(camera_locations))):
        camera_location = camera_locations[i]
        org_camera_pos = copy.deepcopy(camera_location)
        camera_location[-1] = org_camera_pos[-1] + lift_cam  # lift the camera
        target_location = scene_center
        up_vector = np.array([0, 0, -1])
        pose_matrix = lookat(camera_location, target_location, up_vector)
        pose_matrix_calibrated = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
        intrinsic_calibrated = intrinsic_calibration(scan_pc, pose_matrix_calibrated, image_width, image_height)

        # save pose and intrinsic
        intrinsic_folder = f"{folder_saved}/{scene_name}/intrinsic"
        pose_folder = f"{folder_saved}/{scene_name}/pose"
        image_folder = f"{folder_saved}/{scene_name}/image"
        if not os.path.exists(intrinsic_folder): os.makedirs(intrinsic_folder)
        if not os.path.exists(pose_folder): os.makedirs(pose_folder)
        if not os.path.exists(image_folder): os.makedirs(image_folder)
        np.save(f"{intrinsic_folder}/intrinsic_calibrated_angle_{i}.npy",intrinsic_calibrated)
        np.save(f"{pose_folder}/pose_matrix_calibrated_angle_{i}.npy",pose_matrix_calibrated)
        image_folder = f"{folder_saved}/{scene_name}/image"
        # render image and save
        render_pcd(
            pose_matrix_calibrated,
            intrinsic_calibrated[:3, :3],
            image_width,
            image_height,
            scan_pc,
            f"{image_folder}/image_rendered_angle_{i}.png")

    print(f"All snap images for {scene_name} are saved at '{image_folder}'")

    return False

def image_generation_mesh(mesh_file, image_width, image_height, scene_name, folder_saved, adjust_camera=[1, 0.1, 0.3]):
    
    """
    Given a .ply mesh file path, generate synthetic scene-level images
    """
    device = "cuda"

    pt3d_io = IO()
    mesh = pt3d_io.load_mesh(mesh_file, device=device)
    scan_pc = mesh.verts_packed().cpu().numpy()
    lift_cam, zoomout, remove_lip = adjust_camera

    mesh, idx_remained = get_rid_of_lip(mesh, scan_pc, remove_lip)


    w, l, h = get3d_box_from_pcs(scan_pc)
    scene_center = np.array(
        [
            scan_pc[:, 0].max() - w / 2,
            scan_pc[:, 1].max() - l / 2,
            scan_pc[:, 2].max() - h / 2,
        ]
    )

    zoom_factor = 1 + zoomout
    w, l, h = w * zoom_factor, l * zoom_factor, h * zoom_factor
    camera_locations = generate_camera_locations(scene_center, w, l, h, 5)

    for i in tqdm(range(len(camera_locations))):
        camera_location = camera_locations[i]
        org_camera_pos = copy.deepcopy(camera_location)
        camera_location[-1] = org_camera_pos[-1] + lift_cam  # lift the camera
        target_location = scene_center
        up_vector = np.array([0, 0, -1])
        pose_matrix = lookat(camera_location, target_location, up_vector)
        pose_matrix_calibrated = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
        intrinsic_calibrated = intrinsic_calibration(
            scan_pc, pose_matrix_calibrated, image_width, image_height
        )
        intrinsic_folder = f"{folder_saved}/{scene_name}/intrinsic"
        pose_folder = f"{folder_saved}/{scene_name}/pose"
        image_folder = f"{folder_saved}/{scene_name}/image"
        if not os.path.exists(intrinsic_folder):
            os.makedirs(intrinsic_folder)
        if not os.path.exists(pose_folder):
            os.makedirs(pose_folder)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        np.save(
            f"{intrinsic_folder}/intrinsic_calibrated_angle_{i}.npy",
            intrinsic_calibrated,
        )
        np.save(
            f"{pose_folder}/pose_matrix_calibrated_angle_{i}.npy",
            pose_matrix_calibrated,
        )
        render_mesh(
            pose_matrix_calibrated,
            intrinsic_calibrated[:3, :3],
            image_width,
            image_height,
            mesh,
            f"{image_folder}/image_rendered_angle_{i}.png",
        )
    print(f"All snap images for {scene_name} are saved at '{image_folder}'")

    return False

def save_results_2d(scan_pc, image_width, image_height, scene_name, folder_saved, adjust_camera, detection_results):
    
    """
    Given a pcd numpy array, and detection results, generate synthetic scene-level images with labels
    """

    detected_mask_bin, detected_label = detection_results
    
    # colorcode masks
    scan_pc = scan_pc.float()
    color_list_save = []
    for idx_mask in range(detected_mask_bin.shape[1]):
        indices = detected_mask_bin[:, idx_mask]!=0
        random_color = lambda: random.randint(0, 255)
        color =  torch.tensor([random_color(), random_color(), random_color()])
        color_list_save.append(color)
        obj = scan_pc[indices,:3]
        scan_pc[indices,3:6] = color.float()
    
    lift_cam, zoomout, remove_lip = adjust_camera
    
    # get rid of lip
    z_max = scan_pc[:, 2].max()
    idx_remained = scan_pc[:, 2] <= (z_max - remove_lip)
    scan_pc = scan_pc[idx_remained,:]
    w, l, h = get3d_box_from_pcs(scan_pc)
    scene_center = np.array(
        [
            scan_pc[:, 0].max() - w / 2,
            scan_pc[:, 1].max() - l / 2,
            scan_pc[:, 2].max() - h / 2,
        ]
    )

    detected_mask_bin_remove_lip = detected_mask_bin[idx_remained,:]
    zoom_factor = 1 + zoomout
    w, l, h = w * zoom_factor, l * zoom_factor, h * zoom_factor
    camera_locations = generate_camera_locations(scene_center, w, l, h, 5)
    image_folder = f"{folder_saved}/{scene_name}/image"
    if not os.path.exists(image_folder): os.makedirs(image_folder)

    for i in tqdm(range(len(camera_locations))):
        camera_location = camera_locations[i]
        org_camera_pos = copy.deepcopy(camera_location)
        camera_location[-1] = org_camera_pos[-1] + lift_cam  # lift the camera
        target_location = scene_center
        up_vector = np.array([0, 0, -1])
        pose_matrix = lookat(camera_location, target_location, up_vector)
        pose_matrix_calibrated = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
        intrinsic_calibrated = intrinsic_calibration(
            scan_pc, pose_matrix_calibrated, image_width, image_height
        )
        mask2pxiel_map = pcd2img_point_occlusion_aware(scan_pc, detected_mask_bin_remove_lip, pose_matrix_calibrated, intrinsic_calibrated, image_width, image_height, 2)
        render_with_results(pose_matrix_calibrated, intrinsic_calibrated[:3, :3], image_width, image_height, scan_pc, mask2pxiel_map, detected_label, color_list_save, f"{image_folder}/openworld_instance_seg_result_{i}.png")
    
def main():
    image_width = 2000
    image_height = 2000
    mesh_file = "../data_input/scannet-raw/scans/scene0000_00/scene0000_00_vh_clean_2.ply"
    folder_saved = "../export"
    folder_saved_pcd = "../export_pcd"
    folder_saved_results = "../export_results"
    dataset_name = 'scannet'

    # adjust the camera
    lift_cam = 1  # Increase the value to lift the camera
    zoomout = 0.1  # Adding a value allows for viewing the scene with a larger angle
    remove_lip = (
        0.3  # To remove the ceiling of scene for better image capturing, if applicable
    )
    
    pcd, _ = read_plymesh(mesh_file)
    detected_mask_bin = torch.load("../detected_mask_bin.pth")
    with open("../detected_label.txt", "rb") as fp:   # Unpickling
        detected_label = pickle.load(fp)
    scene_name = os.path.basename(mesh_file).split(".")[0]
    adjust_camera = [lift_cam, zoomout, remove_lip]
    xyz, rgb = pcd[:,:3], pcd[:,3:6]
    scan_pc = np.hstack([xyz, rgb])
    detection_results = (detected_mask_bin, detected_label)
    adjust_camera = [10, 0.5, 0.6]
    # image_generation_mesh(mesh_file, image_width, image_height, scene_name, folder_saved, adjust_camera=adjust_camera)
    scan_pc = torch.from_numpy(scan_pc)
    print(scan_pc.shape)
    image_generation_pcd(scan_pc, image_width, image_height, scene_name, folder_saved_pcd, adjust_camera)

    save_results_2d(scan_pc, image_width, image_height, scene_name, folder_saved_results, adjust_camera, detection_results)


    # if dataset_name in ["replica", "scannet"]:
    #     adjust_camera = [2, 0.5, 0.6]
    #     image_generation_mesh(
    #         mesh_file, image_width, image_height, scene_name, folder_saved, adjust_camera=adjust_camera
    #     )
    # elif dataset_name == "mattarport3d":
    #     adjust_camera = [2, 0.1, 0.3]
    #     pcd, _ = read_plymesh(mesh_file)
    #     xyz, rgb = pcd[:,:3], pcd[:,8:11]
    #     scan_pc = np.hstack([xyz,rgb])
    #     image_generation_pcd(scan_pc, image_width, image_height, scene_name, folder_saved, adjust_camera=adjust_camera)
    # elif dataset_name == "s3dis":    
    #     adjust_camera = [2, 0.1, 0.3]
    #     pcd = np.load(mesh_file)
    #     xyz, rgb = pcd[:,:3], pcd[:,3:6]
    #     scan_pc = np.hstack([xyz,rgb])
    #     image_generation_pcd(scan_pc, image_width, image_height, scene_name, folder_saved, adjust_camera=adjust_camera)
    # elif dataset_name == "stpls3d":    
    #     adjust_camera = [10, 2, 0.3]
    #     pcd = np.load(mesh_file)
    #     xyz, rgb = pcd[:,:3], pcd[:,3:6]
    #     scan_pc = np.hstack([xyz,rgb])
    #     image_generation_pcd(scan_pc, image_width, image_height, scene_name, folder_saved, adjust_camera=adjust_camera)

if __name__ == "__main__":
    main()
