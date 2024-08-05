"""
Snap Script for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)

"""

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
import os
import random
from pytorch3d.structures import Pointclouds
from PIL import Image, ImageDraw, ImageFont
import plyfile
import numpy as np
import pandas as pd
from utils import mask_lable_location
import glob
import sys

class Snap:
    def __init__(self, image_size, adjust_camera, save_folder):
        self.image_width, self.image_height = image_size
        self.lift_cam, self.zoomout, self.remove_lip = adjust_camera
        self.save_folder = save_folder
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
    
    def read_plymesh(self, filepath):
        """Read ply file and return it as numpy array. Returns None if emtpy."""
        with open(filepath, "rb") as f:
            plydata = plyfile.PlyData.read(f)
        if plydata.elements:
            vertices = pd.DataFrame(plydata["vertex"].data).values
            faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
            return vertices, faces
    
    def get3d_box_from_pcs(self, pc):
        """
        Given point clouds, return the width, length, and height dimensions of the box that contains the point clouds.
        """
        w = pc[:, 0].max() - pc[:, 0].min()
        l = pc[:, 1].max() - pc[:, 1].min()
        h = pc[:, 2].max() - pc[:, 2].min()

        scene_center = np.array(
        [
            pc[:, 0].max() - w / 2,
            pc[:, 1].max() - l / 2,
            pc[:, 2].max() - h / 2,
        ])

        return w, l, h, scene_center

    def get_rid_of_lip(self, mesh, scan_pc, remove_lip):
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
        map_idx = torch.zeros_like(a[:, 0], dtype=torch.long, device = self.device) - 1
        map_idx[idx] = torch.arange(idx.sum(), device = self.device)
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
        )
        return cropped_mesh, idx

    def render_mesh(self, pose, intrin_path, image_width, image_height, mesh, name, depth_name):
        """
        Given the mesh in PyTorch3D format, render images with a defined pose, intrinsic matrix, and image dimensions
        """

        device = self.device
        background_color = (1.0, 1.0, 1.0)
        intrinsic_matrix = torch.zeros([4, 4])
        intrinsic_matrix[3, 3] = 1

        intrinsic_matrix_load_torch = torch.from_numpy(intrin_path)
        intrinsic_matrix[:3, :3] = intrinsic_matrix_load_torch

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

        rasterizer = MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        )
        fragments = rasterizer(mesh, cameras=camera)
        zbuf =fragments.zbuf[0, ..., 0].cpu().numpy()
        # replace -1 to 0
        zbuf[zbuf == -1] = 0
        np.save(depth_name, zbuf)

        # save depth as image
        zbuf = (zbuf - zbuf.min()) / (zbuf.max() - zbuf.min())
        zbuf = (zbuf * 255).astype(np.uint8)
        zbuf = Image.fromarray(zbuf)
        zbuf.save(depth_name.replace(".npy", ".png"))

    def render_pcd(self, pose, intrinsic, image_width, image_height, pcd, name):
        """
        Given the torch.tensor pcd, render images with a defined pose, intrinsic matrix, and image dimensions
        """

        device = self.device
        intrinsic_matrix = torch.zeros([4, 4])
        intrinsic_matrix[3, 3] = 1
        point_cloud = Pointclouds(points=[pcd[:,:3]], features=[pcd[:,3:6]])
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

    def scene_image_rendering(self, scan_pc_raw, scene_name, mode = ["global"], mask = None):
        
        """
        scene3d: path of .ply file or numpy array of point cloud in the shape of (N, 6) [xyz,rgb]
        scene_name: name of the scene
        mode: list of modes for camera generation, including "global", "wide", "corner"
        """
        self.scene_name = scene_name
        # If scene3d is a path, load the mesh; otherwise, treat it as a point cloud
        if isinstance(scan_pc_raw, str):
            data_type = "mesh"
            # Load the mesh 
            pt3d_io = IO()
            mesh = pt3d_io.load_mesh(scan_pc_raw, device=self.device)
            
            # if mask is available, colorcode the masks in the mesh
            if mask is not None:
                detected_mask_list, _ = mask
                texture_tensor = mesh.textures.verts_features_packed()
                color_list_save = []
                
                for idx_mask in range(detected_mask_list.shape[1]):
                    indices = detected_mask_list[:, idx_mask] != 0
                    random_color = lambda: random.randint(0, 255)
                    color = torch.tensor([random_color(), random_color(), random_color()], device=self.device)
                    color_list_save.append(color)
                    texture_tensor[indices, :] = color.float()/255.
                
                converted_texture = pytorch3d.renderer.mesh.textures.TexturesVertex([texture_tensor])
                mesh = pytorch3d.structures.Meshes(
                    verts=[mesh.verts_packed()],
                    faces=[mesh.faces_packed()],
                    textures=converted_texture
                )
            
            # Remove the lip if needed
            scan_pc_raw = mesh.verts_packed().cpu().numpy()
            mesh, _ = self.get_rid_of_lip(mesh, scan_pc_raw, self.remove_lip)
        else:
            data_type = "pcd"
            # if mask is available, colorcode the masks in the mesh
            if mask is not None:
                detected_mask_list, _ = mask
                color_list_save = []
                
                for idx_mask in range(detected_mask_list.shape[1]):
                    indices = detected_mask_list[:, idx_mask] != 0
                    random_color = lambda: random.randint(0, 255)
                    color = np.array([random_color(), random_color(), random_color()])
                    color_list_save.append(color)
                    scan_pc_raw[indices, 3:6] = torch.tensor(color.astype(np.float32))
            
            # Remove the lip if needed
            z_max = scan_pc_raw[:, 2].max()
            idx_remained = scan_pc_raw[:, 2] <= (z_max - self.remove_lip)
            scan_pc = scan_pc_raw[idx_remained, :]

        # create all folders
        intrinsic_folder = f"{self.save_folder}/{scene_name}/intrinsic"
        pose_folder = f"{self.save_folder}/{scene_name}/pose"
        image_folder = f"{self.save_folder}/{scene_name}/image"
        depth_folder = f"{self.save_folder}/{scene_name}/depth"
        for folder in [intrinsic_folder, pose_folder, image_folder, depth_folder]:
            if data_type == "pcd" and folder == depth_folder:
                continue
            if not os.path.exists(folder): os.makedirs(folder)

        # Get 3D box dimensions and scene center
        w_raw, l_raw, h_raw, scene_center = self.get3d_box_from_pcs(scan_pc_raw)
        scale_factor = self.zoomout + 1
        w, l, h = w_raw * scale_factor, l_raw * scale_factor, h_raw * scale_factor

        # Initialize lists
        extinsic_list = []
        intrinsic_list = []

        # Map modes to their corresponding generation functions
        generation_functions = {
            "global": self.global_level_camera_generation,
            "wide": self.wide_angle_level_camera_generation,
            "corner": self.corner_angle_level_camera_generation
        }

        # Iterate over modes and update lists accordingly
        for mode_key, func in generation_functions.items():
            if mode_key in mode:
                extinsic, intrinsic = func(w, l, h, scene_center, scan_pc_raw)
                extinsic_list.extend(extinsic)
                intrinsic_list.extend(intrinsic)

        # output start to do snap

        print(f"*****************Start to render snap images for {scene_name}*****************")

        for extrinsic, intrinsic, i in zip((extinsic_list), intrinsic_list, range(len(extinsic_list))):

            sys.stdout.write(f"\rSnap module: {i} out of {len(extinsic_list)} images rendered")
            sys.stdout.flush()

            # Save intrinsic and pose matrices
            np.save(f"{intrinsic_folder}/intrinsic_calibrated_angle_{i}.npy", intrinsic)
            np.save(f"{pose_folder}/pose_matrix_calibrated_angle_{i}.npy", extrinsic)
            if data_type == "mesh":
                self.render_mesh(
                    extrinsic,
                    intrinsic[:3, :3],
                    self.image_width,
                    self.image_height,
                    mesh,
                    f"{image_folder}/image_rendered_angle_{i}.png",
                    f"{depth_folder}/image_rendered_angle_{i}.npy",
                )
            elif data_type == "pcd":
                self.render_pcd(
                    extrinsic,
                    intrinsic[:3, :3],
                    self.image_width,
                    self.image_height,
                    scan_pc.to(self.device),
                    f"{image_folder}/image_rendered_angle_{i}.png"
                )

        if mask is not None:
            self.plot_results_label(color_list_save, scan_pc_raw, mask)
        # add the labels if necessary
        return extrinsic, intrinsic

    def plot_results_label(self, color_list_save, scan_pc, mask):
        """

        This function is used to write the name of each mask in the final images. 
        It works fine but is not very fast, as it uses a for loop to write every label for each mask in each image sequentially

        """

        detected_mask, labels = mask

        z_max = scan_pc[:, 2].max()
        idx_remained = scan_pc[:, 2] <= (z_max - self.remove_lip)
        mask_binary = detected_mask[idx_remained, :]
        scan_pc = scan_pc[idx_remained, :]
        scan_pc = torch.tensor(scan_pc, dtype=torch.float32, device=self.device)

        num_image = len(glob.glob(f"{self.save_folder}/{self.scene_name}/image/*.png"))

        # calculate the label location with funcation in Lookup
        for i in range(num_image):
            image_location = f"{self.save_folder}/{self.scene_name}/image/image_rendered_angle_{i}.png"
            camera_to_world = np.load(f"{self.save_folder}/{self.scene_name}/pose/pose_matrix_calibrated_angle_{i}.npy")
            intrinsic_matrix = np.load(f"{self.save_folder}/{self.scene_name}/intrinsic/intrinsic_calibrated_angle_{i}.npy")
            
            label_location = mask_lable_location(
                scan_pc, mask_binary, camera_to_world, intrinsic_matrix, 
                self.image_width, self.image_height, 2, 
                name=None, display_rate_calculation=False
            )

            # Open the existing image
            img = Image.open(image_location)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            for j in range(label_location.shape[0]):
                x, y = label_location[j]

                if x == -1 or y == -1:
                    continue
                # Get text size
                text = labels[j]
                text_bbox = draw.textbbox((x, y), text, font=font)
                text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
                # Draw white rectangle as background for the text
                draw.rectangle([x, y, x + text_size[0], y + text_size[1]], fill='white')
                # Draw the text over the white rectangle
                draw.text((x, y), text, font=font, fill=tuple(color_list_save[j]))
            img.save(image_location)

        return None

    def plot_results_display_rate(self, color_list_save, scan_pc, mask):
        """

        This function is used to write the name of each mask in the final images. 
        It works fine but is not very fast, as it uses a for loop to write every label for each mask in each image sequentially

        """

        detected_mask, _ = mask

        # calculate the label location with funcation in Lookup
        lookup_module = Lookup([self.image_width, self.image_height], self.remove_lip, snap_folder=self.save_folder)
        mask2pxiel_map_list = lookup_module.mask2pixel_map(scan_pc, detected_mask, self.scene_name)
        label_location_visulization = lookup_module.label_location_visulization(mask2pxiel_map_list).cpu().numpy()
        image_num = label_location_visulization.shape[0]
        for i in range(image_num):
            image_location = f"{self.save_folder}/{self.scene_name}/image/image_rendered_angle_{i}.png"
            label_location = label_location_visulization[i]

            # Open the existing image
            img = Image.open(image_location)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            for j in range(label_location.shape[0]):
                x, y = label_location[j]
                if x == -1 or y == -1:
                    continue
                # Get text size
                text = labels[j]
                text_bbox = draw.textbbox((x, y), text, font=font)
                text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
                # Draw white rectangle as background for the text
                draw.rectangle([x, y, x + text_size[0], y + text_size[1]], fill='white')
                # Draw the text over the white rectangle
                draw.text((x, y), text, font=font, fill=tuple(color_list_save[j]))

            img.save(image_location)

        return None

    # all functions for camera parameter generation
    def global_level_camera_generation(self, w, l, h, scene_center, scan_pc):
        extrinsic_list = []
        intrinsic_list = []
        camera_locations = self.generate_camera_locations(scene_center, w, l, h, 5)
        for camera_location in (camera_locations):
            camera_location[-1] += self.lift_cam  # lift the camera
            # Compute pose matrix
            pose_matrix = self.lookat(camera_location, scene_center, np.array([0, 0, -1]))
            pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            # Perform intrinsic calibration
            intrinsic_calibrated = self.intrinsic_calibration(
                scan_pc, pose_to_RBT, self.image_width, self.image_height
            )
            extrinsic_list.append(pose_to_RBT)
            intrinsic_list.append(intrinsic_calibrated)
        return extrinsic_list, intrinsic_list
    
    def wide_angle_level_camera_generation(self, w, l, h, scene_center, scan_pc):

        # Calculate fractions of dimensions
        third_width, third_length, half_height = w / 3, l / 3, h / 2
        half_width, half_length = w / 2, l / 2

        # Define top and bottom z-coordinates
        top_z = scene_center[2] + half_height
        bottom_z = scene_center[2] - half_height

        # Calculate camera positions [4 by 4 line 1,1; 1,2; 2,1; 2,2]
        camera_positions = [
            np.array([scene_center[0] - third_width/2, scene_center[1] - third_length/2, top_z]),
            np.array([scene_center[0] + third_width/2, scene_center[1] - third_length/2, top_z]),
            np.array([scene_center[0] - third_width/2, scene_center[1] + third_length/2, top_z]),
            np.array([scene_center[0] + third_width/2, scene_center[1] + third_length/2, top_z])
        ]

        # Calculate target positions
        target_positions = [
            np.array([scene_center[0] + half_width, scene_center[1] + half_length, bottom_z]),
            np.array([scene_center[0] - half_width, scene_center[1] + half_length, bottom_z]),
            np.array([scene_center[0] + half_width, scene_center[1] - half_length, bottom_z]),
            np.array([scene_center[0] - half_width, scene_center[1] - half_length, bottom_z])
        ]

        # Create scene masks
        scene_masks = [
            (scan_pc[:, 0] > camera_positions[0][0]) & (scan_pc[:, 1] > camera_positions[0][1]),
            (scan_pc[:, 0] < camera_positions[1][0]) & (scan_pc[:, 1] > camera_positions[1][1]),
            (scan_pc[:, 0] > camera_positions[2][0]) & (scan_pc[:, 1] < camera_positions[2][1]),
            (scan_pc[:, 0] < camera_positions[3][0]) & (scan_pc[:, 1] < camera_positions[3][1])
        ]

        # Initialize lists for extrinsic and intrinsic parameters
        extrinsic_list, intrinsic_list = [], []

        for i, camera_location in enumerate(camera_positions):
            if scan_pc[scene_masks[i]].shape[0] == 0:
                print("\n*** WARNING ***\nFailed to render this localized snap image, as the given camera model rendered no points. This image will be skipped. This will not cause any issues, but the number of images will be reduced by one.\n")
                continue
            camera_location[-1] += self.lift_cam  # lift the camera
            # Compute pose matrix
            pose_matrix = self.lookat(camera_location, target_positions[i], np.array([0, 0, -1]))
            pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            # Perform intrinsic calibration
            intrinsic_calibrated = self.intrinsic_calibration(
                scan_pc[scene_masks[i]], pose_to_RBT, self.image_width, self.image_height
            )
            extrinsic_list.append(pose_to_RBT)
            intrinsic_list.append(intrinsic_calibrated)
        
        return extrinsic_list, intrinsic_list

    def corner_angle_level_camera_generation(self, w, l, h, scene_center, scan_pc):

        # Calculate fractions of dimensions
        half_width, half_length, half_height = w / 2, l / 2, h / 2

        # Define top and bottom z-coordinates
        top_z = scene_center[2] + half_height
        bottom_z = scene_center[2] - half_height

        camera_positions = [scene_center] * 4

        # Calculate target positions
        target_positions = [
            np.array([scene_center[0] + half_width, scene_center[1] + half_length, bottom_z]),
            np.array([scene_center[0] - half_width, scene_center[1] + half_length, bottom_z]),
            np.array([scene_center[0] + half_width, scene_center[1] - half_length, bottom_z]),
            np.array([scene_center[0] - half_width, scene_center[1] - half_length, bottom_z])
        ]

        # Create scene masks
        scene_masks = [
            (scan_pc[:, 0] > camera_positions[0][0]) & (scan_pc[:, 1] > camera_positions[0][1]),
            (scan_pc[:, 0] < camera_positions[1][0]) & (scan_pc[:, 1] > camera_positions[1][1]),
            (scan_pc[:, 0] > camera_positions[2][0]) & (scan_pc[:, 1] < camera_positions[2][1]),
            (scan_pc[:, 0] < camera_positions[3][0]) & (scan_pc[:, 1] < camera_positions[3][1])
        ]

        # Initialize lists for extrinsic and intrinsic parameters
        extrinsic_list, intrinsic_list = [], []

        for i, camera_location in enumerate(camera_positions):
            if scan_pc[scene_masks[i]].shape[0] == 0:
                print("\n*** WARNING ***\nFailed to render this localized snap image, as the given camera model rendered no points. This image will be skipped. This will not cause any issues, but the number of images will be reduced by one.\n")
                continue

            camera_location[-1] += self.lift_cam  # lift the camera
            # Compute pose matrix
            pose_matrix = self.lookat(camera_location, target_positions[i], np.array([0, 0, -1]))
            pose_to_RBT = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            # Perform intrinsic calibration
            intrinsic_calibrated = self.intrinsic_calibration(
                scan_pc[scene_masks[i]], pose_to_RBT, self.image_width, self.image_height
            )
            extrinsic_list.append(pose_to_RBT)
            intrinsic_list.append(intrinsic_calibrated)
        
        return extrinsic_list, intrinsic_list

    def intrinsic_calibration(self, point_cloud, pose, width, height):

        """
        Given a predefined pose and point cloud, calibrate the intrinsic matrix to encompass all points in the image
        """

        # Define intrinsic matrix
        depth_intrinsic = np.array([
            [577.590698, 0.000000, 318.905426, 0.000000],
            [0.000000, 578.729797, 242.683609, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ])

        # Extract intrinsic parameters
        fx, fy = depth_intrinsic[0, 0], depth_intrinsic[1, 1]
        cx, cy = depth_intrinsic[0, 2], depth_intrinsic[1, 2]

        # Prepare points in homo coordinates and apply pose transformation
        points = np.hstack([point_cloud[:, :3], np.ones((point_cloud.shape[0], 1))])
        inv_pose = np.linalg.inv(pose.T)
        points_new = points @ inv_pose

        # Project points onto the image plane
        point_projected = np.zeros((points_new.shape[0], 2))
        point_projected[:, 0] = points_new[:, 0] * fx / points_new[:, 2] + cx
        point_projected[:, 1] = points_new[:, 1] * fy / points_new[:, 2] + cy

        # Compute new intrinsic parameters
        cx_new = cx - point_projected[:, 0].min()
        cy_new = cy - point_projected[:, 1].min()

        # Re-project points with updated intrinsic parameters
        point_projected[:, 0] = points_new[:, 0] * fx / points_new[:, 2] + cx_new
        point_projected[:, 1] = points_new[:, 1] * fy / points_new[:, 2] + cy_new

        # Determine scaling factors
        scale_1 = width / point_projected[:, 0].max()
        scale_2 = height / point_projected[:, 1].max()
        scale = min(scale_1, scale_2)

        # Update intrinsic parameters with scaling
        fx_new = depth_intrinsic[0, 0] * scale
        fy_new = depth_intrinsic[1, 1] * scale
        cx_new *= scale
        cy_new *= scale

        # Update the intrinsic matrix
        new_intrinsic = depth_intrinsic.copy()
        new_intrinsic[0, 0] = fx_new
        new_intrinsic[1, 1] = fy_new
        new_intrinsic[0, 2] = cx_new
        new_intrinsic[1, 2] = cy_new


        return new_intrinsic

    def generate_camera_locations(self, center, width, length, height, num_split=5):
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

    def lookat(self, center, target, up):
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

    # generated

def main():

    """
    here we show how to use the SNAP module to render scene-level images
    """
    
    # Define image dimensions
    image_width = 2000
    image_height = 2000

    # Camera adjustment parameters
    lift_cam = 3  # Vertical lift of the camera (increase to lift the camera higher)
    zoomout = 0.1  # Zoom out factor to view the scene with a wider angle
    remove_lip = 0.5  # Distance to remove the ceiling or upper part of the scene for better visibility, if needed

    adjust_camera = [lift_cam, zoomout, remove_lip]
    image_size = [image_width, image_height]

    # Initialize the SNAP module
    snap_module = Snap(image_size, adjust_camera, save_folder="example_snap")


    # Path to the mesh file
    mesh_path = "/home/zelda/zh340/myzone/OpenIns3D_final_github/data/replica/office0/office0_mesh.ply"

    # Load pcd from mesh
    pcd_rgb, _ = snap_module.read_plymesh(mesh_path)
    pcd_rgb = np.hstack((pcd_rgb[:, :3], pcd_rgb[:, 6:9]))


    #### start using SNAP

    # Render the scene with the point cloud data
    snap_module.scene_image_rendering(pcd_rgb[:, :6], "scannet_scene_pcd", mode=["global", "wide", "corner"])

    # # Render the scene directly from the mesh file
    snap_module.scene_image_rendering(mesh_path, "scannet_scene_mesh", mode=["global", "wide", "corner"])


    #### start rendering with masks

    # Load the mask data and generate labels
    mask_path = "/home/zelda/zh340/myzone/OpenIns3D_final_github/output/replica/replica_ground_truth_masks/office0.pt"
    mask_list = torch.load(mask_path)[0]
    mask_label = [f"label_{i}" for i in range(mask_list.shape[1])]
    detection_results = [mask_list, mask_label]

    # Render the scene with the mask and labels applied
    snap_module.scene_image_rendering(pcd_rgb[:, :6], "scannet_scene_pcd_with_results", mode=["global", "wide", "corner"], mask=detection_results)

    # Render the scene with the mask and labels applied directly from the mesh
    # snap_module.scene_image_rendering(mesh_path, "scannet_scene_mesh_with_results", mode=["global", "wide", "corner"], mask=detection_results)
    
    # mask2pxiel_map_list = lookup_module.mask2pixel_map(pcd_rgb, mask_list, "scannet_scene_mesh_with_results")
    # # test 2: use the mask2pixel map to plot the label location
    # label_location_visulization = lookup_module.bbox_from_mask2pixel(mask2pxiel_map_list).cpu().numpy()
    # # save this as np
    # np.save(f"label_location_visulization.npy", label_location_visulization)

    # Render the scene with mask and labels

if __name__ == "__main__":
    main()