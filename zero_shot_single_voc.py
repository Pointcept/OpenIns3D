"""
This is the demo script to show how to use the pipeline to detect open vocabulary single object in the scene
"""

import os
import sys
sys.path.append("openins3d/")
import torch
import numpy as np
import open3d as o3d
from openins3d.lookup import Lookup
from openins3d.snap import Snap
import cv2
import pyviz3d.visualizer as viz
# from openins3d.mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud


def single_vocabulary_detection(path_3D_scans, vocabulary, path_masks= None, path_images = None):
    name_of_scene = path_3D_scans.split("/")[-1].split(".")[0]
    print("scene name: ", name_of_scene)    
    if path_3D_scans.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(path_3D_scans) 
        xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors)* 255.
        xyz_rgb = torch.from_numpy(np.concatenate([xyz, rgb], axis=1)).float()
    elif path_3D_scans.endswith(".npy"):
        xyz_rgb = np.load(path_3D_scans)[:, :6]
    else:
        raise ValueError("Unsupported file format")
    
    if path_masks:
        mask = torch.load(path_masks).to_dense()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model("third_party/scannet200_val.ckpt").to(device).eval()
        data, features, _, inverse_map = prepare_data(pcd_rgb, device)
        with torch.no_grad():
            mask_list = map_output_to_pointcloud(model(data, raw_coordinates=features), inverse_map, 0.5)

    snap_class = Snap([800, 800], [2, 0.1, 1.0], "output/snap_demo")
    lookup_class = Lookup([800, 800], 0.5, "output/snap_demo", text_input=[vocabulary], results_folder="output/results_demo")

    if path_images:
        # if image exists, load the image and set the image size
        
        image = cv2.imread(os.path.join(path_images, name_of_scene, "image/image_rendered_angle_0.png"))
        image_width, image_height = image.shape[1], image.shape[0]
        lookup_class.image_width = image_width
        lookup_class.image_height = image_height
        lookup_class.remove_lip = 0
        lookup_class.snap_folder = path_images
        lookup_class.depth_shift = 1000
    else:
        # else, render the image
        snap_class.scene_image_rendering(xyz_rgb, name_of_scene, mode=["global", "wide", "corner"])

    lookup_class.call_YOLOWORLD()

    if path_images:
        mask_classfication, score = lookup_class.lookup_pipelie(xyz_rgb, mask, name_of_scene, threshold = 0.7, use_2d=True, single_detection=True)
    else:
        mask_classfication, score = lookup_class.lookup_pipelie(xyz_rgb, mask, name_of_scene, threshold = 0.7, use_2d=False, single_detection=True)

    mask_final = mask[:, [i for i in range(len(mask_classfication)) if mask_classfication[i] != -1]]
    # save the results as image
    snap_class.scene_image_rendering(torch.tensor(xyz_rgb).float(), f"{name_of_scene}_vis", mode=["global"], mask=[mask_final, None])
    print("Detection compelted. There are {} objects detected.".format(mask_final.shape[1]))
    return xyz, rgb, mask_final, mask, v

def plot_mask(original_mask, final_mask, scene_coord, scene_color, name_of_scene, v):


    for idx_mask in range(original_mask.shape[1]):
        mask_individual = original_mask[:, idx_mask].bool()
        mask_point = scene_coord[mask_individual]
        mask_color = scene_color.copy()
        mask_final_color = scene_color.copy()
        for i in range(original_mask.shape[1]):
            mask_i = original_mask[:, i]
            mask_i = mask_i.bool()
            random_colr = np.random.rand(3)* 255.
            mask_color[mask_i, :] = random_colr

        for i in range(final_mask.shape[1]):
            mask_i = final_mask[:, i]
            mask_i = mask_i.bool()
            mask_final_color[mask_i, :] = [255, 0, 0]
    v.add_points(f"{name_of_scene}_rgb", scene_coord, scene_color, point_size=20, visible=False)
    v.add_points(f"{name_of_scene}_allmask", scene_coord, mask_color, point_size=20, visible=False)
    v.add_points(f"{name_of_scene}_detected", scene_coord, mask_final_color, point_size=20, visible=True)

if __name__ == "__main__":
    v = viz.Visualizer()

    name_of_scene = "demo_1"
    path_3D_scans = f"data/demo_single_voc/scenes/{name_of_scene}.ply"
    path_masks = f"data/demo_single_voc/masks/{name_of_scene}.pt"
    path_images = "data/demo_single_voc/rgbd"
    vocabulary = "device to watch BBC news"
    xyz, rgb, mask_final, mask, v = single_vocabulary_detection(path_3D_scans, vocabulary, path_masks, path_images)
    plot_mask(mask, mask_final, xyz, rgb, name_of_scene, v)

    name_of_scene = "demo_2"
    path_3D_scans = f"data/demo_single_voc/scenes/{name_of_scene}.ply"
    path_masks = f"data/demo_single_voc/masks/{name_of_scene}.pt"
    path_images = "data/demo_single_voc/rgbd"
    vocabulary = "laptop"
    xyz, rgb, mask_final, mask, v = single_vocabulary_detection(path_3D_scans, vocabulary, path_masks, path_images)
    plot_mask(mask, mask_final, xyz, rgb, name_of_scene, v)

    name_of_scene = "demo_3"
    path_3D_scans = f"data/demo_single_voc/scenes/{name_of_scene}.ply"
    path_masks = f"data/demo_single_voc/masks/{name_of_scene}.pt"
    path_images = "data/demo_single_voc/rgbd"
    vocabulary = "chair with the backbag"
    xyz, rgb, mask_final, mask, v = single_vocabulary_detection(path_3D_scans, vocabulary, path_masks, path_images)
    plot_mask(mask, mask_final, xyz, rgb, name_of_scene, v)
    v.save(f'output/demo/viz')