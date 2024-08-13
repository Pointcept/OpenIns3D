"""
Lookup Script for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)
"""

import torch
import numpy as np
from tqdm import tqdm
import os
from utils import read_plymesh
from glob import glob
import torch.nn.functional as F
from utils import *
from snap import Snap
import cv2


class Lookup:

    def __init__(self, image_size, remove_lip, snap_folder, text_input, results_folder = None):
        self.image_height, self.image_width = image_size
        self.remove_lip = remove_lip
        self.snap_folder = snap_folder
        self.results_folder = results_folder
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.text_input = text_input
        self.depth_shift = 6553.5

    def call_ODISE(self):
        from build_lookup_odise import ODISE
        if hasattr(self, 'YOLOWORLD'):
            delattr(self, 'YOLOWORLD')
        self.ODISE = ODISE(self.snap_folder, self.results_folder, self.text_input)

    def call_YOLOWORLD(self):
        from build_lookup_yoloworld import YOLOWORLD
        if hasattr(self, 'ODISE'):
            delattr(self, 'ODISE')
        self.YOLOWORLD = YOLOWORLD(self.snap_folder, self.results_folder, self.text_input)

    def mask2pixel_map(self, scan_pc, mask_binary, scene_id, save_image=False, bbox=False, use_depth = False):    
        
        """
        This is the code to calculate the pixel locations for all 3d masks.
        input: scan_pc: torch.tensor, [N,6]
                mask_binary: torch.tensor, [N, num_masks]
                scene_id: str, the scene id
                save_image: bool, whether to save the image of mask2pixel map
                bbox: bool, whether to return the bbox or mask2pixel map
        output:
            if bbox is False: [num_imgs, width, height], where -1 is the background, and other values are the mask index
            if bbox is True: [num_imgs, num_mask, 4], where the 4 is [x_min, y_min, x_max, y_max]
        """

        # prepare the pcd by removing the lip as what we did for the snap module.
        z_max = scan_pc[:, 2].max()
        idx_remained = scan_pc[:, 2] <= (z_max - self.remove_lip)
        mask_binary = mask_binary[idx_remained, :]
        scan_pc = scan_pc[idx_remained, :]
        scan_pc = torch.tensor(scan_pc, dtype=torch.float32, device=self.device)
        num_imgs = len(glob(f"{self.snap_folder}/{scene_id}/pose/*.npy"))
        self.num_imgs = num_imgs

        if save_image:
            if not os.path.exists(f"{self.results_folder}/{scene_id}"):os.makedirs(f"{self.results_folder}/{scene_id}/")
        # start to calculate the mask2pixel map
        mask2pxiel_map_list = torch.zeros((self.image_height, self.image_width, num_imgs), device=self.device)
        self.num_mask = mask_binary.shape[1]

        for angle in range(num_imgs):
            # load the camera pose and intrinsic matrix
            camera_to_world = np.load(f"{self.snap_folder}/{scene_id}/pose/pose_matrix_calibrated_angle_{angle}.npy")
            intrinsic_matrix = np.load(f"{self.snap_folder}/{scene_id}/intrinsic/intrinsic_calibrated_angle_{angle}.npy")

            if save_image:
                save_mask2pixel_map = f"{self.results_folder}/{scene_id}/mask2pixel_map_{angle}.png"  # Optional: Path for saving the image
            else:
                save_mask2pixel_map = None

            # calculate the mask2pixel map
            if not use_depth:
                mask2pxiel_map, _ = mask_rasterization(
                    scan_pc, mask_binary, camera_to_world, intrinsic_matrix, 
                    self.image_width, self.image_height, 2, 
                    name=save_mask2pixel_map, display_rate_calculation=False
            )
                mask2pxiel_map_list[:,:,angle] = mask2pxiel_map
            else:
                depth_map_file = f"{self.snap_folder}/{scene_id}/depth/depth_rendered_angle_{angle}.png"
                depth = cv2.imread(depth_map_file, -1)
                filter_index = filter_pcd_with_depthmap(scan_pc[:, :3], torch.from_numpy(intrinsic_matrix).cuda(), torch.from_numpy(depth.astype(np.int32)).cuda(), torch.from_numpy(camera_to_world).cuda(), depth_shift= self.depth_shift, device="cuda")
                filter_index = filter_index.cpu().numpy()
                mask2pxiel_map, _ = mask_rasterization(
                    scan_pc[filter_index, :], mask_binary[filter_index, :], camera_to_world, intrinsic_matrix, 
                    self.image_width, self.image_height, 1, 
                    name=save_mask2pixel_map, display_rate_calculation=False
                )
                mask2pxiel_map_list[:,:,angle] = mask2pxiel_map

        if bbox:
            return self.bbox_from_mask2pixel(mask2pxiel_map_list)
        else:
            return mask2pxiel_map_list

    def display_report(self, scan_pc, mask_binary, camera_to_world, intrinsic_matrix):
        """
        This is the code to calculate the display rate for each mask.
        input: scan_pc: torch.tensor, [N,6]
                mask_binary: torch.tensor, [N, num_masks]
                camera_to_world: np.array, [4,4]
                intrinsic_matrix: np.array, [4,4]
        output: display_rate: np.array, [num_masks, 1]
        """
        return mask_rasterization(
                scan_pc, mask_binary, camera_to_world, intrinsic_matrix, 
                self.image_width, self.image_height, point_size=1, # Important: point_size has to be 1 for occlusion report.
                name=None, display_rate_calculation=True
            )[1]

    def label_location_visulization(self, mask2pxiel_map_list):
        """
        This funcation is used to compute the label labels in 2D image for visualization.
        input: mask2pxiel_map_list: torch.tensor, [width, height, num_imgs]
        output: [num_imgs, num_mask, 2]
        """
        final_box = self.bbox_from_mask2pixel(mask2pxiel_map_list)
        center_bbox = (final_box[:, :, [0, 1]] + final_box[:, :, [2, 3]]) / 2
        return center_bbox

    def bbox_from_mask2pixel(self, mask2pxiel_map_list):

        """
        This function is used to convert pixel location to bbox.
        input: mask2pxiel_map_list: torch.tensor, [width, height, num_imgs]
        output: [num_mask, num_imgs, 4], where 4 is [x_min, y_min, x_max, y_max]
        """
        max_mask_idx = self.num_mask -1

        # set the background to be max_mask_idx + 1 
        mask2pxiel_map_list[mask2pxiel_map_list == -1] = max_mask_idx + 1
        num_classes = int(max_mask_idx) + 2
        number_image = mask2pxiel_map_list.shape[2]
        final_bbox = torch.zeros((int(number_image), int(max_mask_idx)+1, 4), device=self.device)
        for i in range(number_image):
            mask2pxiel_map_single = mask2pxiel_map_list[:, :, i].unsqueeze(-1)
            one_hot_encoded = F.one_hot(mask2pxiel_map_single.long(), num_classes=num_classes)
            binary_mask = one_hot_encoded.permute(0, 1, 3, 2).float()
            binary_mask = binary_mask[:, :, :-1, :].squeeze(-1)
            final_bbox[i] = to_bounding_boxes(binary_mask.bool())
        return final_bbox


    
    def asslign_label_with_pixel(self, project_mask_list, pred_mask_list, pred_label_list):
        """
        This function is used to asslign the label with pixel location.
        input: mask2pxiel_map_list: torch.tensor, [num_imgs, width, height]
               pred_label: torch.tensor, ([width, height, num_imgs], [labels_dict])
        output: predict: [num_mask]
        """

        num_mask = self.num_mask # total number of mask in 3d scene
        num_image = self.num_imgs # total number of images

        # create a dict to store mask from multiple images
        perdiction_collections = {} # this save all pred masks, of which iou with the project mask is greater than threshold.
        score_colletions = {} # this save the iou score of the pred masks that has been assigned to the project mask

        # set up the dict for each mask in 3d
        for mask_idx in range(num_mask):
            perdiction_collections[int(mask_idx)] = []
            score_colletions[int(mask_idx)] = []

        for i in tqdm(range(num_image)):
            project_mask = project_mask_list[:, :, i]  # where -1 is the background
            pred_mask = pred_mask_list[i, :, :] # where 0 is the background
            labels_for_pred = pred_label_list[i]
            pred_mask[project_mask==-1] = 0 # only consider the pixel that has a projection

            assigned_masks, score_masks = assign_pred_mask_to_gt(pred_mask, project_mask, 0.5)
            for mask_3d_idx, pred_2d in assigned_masks.items():
                perdiction_collections[int(mask_3d_idx)].append(int(labels_for_pred[pred_2d])) 
                score_colletions[int(mask_3d_idx)].append(score_masks[mask_3d_idx])
        return perdiction_collections, score_colletions

    def multiview_aggregation(self, perdiction_collections, score_colletions, threshold = 0.5, single_detection = False):
        final_mask_classfication = {} # this is the final mask classification for each mask in 3d after multi view aggregation
        mask2pixel_lookup_score = {} # this give the score the the prediction

        for mask_id, prediction_list in perdiction_collections.items():
            if len(prediction_list) == 0 or len(prediction_list) == 1:
                final_mask_classfication[mask_id] = None
                mask2pixel_lookup_score[mask_id] = None
                continue

            predict_score = score_colletions[mask_id]
            top_1_predict, top_1_score = normalize_scores(predict_score, prediction_list)

            if top_1_score < threshold:
                final_mask_classfication[mask_id] = None
                mask2pixel_lookup_score[mask_id] = None
                continue

            mask2pixel_lookup_score[mask_id] = top_1_score
            final_mask_classfication[mask_id] = top_1_predict 

        mask_results = [i if i is not None else -1 for i in final_mask_classfication.values()]
        mask_score = [i if i is not None else -1 for i in mask2pixel_lookup_score.values()]

        return mask_results, mask_score

    def assign_label_with_bbox(self, project_bbox, pred_bbox, labels, threshold):
        """
        This function is used to asslign the label with pixel location.
        
        input: mask2pxiel_map_list: torch.tensor, [num_imgs, num_masks, 4]
                pred_label: torch.tensor, ([num_imgs, num_masks, 4], [labels_dict])
        output: mask_results: list, [num_masks], the final mask results
                mask_score
        
        """
        # batch calculation of IOU between predicted and project bounding boxes

        project_bbox = project_bbox.unsqueeze(2)  
        pred_bbox = pred_bbox.unsqueeze(1)  
        x1 = torch.maximum(project_bbox[..., 0], pred_bbox[..., 0])
        y1 = torch.maximum(project_bbox[..., 1], pred_bbox[..., 1])
        x2 = torch.minimum(project_bbox[..., 2], pred_bbox[..., 2])
        y2 = torch.minimum(project_bbox[..., 3], pred_bbox[..., 3])
        intersection_area = torch.maximum(torch.tensor(0.0), x2 - x1) * torch.maximum(torch.tensor(0.0), y2 - y1)

        projection_area = (project_bbox[..., 2] - project_bbox[..., 0]) * (project_bbox[..., 3] - project_bbox[..., 1])
        pred_bbox_area = (pred_bbox[..., 2] - pred_bbox[..., 0]) * (pred_bbox[..., 3] - pred_bbox[..., 1])
        union_area = projection_area + pred_bbox_area - intersection_area
        iou_array = intersection_area / (union_area + 1e-6)

        # Find the index of the bbox with the highest IoU for each projection
        best_match_idx = torch.argmax(iou_array, dim=2)  
        best_iou_values = torch.max(iou_array, dim=2).values

        # Get the number of images and masks
        num_image = best_match_idx.shape[0]
        num_mask = best_match_idx.shape[1]
    
        perdiction_collections = {} # this save all pred masks, of which iou with the project mask is greater than threshold.
        score_colletions = {} # this save the iou score of the pred masks that has been assigned to the project mask

        # set up the dict for each mask in 3d
        for mask_idx in range(num_mask):
            perdiction_collections[int(mask_idx)] = []
            score_colletions[int(mask_idx)] = []

        for i in range(num_image):
            assigned_masks, score_masks = best_match_idx[i], best_iou_values[i]
            this_label = labels[i]
            for mask_i, pred_2d in enumerate(assigned_masks):
                if score_masks[mask_i] >= threshold: # here the where the threshold is set
                    perdiction_collections[mask_i].append(int(this_label[pred_2d]))
                    score_colletions[mask_i].append(float(score_masks[mask_i]))

        return perdiction_collections, score_colletions

    def lookup_pipelie(self, scan_pc, mask_binary, scene_id, threshold = 0.6, use_2d = False, single_detection = False):    

        if hasattr(self, 'YOLOWORLD'):
            bbox = True
            mask2pxiel_map_list = self.mask2pixel_map(scan_pc, mask_binary, scene_id, save_image=True, bbox=bbox, use_depth = use_2d)
            mask, label = self.YOLOWORLD.build_lookup_dict(scene_id, save = True,  single_detection = single_detection)
            perdiction_collections, score_colletions = self.assign_label_with_bbox(mask2pxiel_map_list, mask, label, threshold) 
            mask, score = self.multiview_aggregation(perdiction_collections, score_colletions, threshold = 0.5, single_detection = single_detection)
        elif hasattr(self, 'ODISE'):
            bbox = False
            mask2pxiel_map_list = self.mask2pixel_map(scan_pc, mask_binary, scene_id, save_image=True, bbox=bbox, use_depth = use_2d)
            mask, label = self.ODISE.build_lookup_dict(scene_id, save = True)
            perdiction_collections, score_colletions = self.asslign_label_with_pixel(mask2pxiel_map_list, mask, label)
            mask, score = self.multiview_aggregation(perdiction_collections, score_colletions, threshold = 0.5)
        else:
            raise ValueError("Please call either ODISE or YOLOWORLD first")
        return mask, score


if __name__ == "__main__":

    # Define image dimensions
    # image_width = 800
    # image_height = 800

    image_width = 360
    image_height = 640

    remove_lip = 0  # Distance to remove the ceiling or upper part of the scene for better visibility, if needed
    image_size = [image_width, image_height]


    # Path to the mesh file
    mesh_path = "data/replica/scenes/office3.ply"

    # Load pcd from mesh
    pcd_rgb, _ = read_plymesh(mesh_path)
    pcd_rgb = np.hstack((pcd_rgb[:, :3], pcd_rgb[:, 6:9]))

    # Load the mask data and generate labels
    mask_path = "data/replica/masks/ground_truth/office3.pt"


    text_prompts_replica, VALID_CLASS_IDS = get_label_and_ids("replica")
    snap_folder = "replica_oi3_depth"
    save_folder = "output/lookup_folder"

    scene_id = mesh_path.split("/")[-1].split(".")[0]
    lookup_module = Lookup(image_size, remove_lip, snap_folder, text_input=text_prompts_replica, results_folder = save_folder)

    # test 1: compute the mask2pixel map
    lookup_module.call_YOLOWORLD()
    pcd_rgb = torch.tensor(pcd_rgb).cuda()
    mask_list = torch.load(mask_path).to_dense().cuda()
    mask2pxiel_map_list = lookup_module.mask2pixel_map(pcd_rgb, mask_list, scene_id, save_image=False, bbox=True)
    mask, label = lookup_module.YOLOWORLD.build_lookup_dict(scene_id, save = True)
    perdiction_collections, score_colletions = lookup_module.assign_label_with_bbox(mask2pxiel_map_list, mask, label, 0.5) 
    masks, score = lookup_module.multiview_aggregation(perdiction_collections, score_colletions, threshold = 0.5)

    all_valid_idx = [i for i in range(len(masks)) if masks[i] != -1]
    label_results = [text_prompts_replica[i] for i in masks if i != -1]
    mask_final = mask_list[:, all_valid_idx]   
    detection_results_yolo = [mask_final, label_results]

    # lookup_module.call_ODISE()
    # masks, score = lookup_module.lookup_pipelie(pcd_rgb, mask_list, "office2", threshold = 0.3)
    # all_valid_idx = [i for i in range(len(masks)) if masks[i] != -1]
    # label_results = [text_prompts_replica[i] for i in masks if i != -1]
    # mask_final = mask_list[:, all_valid_idx]
    # detection_results_odise = [mask_final, label_results]

    # # plot the results
    adjust_camera = [5, 0.1, 0.5]
    snap_module = Snap(image_size, adjust_camera, save_folder=save_folder)
    snap_module.scene_image_rendering(mesh_path, "odise_results_vis_yolo", mode=["global"], mask=detection_results_yolo)
    # snap_module.scene_image_rendering(mesh_path, "odise_results_vis_odise", mode=["global"], mask=detection_results_odise)


