"""
Lookup Script for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)
"""

import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import os
import time
from torch_scatter import scatter_max


def generate_neighbors(thickness):
    
    size = (thickness - 1) * 2 + 1
    x_coords = torch.arange(0 - thickness + 1, 0 + thickness).unsqueeze(1).repeat(1, size)
    y_coords = torch.arange(0 - thickness + 1, 0 + thickness).repeat(size, 1)
    x_coords = x_coords.view(-1)
    y_coords = y_coords.view(-1)

    neighbor_indices = torch.column_stack((x_coords, y_coords)).cuda()

    return neighbor_indices

def single_mask_projection(obj_pcd_rgb, depth_intrinsic, pose_matrix, mask_idx_input, width, height, point_size):
    point_cloud_mask = obj_pcd_rgb
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    bx = depth_intrinsic[0, 3]
    by = depth_intrinsic[1, 3]

    points = torch.ones((point_cloud_mask.shape[0], 4), device = 'cuda')

    points[:, :3] = point_cloud_mask[:, :3]

    inv_pose = np.linalg.inv(np.transpose(pose_matrix))
    inv_pose_torch = torch.from_numpy(inv_pose).float().cuda()
    points_new = torch.matmul(points, inv_pose_torch)

    X = points_new[:, 0]
    Y = points_new[:, 1]
    Z = points_new[:, 2]
    
    non_zero_mask = Z != 0

    image_x = (X[non_zero_mask] - bx) * fx / Z[non_zero_mask] + cx
    image_y = (Y[non_zero_mask] - by) * fy / Z[non_zero_mask] + cy
    
    x_idx = torch.round(image_x).int()
    y_idx = torch.round(image_y).int()

    index_xy = torch.vstack((x_idx,y_idx)).T
    depth = Z[non_zero_mask]
    
    point_size = 2
    index_xy_expand = index_xy[:, None, :] + generate_neighbors(point_size)

    neighborhood_size = ((point_size - 1) * 2 + 1)**2
    
    index_xy_expand[index_xy_expand >= width] = width-1
    index_xy_expand[index_xy_expand < 0] = 0
    index_xy = index_xy_expand.reshape(-1, 2)
    
    depth = depth.unsqueeze(1).expand(depth.shape[0], neighborhood_size).reshape(-1)


    unique_idx, indices = torch.unique(index_xy, dim=0, return_inverse=True)
    
    out, _ = scatter_max(depth, indices)

    mask_map = torch.zeros((width, height), device = 'cuda')
    depth_map = torch.ones((width, height), device = 'cuda') * 999

    index = unique_idx
    mask_map[index[:, 0], index[:, 1]] = mask_idx_input
    depth_map[index[:, 0], index[:, 1]] = out.cuda()

    return mask_map, depth_map

def pcd2img_point_occlusion_aware(all_point, mask_bin, pose_matrix, depth_intrinsic, width, height, point_size = 1, name = None):
    
    "mask rasterization for mask2pixel mapping geneartion"

    mask_map_total = []
    depth_map_total = []

    point_size = 1
    
    for mask_idx in range(mask_bin.shape[1]):
        indices = torch.nonzero(mask_bin[:, mask_idx])
        if indices.shape[0]<=2:
            continue
        obj_pcd_rgb = all_point[indices.squeeze()]
        mask_map, depth_map = single_mask_projection(obj_pcd_rgb, depth_intrinsic, pose_matrix,mask_idx, width, height, point_size)
        mask_map_total.append(mask_map.unsqueeze(2))
        depth_map_total.append(depth_map.unsqueeze(2))

    mask_map_total_stack = torch.concatenate(mask_map_total, axis =2 )
    depth_map_total_stack = torch.concatenate(depth_map_total, axis =2)

    depth_map_max_idx = torch.argmin(depth_map_total_stack, axis=-1)

    condensed_depth_map = mask_map_total_stack[torch.arange(mask_map_total_stack.shape[0])[:, None, None], torch.arange(mask_map_total_stack.shape[1])[None, :, None], depth_map_max_idx[:, :, None]]
    final_depth_map = condensed_depth_map.int().transpose(1,0).squeeze()

    if name:
        plot_mask = final_depth_map.cpu().numpy()
        num_unique_values = len(np.unique(plot_mask))
        cmap = plt.cm.get_cmap('tab10', num_unique_values)
        plt.imshow(plot_mask, cmap=cmap, interpolation='none')
        plt.savefig(f'{name}.png')

    return final_depth_map

def calculate_iou_matrix(mask_tensor1, mask_tensor2):
    
    mask_tensor1 = mask_tensor1.cuda().reshape(-1,1)
    mask_tensor2 = mask_tensor2.cuda().reshape(-1,1)
    unique_groups_1 = torch.unique(mask_tensor1)
    unique_groups_1 = unique_groups_1[unique_groups_1!=0]
    original_mask_1 = (mask_tensor1 == unique_groups_1.view(1, -1)).float()
    original_mask_1_t = original_mask_1.t()
    unique_groups_2 = torch.unique(mask_tensor2)
    # unique_groups_2 = unique_groups_2[unique_groups_2!=-1]
    original_mask_2 = (mask_tensor2 == unique_groups_2.view(1, -1)).float()
    
    intersection = torch.mm(original_mask_1_t, original_mask_2)
    union = original_mask_1_t.sum(dim=1, keepdim=True) + original_mask_2.sum(dim=0, keepdim=True) - intersection
    iou_matrix = intersection / union
    return iou_matrix.cpu().numpy(), unique_groups_1, unique_groups_2

def assign_pred_mask_to_gt(pred_mask, project_mask, threshold):
    assigned_masks = {}
    score_masks = {}
    iou_matrix, idx_pred, idx_project = calculate_iou_matrix(pred_mask, project_mask)
    iou_matrix[iou_matrix<threshold] = 0
    if iou_matrix.shape[0]==0:
        return assigned_masks, score_masks
    max_indices = np.argmax(iou_matrix, axis=0)
    max_iou = np.max(iou_matrix, axis=0)
    for count, best_match_iou in enumerate(max_iou):
        targeting_proj_mask = int(idx_project[count])
        if best_match_iou !=0:
            assigned_masks[targeting_proj_mask] =  int(idx_pred[max_indices[count]])  
            score_masks[targeting_proj_mask] = best_match_iou
    return assigned_masks, score_masks

def normalize_scores(predict_score, predict_list):
    total_score = sum(predict_score)
    normalized_scores = [score / total_score for score in predict_score]
    combined_normalized_data = list(zip(normalized_scores, predict_list))
    grouped_data = {}
    for normalized_score, predict in combined_normalized_data:
        if predict not in grouped_data:
            grouped_data[predict] = 0
        grouped_data[predict] += normalized_score
    predict_ranks = sorted(grouped_data, key=grouped_data.get, reverse=True)
    sorted_data = [grouped_data[rank]  for rank in predict_ranks]
    return predict_ranks, sorted_data

def mask_classfication(mask_bin, all_point, adjust_camera, scene_id, width, height, SNAP_location, Lookup_location, final_results,  CLASS_LABELS, VALID_CLASS_IDS ):
    
    lift_cam, zoomout, remove_lip = adjust_camera
    
    # get rid of lip
    z_max = all_point[:, 2].max()
    idx_remained = all_point[:, 2] <= (z_max - remove_lip)

    mask_bin_original = mask_bin.clone()

    mask_bin = mask_bin[idx_remained, :]
    all_point = all_point[idx_remained, :]
    # number of mask
    num_mask = mask_bin.shape[1]

    # create a dict to store mask from multiple images
    perdiction_collections = {}
    score_colletions = {}
    for mask_idx in range(0, num_mask):
        perdiction_collections[int(mask_idx)] = []
        score_colletions[int(mask_idx)] = []

    for angle in tqdm(range(16)):
        camera_to_world = f"{SNAP_location}/{scene_id}/pose/pose_matrix_calibrated_angle_{angle}.npy"
        intrinsic_matrix = f"{SNAP_location}/{scene_id}/intrinsic/intrinsic_calibrated_angle_{angle}.npy"
        
        intrinsic_matrix = np.load(intrinsic_matrix)
        pose_matrix = np.load(camera_to_world)    
    
        mask2pxiel_map = pcd2img_point_occlusion_aware(all_point, mask_bin, pose_matrix, intrinsic_matrix, width, height, 2) # vectorized this operation
        
        load_2d_pred_map = f"{Lookup_location}/{scene_id}/map/image_rendered_angle_{angle}.pt"
        load_2d_pred_label = f"{Lookup_location}/{scene_id}/label/image_rendered_angle_{angle}"
        pred_map = torch.load(load_2d_pred_map).to_dense().cpu()
        with open(load_2d_pred_label, "r") as read_file:
            label_2d = json.load(read_file)

        mask_align_between_2d_3d, score_2d_3d = assign_pred_mask_to_gt(pred_map, mask2pxiel_map, 0.30)

        for mask_3d_idx, pred_2d in mask_align_between_2d_3d.items():
            perdiction_collections[int(mask_3d_idx)].append(label_2d[pred_2d-1]["category_id"])
            score_colletions[int(mask_3d_idx)].append(score_2d_3d[mask_3d_idx])

    mask2pixel_lookup = {}
    mask2pixel_lookup_score = {}
    for mask_id, prediction_list in perdiction_collections.items():
        if len(prediction_list) == 0:
            mask2pixel_lookup[mask_id] = None
            mask2pixel_lookup_score[mask_id] = 0
            continue
        predict_score = score_colletions[mask_id]
        prediction_ranks, score_ranks = normalize_scores(predict_score, prediction_list)
        prediction = prediction_ranks[0]
        normalized_iou = score_ranks[0]
        if normalized_iou < 0.5:
            mask2pixel_lookup[mask_id] = None
            mask2pixel_lookup_score[mask_id] = 0
            continue
        mask2pixel_lookup_score[mask_id] = normalized_iou
        mask2pixel_lookup[mask_id] = prediction 

    save_path = f"{final_results}/{scene_id}"
    output_dir= f"{save_path}/pred_mask"

    final_mask_idx = []
    for mask_idx in mask2pixel_lookup.keys():
        if mask2pixel_lookup[mask_idx] != None:
            final_mask_idx.append(int(mask_idx))
            mask2pixel_lookup[mask_idx] = VALID_CLASS_IDS[(mask2pixel_lookup[int(mask_idx)])]

    if not os.path.exists(output_dir):os.makedirs(output_dir)
    labels_list = []
    confidences_list = []
    masks_binary_list = []
    
    for mask_idx in range(0, num_mask):
        labels_list.append(mask2pixel_lookup[mask_idx])
        confidences_list.append(mask2pixel_lookup_score[mask_idx])
        masks_binary_list.append(mask_bin_original[:, mask_idx])

    with open(os.path.join(save_path, f'{scene_id}.txt'), 'w') as f:
        for i, (l, c, m) in enumerate(zip(labels_list, confidences_list, masks_binary_list)):
            if  l is None:
                continue
            mask_file = f'pred_mask/{str(i).zfill(3)}.txt'
            f.write(f'{mask_file} {l} {c}\n')
            np.savetxt(os.path.join(save_path, mask_file), m.numpy(), fmt='%d')

    return mask2pixel_lookup, mask2pixel_lookup_score

if __name__ == "__main__":
    all_point = torch.load("all_points.pt").cpu()
    binary_mask = torch.load("gt_mask.pt").cpu()
    SNAP_location = "export/Snap"
    Lookup_location = "Lookup_dict"
    scene_id = "scene0011_00"
    mask_classfication(binary_mask, all_point, scene_id, 1000, 1000, SNAP_location, Lookup_location, "export/final_result")
