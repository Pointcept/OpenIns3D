import os
from PIL import Image
import numpy as np
import torch
from torch_scatter import scatter_max
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
import cv2
import torch.nn.functional as F
import pandas as pd
import plyfile


def get_label_and_ids(dataset):
    if dataset == "s3dis":
        CLASS_LABELS = [
            'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 
            'door', 'table', 'chair', 'sofa', 'bookcase', 'board'
        ]
        VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        
    elif dataset == "scannet":
        CLASS_LABELS = [
            "cabinet", "bed", "chair", "sofa", "table", "door", "window", 
            "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", 
            "shower curtain", "toilet", "sink", "bathtub"
        ]
        VALID_CLASS_IDS = np.array([
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36
        ])
        
    elif dataset == "replica":
        CLASS_LABELS = [
            "basket", "bed", "bench", "bin", "blanket", "blinds", "book", "bottle", 
            "box", "bowl", "camera", "cabinet", "candle", "chair", "clock", "cloth", 
            "comforter", "cushion", "desk", "desk-organizer", "door", "indoor-plant", 
            "lamp", "monitor", "nightstand", "panel", "picture", "pillar", "pillow", 
            "pipe", "plant-stand", "plate", "pot", "sculpture", "shelf", "sofa", 
            "stool", "switch", "table", "tablet", "tissue-paper", "tv-screen", 
            "tv-stand", "vase", "vent", "wall-plug", "window", "rug"
        ]
        VALID_CLASS_IDS = np.array([
            3, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 34, 
            35, 37, 44, 47, 52, 54, 56, 59, 60, 61, 62, 63, 64, 65, 70, 71, 76, 78, 
            79, 80, 82, 83, 87, 88, 91, 92, 95, 97, 98
        ])
        
    elif dataset == "stpls3d":
        CLASS_LABELS = [
            "buildings",
            "vegetation",
            "vehicle",
            "truck",
            "aircraft",
            "military vehicle",
            "bike",
            "motorcycle",
            "light pole",
            "street sign",
            "clutter",
            "fence",
        ]
        VALID_CLASS_IDS = np.array(
            [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return CLASS_LABELS, VALID_CLASS_IDS

def generate_distinguished_colors(num_colors, seed=42):
    random.seed(seed)
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
    random.shuffle(rgb_colors)
    rgb_colors = np.array(rgb_colors)
    return rgb_colors

color_map_array = generate_distinguished_colors(100)

def get_image_resolution(image_path):
    """
    Get the resolution of an image.

    :param image_path: Path to the image file
    :return: A tuple containing the width and height of the image
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
        return vertices, faces

def to_bounding_boxes(masks):

    """
    convert mask to bbox
    input: masks: torch.tensor, [width, height, num_masks] this store the index of the mask on each pixel
    output: bboxes: torch.tensor, [num_masks, 4], where 4 is [x_min, y_min, x_max, y_max] for the bbox
    code wrote by gpt4o
    """
    
    # Flatten the mask tensor to 2D
    H, W, N = masks.shape
    masks_flat = masks.view(H * W, N)

    # Get the coordinates of each pixel
    y_coords = torch.arange(H, device=masks.device).repeat_interleave(W)
    x_coords = torch.arange(W, device=masks.device).repeat(H)

    # Expand the coordinates to match the flattened masks shape
    y_coords = y_coords.view(-1, 1).expand(-1, N)
    x_coords = x_coords.view(-1, 1).expand(-1, N)

    # Mask the coordinates
    y_coords_masked = y_coords * masks_flat
    x_coords_masked = x_coords * masks_flat

    # Use a large integer value for masking
    large_value = torch.iinfo(torch.int32).max

    # Set masked out coordinates to a large value (for min) and a small value (for max)
    y_coords_masked = y_coords_masked.masked_fill(~masks_flat, large_value)
    x_coords_masked = x_coords_masked.masked_fill(~masks_flat, large_value)

    y_min = y_coords_masked.min(dim=0).values
    x_min = x_coords_masked.min(dim=0).values

    y_coords_masked = y_coords_masked.masked_fill(~masks_flat, -large_value)
    x_coords_masked = x_coords_masked.masked_fill(~masks_flat, -large_value)

    y_max = y_coords_masked.max(dim=0).values
    x_max = x_coords_masked.max(dim=0).values

    # Stack the results into a (N, 4) tensor
    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    
    # Replace large_value with -1 for masks with no non-zero elements
    bboxes = bboxes.masked_fill(bboxes == large_value, -1)
    bboxes = bboxes.masked_fill(bboxes == -large_value, -1)

    return bboxes

def plot_bounding_boxes(image_path, bboxes, labels, output_path):
    """
    image_path: original image path
    bboxes: [N, 4] where N is the number of bounding boxes
    labels: [N] where each value is the label for the corresponding bounding box
    output_path: path to save the image with bounding boxes
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if torch.all(bboxes == -1):
        return None
    for (x1, y1, x2, y2), label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = tuple(random.randint(0, 255) for _ in range(3))
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

def assign_pred_mask_to_gt(pred_mask, project_mask, threshold):
    """
    pred_mask: [width, heigh, num_image] masks detected in the 2d rendered images
    project_mask: [width, heigh, num_image] projection of 3d masks on 2d images
    threshold: float iou threshold to consider a match

    output:
    assigned_masks: dict, key is the mask index in 3d, value is the mask index in 2d pred mask
    score_masks: dict, key is the mask index in 3d, value is the iou score
    """

    assigned_masks = {}
    score_masks = {}


    iou_matrix, unique_pred, unique_project = calculate_iou_matrix(pred_mask, project_mask) 

    # here the unique_pred and unique_project are without background. unique_pred start from 1 while unique_project start from 0
    iou_matrix[iou_matrix<threshold] = 0
    if iou_matrix.shape[0]==0: # meaning no masks detected in the 2d rendered images
        return assigned_masks, score_masks

    max_indices = np.argmax(iou_matrix, axis=0) # (num_of_unique_project) the index of best match in the pred mask for each project mask
    max_iou = np.max(iou_matrix, axis=0) # (num_of_unique_project) the iou score of the best match in the pred mask for each project mask

    for count, best_match_iou in enumerate(max_iou):
        targeting_proj_mask = int(unique_project[count]) 
        if best_match_iou !=0:
            assigned_masks[targeting_proj_mask] =  int(unique_pred[max_indices[count]])-1   # the pred mask index
            score_masks[targeting_proj_mask] = best_match_iou

    return assigned_masks, score_masks

def plot_mask_2_pixel_map(final_depth_map, name):
    
    plot_mask = final_depth_map.cpu().numpy()

    # Get unique values in the plot_mask
    unique_values, count = np.unique(plot_mask, return_counts=True)
    # Create random colors for each unique value
    fixed_color_map = generate_distinguished_colors(100, seed=42)

    # Ensure the background (if specified) is white
    bg_index = np.where(unique_values == -1)[0][0] if -1 in unique_values else None
    if bg_index is not None:
        fixed_color_map[bg_index] = (1.0, 1.0, 1.0)  # RGBA for white

    # Create the custom colormap
    custom_cmap = ListedColormap(fixed_color_map)

    # Plot the mask with the custom colormap
    plt.imshow(plot_mask, cmap=custom_cmap, interpolation='none')
    plt.savefig(f'{name}')
    plt.close()

def filter_pcd_with_depthmap(points, depth_intrinsic, depth, pose, depth_shift, device):
    """
    :param points: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :return p: N x 2 format
    """

    vis_thres = 0.1

    
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    
    points_world = torch.cat([points, torch.ones((points.shape[0], 1), dtype=torch.float64).to(device)], dim=-1).to(torch.float64)
    world_to_camera = torch.inverse(pose).to(torch.float64)
    
    p = torch.matmul(world_to_camera, points_world.T)  # [Xb, Yb, Zb, 1]: 4, n
    p[0] = ((p[0] - bx) * fx) / p[2] + cx 
    p[1] = ((p[1] - by) * fy) / p[2] + cy
    
    all_idx = torch.arange(0, len(points)).to(device)  # to save the corresponding point idx later as the prompt ID
    # out-of-image check
    idx = torch.unique(torch.cat([torch.where(p[0]<=0)[0], torch.where(p[1]<=0)[0], \
                                    torch.where(p[0]>=depth.shape[1]-1)[0], \
                                    torch.where(p[1]>=depth.shape[0]-1)[0]], dim=0), dim=0)
    keep_idx = all_idx[torch.isin(all_idx, idx, invert=True)]
    p = p[:, keep_idx]

    if p.shape[1] == 0:
        return keep_idx  # no 3D prompt is visible in this frame

    pi = torch.round(p).to(torch.int64)
    # Check occlusion
    est_depth = p[2]
    trans_depth = depth[pi[1], pi[0]] / depth_shift
    idx_keep = torch.where(torch.abs(est_depth - trans_depth) <= vis_thres)[0]
    keep_idx = keep_idx[idx_keep]
    return keep_idx

def generate_neighbors(thickness):
    
    size = (thickness - 1) * 2 + 1
    x_coords = torch.arange(0 - thickness + 1, 0 + thickness).unsqueeze(1).repeat(1, size)
    y_coords = torch.arange(0 - thickness + 1, 0 + thickness).repeat(size, 1)
    x_coords = x_coords.view(-1)
    y_coords = y_coords.view(-1)

    neighbor_indices = torch.column_stack((x_coords, y_coords)).cuda()

    return neighbor_indices

def display_rate_compute(mask_map, occlusion_map, num_points_for_masks, num_masks):
    
    "display_rate = number_of_point_not_occluded / total_number_of_points_for_the_mask"
    "number_of_point_not_occluded: number of points that has not been occluded by other masks or background; it is ok to be occluded by itself"
    max_mask_idx = num_masks - 1
    mask_idx_list = torch.range(0, max_mask_idx, device = 'cuda') # skip the background -1
    binary_mask = (mask_map == mask_idx_list).float()
    product = binary_mask * occlusion_map
    weighted_sum = torch.sum(product, axis=(0, 1)).int()
    # avoid zero division
    num_points_for_masks[num_points_for_masks == 0] = 1
    display_rate = weighted_sum / num_points_for_masks
    return display_rate

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
    
    index_xy_expand = index_xy[:, None, :] + generate_neighbors(point_size)

    neighborhood_size = ((point_size - 1) * 2 + 1)**2
    
    # only keypoint within the image boundary is considered, drop others
    mask = (index_xy_expand[:, :, 0] >= 1) & (index_xy_expand[:, :, 0] < width-1) & (index_xy_expand[:, :, 1] >= 1) & (index_xy_expand[:, :, 1] < height-1)
    mask = mask.all(dim=1)
    index_xy_expand = index_xy_expand[mask]
    index_xy = index_xy_expand.reshape(-1, 2)
    
    depth = depth[mask]

    depth = depth.unsqueeze(1).expand(depth.shape[0], neighborhood_size).reshape(-1)

    unique_idx, indices = torch.unique(index_xy, dim=0, return_inverse=True)
    
    out, _ = scatter_max(depth, indices)

    mask_map = torch.ones((width, height), device = 'cuda') * -1 # -1 for background
    depth_map = torch.ones((width, height), device = 'cuda') * 999
    occlusion_map = torch.zeros((width, height), device = 'cuda')

    index = unique_idx
    mask_map[index[:, 0], index[:, 1]] = mask_idx_input
    depth_map[index[:, 0], index[:, 1]] = out.cuda()

    ones = torch.ones(index_xy.size(0), device='cuda:0')
    # Use scatter_add_ to add the ones tensor to the occlusion_map at the specified indices
    occlusion_map.index_put_((index_xy[:, 0], index_xy[:, 1]), ones, accumulate=True)
    return mask_map, depth_map, occlusion_map

def mask_rasterization(all_point, mask_bin, pose_matrix, depth_intrinsic, width, height, point_size = 1, name = None, display_rate_calculation = False):
    
    """
    Mask rasterization for mask-to-pixel mapping generation.
    This code breaks down the 3D to 2D projection for each mask to obtain the final pixel location, considering occlusion cases.
    It also calculates the display rate for each mask when display_rate_calculation is set to True.

    output: final_depth_map: torch.tensor, [width, height] where -1 is the background, and other values are the mask index
            display_rate: np.array, [num_masks, 1] where the display rate for each mask is calculated
    """

    num_masks = mask_bin.shape[1]
    # the list of mask_map, depth_map, occlusion_map for each mask, all in the shape of (width, height, num_masks)
    mask_map_total = torch.ones((width, height, num_masks), device = 'cuda')  # -1 for background
    depth_map_total = torch.ones((width, height, num_masks), device = 'cuda') * 999999 
    display_map_total = torch.zeros((width, height, num_masks), device = 'cuda')

    num_points_for_masks = torch.sum(mask_bin.bool(), axis=0).cuda()

    for mask_idx in range(num_masks):
        indices = torch.nonzero(mask_bin[:, mask_idx])
        if indices.shape[0]<=1: # remove mask with 0 points, due to the removal of the lip
            continue
        obj_pcd_rgb = all_point[indices.squeeze()]
        mask_map, depth_map, occulsion_map = single_mask_projection(obj_pcd_rgb, depth_intrinsic, pose_matrix,mask_idx, width, height, point_size)
        mask_map_total[:,:,mask_idx] = mask_map
        depth_map_total[:,:,mask_idx] = depth_map
        display_map_total[:,:,mask_idx] = occulsion_map

    depth_map_max_idx = torch.argmin(depth_map_total, axis=-1)
    condensed_depth_map = mask_map_total[torch.arange(mask_map_total.shape[0])[:, None, None], torch.arange(mask_map_total.shape[1])[None, :, None], depth_map_max_idx[:, :, None]]

    final_depth_map = condensed_depth_map.int().transpose(1,0).squeeze()

    if display_rate_calculation: # to be refined
        display_rate = display_rate_compute(condensed_depth_map, display_map_total, num_points_for_masks, num_masks)
    else:
        display_rate = None

    if name:
        plot_mask_2_pixel_map(final_depth_map, name)

    return final_depth_map, display_rate

def mask_lable_location(all_point, mask_bin, pose_matrix, depth_intrinsic, width, height, point_size = 1, name = None, display_rate_calculation = False):
    
    """
    Mask rasterization for mask-to-pixel mapping generation.
    This code breaks down the 3D to 2D projection for each mask to obtain the final pixel location, considering occlusion cases.
    It also calculates the display rate for each mask when display_rate_calculation is set to True.

    output: final_depth_map: torch.tensor, [width, height] where -1 is the background, and other values are the mask index
            display_rate: np.array, [num_masks, 1] where the display rate for each mask is calculated
    """

    mask2pxiel_map, _ = mask_rasterization(all_point, mask_bin, pose_matrix, depth_intrinsic, width, height, point_size = 1, name = None, display_rate_calculation = False)
    mask2pxiel_map = mask2pxiel_map.unsqueeze(-1)
    max_mask_idx = mask_bin.shape[1] - 1
    mask2pxiel_map[mask2pxiel_map == -1] = max_mask_idx + 1
    num_classes = int(max_mask_idx) + 2
    one_hot_encoded = F.one_hot(mask2pxiel_map.long(), num_classes=num_classes)
    binary_mask = one_hot_encoded.permute(0, 1, 3, 2).float()
    binary_mask = binary_mask[:, :, :-1, :]
    final_box = to_bounding_boxes(binary_mask[:, :, :, 0].bool())
    label_location = (final_box[:, [0, 1]] + final_box[:, [2, 3]]) / 2
    return label_location

def calculate_iou_matrix(pred_mask, project_mask):

    """
    input:
    pred_mask: [width, heigh, num_image] masks detected in the 2d rendered images
    project_mask: [width, heigh, num_image] projection of 3d masks on 2d images
    output:
    iou_matrix: [num_of_unique_pred, num_of_unique_project] iou matrix of the pred mask and project mask
    """
    pred_mask = pred_mask.cuda().reshape(-1,1)
    project_mask = project_mask.cuda().reshape(-1,1)
    unique_pred = torch.unique(pred_mask)
    unique_pred = unique_pred[unique_pred!=0]
    pred_mask_flatten = (pred_mask == unique_pred.view(1, -1)).float()
    pred_mask_flatten_t = pred_mask_flatten.t()
    unique_project = torch.unique(project_mask)
    unique_project = unique_project[unique_project!=-1]
    project_mask_flatten = (project_mask == unique_project.view(1, -1)).float()
    intersection = torch.mm(pred_mask_flatten_t, project_mask_flatten)
    union = pred_mask_flatten_t.sum(dim=1, keepdim=True) + project_mask_flatten.sum(dim=0, keepdim=True) - intersection
    iou_matrix = intersection / union
    return iou_matrix.cpu().numpy(), unique_pred, unique_project

def normalize_scores(predict_score, predict_list):
    """
    This takes all the matched results for one mask, normalize the score
    """
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
    return predict_ranks[0], sorted_data[0]

def openworld_recognition(gt_masks, predict_label_converted, gt_path, class_counts, classes_of_interest):
    gt_labels = torch.tensor(np.loadtxt(gt_path))
    
    # KNOWING THE GT FOR EACH MASK
    gt_idx = []
    num_mask = gt_masks.shape[1]
    for i in range(num_mask):
        this_mask = gt_masks[:, i].bool()
        this_label = gt_labels[this_mask]
        most_frequent_label = torch.mode(this_label)[0]
        gt_idx.append(int(most_frequent_label // 1000))


    if len(gt_idx) != len(predict_label_converted):
        raise ValueError("Ground truth and predictions lists must be of the same length.")

    # Initialize dictionaries to track counts
    for true_class, predicted_class in zip(gt_idx, predict_label_converted):
        if true_class in classes_of_interest:
            class_counts[true_class]['total'] += 1
            if true_class == predicted_class:
                class_counts[true_class]['correct'] += 1
    return class_counts

def display_results(class_counts, classes_of_interest, class_names):
    class_index_to_name = {idx: name for idx, name in zip(classes_of_interest, class_names)}

    # Print table header
    print(f"{'Class Index':<12} {'Class Name':<20} {'Accuracy':<10}")
    print('-' * 42)

    # Calculate and print accuracy for each class in the subset
    for cls in classes_of_interest:
        class_name = class_index_to_name.get(cls, 'Unknown')
        if cls in class_counts:
            counts = class_counts[cls]
            accuracy = counts['correct'] / counts['total']
            print(f"{cls:<12} {class_name:<20} {accuracy:.3f}")
        else:
            # If a class of interest has no samples in the ground truth, denote as NaN
            print(f"{cls:<12} {class_name:<20} NaN")

    # Optional: Print overall accuracy across the classes of interest
    overall_correct = sum(counts['correct'] for cls, counts in class_counts.items() if cls in classes_of_interest)
    overall_total = sum(counts['total'] for cls, counts in class_counts.items() if cls in classes_of_interest)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else np.nan

    print('-' * 42)
    print(f"Overall Accuracy: {overall_accuracy:.3f}" if not np.isnan(overall_accuracy) else "Overall Accuracy: NaN")

def downsample_image(image_path, size=(250, 250)):
    return Image.open(image_path).resize(size, Image.ANTIALIAS)

def display_snap_image(snap_image_path, size=(250, 250)): 
    # Get all PNG files in the directory
    image_files = [f for f in os.listdir(snap_image_path) if f.lower().endswith('png')]

    # Determine grid size
    grid_size = int(len(image_files) ** 0.5) + (1 if len(image_files) ** 0.5 % 1 > 0 else 0)

    # Plot images in a grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()
    for ax, image_file in zip(axes, image_files):
        img = Image.open(os.path.join(snap_image_path, image_file)).resize(size, Image.ANTIALIAS)
        ax.imshow(img)
        ax.set_title(f'Angle {image_file.split(".")[0]}', fontsize=10)
        ax.axis('off')

    for ax in axes[len(image_files):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
