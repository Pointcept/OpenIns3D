import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
import pyviz3d.visualizer as viz
import plyfile
import os
import torch
import random
from plyfile import PlyData, PlyElement

def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
        return vertices, faces

def create_color_map_and_legend(class_labels, class_ids, save_legend):
    # Use the 'tab20' colormap for more distinct colors
    class_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(class_ids)))
    # Create a color map dictionary
    color_map = {class_id: to_rgba(class_colors[i]) for i, class_id in enumerate(class_ids)}
    # Create a rectangular color bar
    fig, ax = plt.subplots(figsize=(1, 4))  # Adjusted figsize to 100x20
    cmap_bar = plt.cm.colors.ListedColormap(class_colors)
    bounds = np.arange(len(class_labels) + 1)
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap_bar.N)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_bar), cax=ax, ticks=np.arange(len(class_labels)))
    cb.set_ticklabels(class_labels)
    # Set tick labels with the same color as the color bar
    tick_labels = ax.get_yticklabels()
    for tick_label, class_label, class_id in zip(tick_labels, class_labels, class_ids):
        tick_label.set_color(color_map[class_id])
    # Save legend image
    plt.savefig(save_legend, bbox_inches='tight', pad_inches=0.1, transparent=True)
    # Plot a single rectangle and its name next to it
    rect_width = 5
    rect_height = 2
    rect_color = color_map[1]  # Use the color corresponding to the first class_id
    rect = Rectangle((5, 5), rect_width, rect_height, linewidth=1, edgecolor='black', facecolor=rect_color)
    ax.add_patch(rect)
    plt.text(5 + rect_width + 0.1, 5 + rect_height / 2, class_labels[0], va='center', ha='left', fontsize=8,
             color=rect_color)
    return color_map

def writeply(filename,xyz,rgb):
    """write into a ply file"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

def save_visulization_3d(original_pcd, vocab, mask_after_MPM, detected_mask, mask_code, scene_id, save_path, save_ply = False):
    # add original rgb
    v = viz.Visualizer()
    v.add_points(f'{scene_id[:6]}_rgb', original_pcd[:,:3], original_pcd[:,3:6], point_size=20, visible=True)
    string_list = vocab.split(';')
    CLASS_LABELS = [s.strip() for s in string_list]
    VALID_CLASS_IDS = np.arange(1, len(CLASS_LABELS) + 1).tolist()
    
    if not os.path.exists(f"{save_path}/{scene_id}"):os.makedirs(f"{save_path}/{scene_id}") 
    color_map = create_color_map_and_legend(CLASS_LABELS, VALID_CLASS_IDS, save_legend = f"{save_path}/{scene_id}/legend.png")
    rgb = original_pcd[:,3:6]
    num_mask = mask_after_MPM.shape[1]
    all_mask_colorcoded = rgb.copy()
    for i in range (num_mask):
        mask = mask_after_MPM[:,i]
        random_color = lambda: random.randint(0, 255)
        all_mask_colorcoded[mask,:] = torch.tensor([random_color(), random_color(), random_color()])
    v.add_points(f'{scene_id[:6]}_masks', original_pcd[:, :3], all_mask_colorcoded, point_size=30, visible=True)
    
    num_mask_detected = detected_mask.shape[1]
    detected_mask_colorcoded = rgb.copy()
    for i in (range(num_mask_detected)):
        mask = detected_mask[:,i]    
        color = color_map[mask_code[i]][:3]
        detected_mask_colorcoded[mask,:] = torch.tensor(color) * 255

    v.add_points(f'{scene_id[:6]}_openins', original_pcd[:,:3], detected_mask_colorcoded, point_size=30, visible=True)
    v.save(f'{save_path}/{scene_id}/viz')
    if save_ply:
        writeply(f'{save_path}/{scene_id}/all_mask.ply',original_pcd[:, :3],all_mask_colorcoded)
        writeply(f'{save_path}/{scene_id}/detected_mask.ply',original_pcd[:, :3],detected_mask_colorcoded)


def save_visulization_3d_viz(original_pcd, mask_after_MPM, scene_id, save_path):
    v = viz.Visualizer()
    rgb = original_pcd[:,3:6]
    num_mask = mask_after_MPM.shape[1]
    all_mask_colorcoded = rgb.clone()
    for i in range (num_mask):
        mask = mask_after_MPM[:,i]
        random_color = lambda: random.randint(0, 255)
        all_mask_colorcoded[mask,:] = torch.tensor([random_color(), random_color(), random_color()]).float()
    v.add_points(f'{scene_id[:6]}_masks', original_pcd[:, :3].numpy(), all_mask_colorcoded.numpy(), point_size=30, visible=True)
    v.save(f'{save_path}/{scene_id}/viz')


def generate_detection_results(mask2pixel_lookup, binary_mask, CLASS_LABELS, VALID_CLASS_IDS):
    detected_mask_idx = []
    detected_label = []
    detected_label_id = []
    for mask_idx in mask2pixel_lookup.keys():
        if mask2pixel_lookup[mask_idx] != None:
            detected_mask_idx.append(mask_idx)
            detected_label.append(CLASS_LABELS[VALID_CLASS_IDS.index(mask2pixel_lookup[mask_idx])])
            detected_label_id.append(mask2pixel_lookup[mask_idx])
            
    detected_mask_bin = binary_mask[:, detected_mask_idx]

    labels_list = []
    masks_binary_list = []

    num_mask = binary_mask.shape[1]
    for mask_idx in range(0, num_mask):
        labels_list.append(mask2pixel_lookup[mask_idx])
        masks_binary_list.append(binary_mask[:, mask_idx])

    detection_results = (detected_mask_bin, detected_label)
    return detection_results, detected_label_id
