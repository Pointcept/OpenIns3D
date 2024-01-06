"""
Mask-Snap-Lookup Script for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)
"""

import torch
from openins3d.lookup import *
from openins3d.snap import *
from openins3d.build_lookup_dict import *
from openins3d.utils import save_visulization_3d_viz
from tqdm import tqdm
import argparse
import sys
sys.path.append("./ODISE")
from openins3d.utils import generate_detection_results
import torch


def get_args():
    
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='OpenIns3D inference')

    parser.add_argument('--processed_scene', default="data/processed/s3dis", type=str, help='the path of the processed dataset')
    parser.add_argument('--scans_path', default="input_data/scans", type=str, help='the path of the colored point cloud ply file, if it exists')
    parser.add_argument('--img_size', default=1000, type=int, help='dimension of the square image')
    parser.add_argument('--ca_mask_path', default="s3dis_saved/s3dis_masks_sparse", type=str, help='the generated class-agnostic mask from MPM, stored as a sparse tensor')
    parser.add_argument('--dataset', default="s3dis", type=str, help='dataset for inference, could be s3dis, scannet, stpls3d')
    parser.add_argument('--result_save', default="output_result_s3dis_new", type=str, help='Where to save the final detection results')
    parser.add_argument('--byproduct_save', default="saved_s3dis", type=str, help='Where to save the byproducts, including snapshot images and lookup_dict')
    parser.add_argument('--save_results_in_2d', default=False, type=bool, help='whether to save the results with 2D visualization. Not recommended, as it slows down the inference massively')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = get_args()    
    result_save_path = args.result_save
    byproduct_save_path = args.byproduct_save
    height, width = args.img_size, args.img_size
    scans_path = args.scans_path
    val_scenes_path = args.processed_scene
    val_scenes_path_list = glob.glob(args.ca_mask_path + "/*")
    dataset = args.dataset

    if dataset == "scannet":
        val_scenes = [scene.split("/")[-1].split(".")[0].split("_masks")[0] for scene in val_scenes_path_list]
        vocab = "cabinet; bed; chair; sofa; table; door; window; bookshelf; picture; counter; desk; curtain; refrigerator; showercurtain; toilet; sink; bathtub" 
        CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
        VALID_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    elif dataset == "s3dis":
        val_scenes = [scene.split("/")[-1].split(".")[0].split("_5_")[1].split("_masks")[0] for scene in val_scenes_path_list]
        vocab = "ceiling; floor; wall; beam; column; window; door; table; chair; sofa; bookcase; board"
        CLASS_LABELS = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board']
        VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).tolist()

    ca_mask_path = args.ca_mask_path
    odise_model = load_2d_model(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    snap_save_path = f"{byproduct_save_path}/Snap/"
    lookup_save_path = f"{byproduct_save_path}/Lookup_dict/"
    result_mask_save_path = args.result_save
    finished_scene_path = glob.glob(result_mask_save_path+"/*")
    finished_scene = [scene.split("/")[-1] for scene in finished_scene_path]

    res = [x for x in val_scenes if x not in finished_scene]
    val_scenes[:] = res

    for scene_id in tqdm(val_scenes):
        print(f"start to process {scene_id}")
        if dataset == "scannet":
            pointcloud_file = f"{val_scenes_path}/{scene_id[5:]}.npy"
        elif dataset == "s3dis":
            pointcloud_file = f"{val_scenes_path}/{scene_id}.npy"
        pcd = np.load(pointcloud_file)
        xyz, rgb = pcd[:,:3], pcd[:,3:6]
        scan_pc = torch.from_numpy(np.hstack([xyz, rgb]))
        
        # snap
        adjust_camera = [10, 2, 0.6]
        if dataset == "scannet":
            ply_file = f"{scans_path}/{scene_id}/{scene_id}_vh_clean_2.ply"
            image_generation_mesh(ply_file, height, width, scene_id, snap_save_path, adjust_camera=adjust_camera)
        elif dataset == "s3dis":
            image_generation_pcd(scan_pc, height, width, scene_id, snap_save_path, adjust_camera=adjust_camera)

        build_lookup_dict_one_scene(odise_model, scene_id, snap_save_path, lookup_save_path)        

        # load mask 
    
        if dataset == "scannet":
            binary_mask = torch.load(f"{ca_mask_path}/{scene_id}_masks.pt").to_dense()
        elif dataset == "s3dis":
            binary_mask = torch.load(f"{ca_mask_path}/Area_5_{scene_id}_masks.pt").to_dense()

        # lookup and save_results
        mask2pixel_lookup, _ = mask_classfication(binary_mask, scan_pc, adjust_camera, scene_id, height, width, snap_save_path, lookup_save_path, result_mask_save_path, CLASS_LABELS, VALID_CLASS_IDS)
        
        # save the results as 2D image
        if args.save_results_in_2d:
            result_save_path = f"{byproduct_save_path}/Result_vis_2d/"
            detection_results, detected_label_id = generate_detection_results(mask2pixel_lookup, binary_mask, CLASS_LABELS, VALID_CLASS_IDS)
            save_results_2d(scan_pc, height, width, scene_id, result_save_path, adjust_camera, detection_results)

        