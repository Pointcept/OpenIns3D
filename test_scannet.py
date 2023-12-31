"""
Mask-Snap-Lookup Script for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)
"""

import torch
from openins3d.lookup import *
from openins3d.snap import *
from openins3d.build_lookup_dict import *
from tqdm import tqdm
import argparse
import sys
sys.path.append("./ODISE")
from zero_shot import generate_detection_results
from openins3d.mask.mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, save_colorized_mesh 
import torch


def get_args():
    
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='OpenIns3D')

    # parser.add_argument('--ply_path', default="/home/zelda/zh340/myzone/OpenIns3D/data/scans/scene0011_00/scene0011_00_vh_clean_2.ply", type=str, help='the path of colored point cloud')
    parser.add_argument('--img_size', default=[1000,1000], help='the path of colored point cloud')
    parser.add_argument('--MPM_checkpoint', default="checkpoints/scannet200_val.ckpt", type=str, help='the path of MPM_checkpoint')
    parser.add_argument('--result_save', default="output_result", type=str, help='Where to save the pcd results')
    parser.add_argument('--byproduct_save', default = "saved", type=str, help='Where to save the byproduct, including snap images, masks, and lookup_dict')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = get_args()    
    model_path = args.MPM_checkpoint
    result_save_path = args.result_save
    byproduct_save_path = args.byproduct_save
    height, width = args.img_size[0], args.img_size[1]

    scannetv2_val_path = "openins3d/meta_data/scannetv2_val.txt"
    scans_path = "input_data/scans"
    with open(scannetv2_val_path) as val_file:
        val_scenes = val_file.read().splitlines()

    vocab = "cabinet; bed; chair; sofa; table; door; window; bookshelf; picture; counter; desk; curtain; refrigerator; showercurtain; toilet; sink; bathtub" 
    
    CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    VALID_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


    # load 2D detectors
    odise_model = load_2d_model(vocab)
    # load MPM model
    model = get_model(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
   
    snap_save_path = f"{byproduct_save_path}/Snap/"
    lookup_save_path = f"{byproduct_save_path}/Lookup_dict/"
    result_save_path = f"{byproduct_save_path}/Result_vis_2d/"
    result_mask_save_path = args.result_save
    finished_scene_path = glob.glob(result_mask_save_path+"/*")
    finished_scene = [scene.split("/")[-1] for scene in finished_scene_path]

    res = [x for x in val_scenes if x not in finished_scene]
    val_scenes[:] = res

    for scene_id in tqdm(val_scenes):
        print(f"start to process {scene_id}")

        pointcloud_file = f"{scans_path}/{scene_id}/{scene_id}_vh_clean_2.ply"

        # load input data
        mesh = load_mesh(pointcloud_file)
        pcd = np.asarray(mesh.vertices)
        rgb = np.asarray(mesh.vertex_colors) * 255
        all_point  = torch.from_numpy(np.hstack([pcd,rgb]))
        data, _, _, features, _, inverse_map = prepare_data(mesh, device)

        # run MPM
        with torch.no_grad():
            outputs = model(data, raw_coordinates=features)
        binary_mask = map_output_to_pointcloud(mesh, outputs, inverse_map, confidence_threshold = 0.6)

        # snap
        print("snap start")
        adjust_camera = [5, 0.1, 0.3]
        image_generation_mesh(pointcloud_file, height, width, scene_id, snap_save_path, adjust_camera=adjust_camera)
        
        print("build lookup dict")
        # build_lookup_dict
        build_lookup_dict_one_scene(odise_model, scene_id, snap_save_path, lookup_save_path)
        # mask2pixel lookup
        print("lookup")
        mask2pixel_lookup, _ = mask_classfication(binary_mask, all_point, scene_id, height, width, snap_save_path, lookup_save_path, result_mask_save_path, CLASS_LABELS, VALID_CLASS_IDS)

        # detection_results, detected_label_id = generate_detection_results(mask2pixel_lookup, binary_mask, CLASS_LABELS, VALID_CLASS_IDS)
        # save results in image
        # save_results_2d(all_point, height, width, scene_id, result_save_path, adjust_camera, detection_results)

        