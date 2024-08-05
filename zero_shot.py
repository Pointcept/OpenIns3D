"""
Zero-shot inference Script for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)
"""

import torch
import argparse
from openins3d.mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud
import torch
from openins3d.lookup import Lookup
from openins3d.snap import Snap
from openins3d.utils import get_label_and_ids

import numpy as np

def get_args():
    
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='OpenIns3D')

    parser.add_argument('--pcd_path', default="demo.ply", type=str, help='path for 3d scans, could be .ply or .npy with shape (N, 6)')
    parser.add_argument('--vocab', nargs='*' , help='list of class names of interests')
    parser.add_argument('--detector', type=str, default="odise", help='choose between odise and yoloworld')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    print("start to load models>>>>>>>>>>>>>>>>>>>>>>>>")
    # mask proposal module
    argparse = get_args()
    pcd_path = argparse.pcd_path
    detector = argparse.detector
    additional_vocab = argparse.vocab
    
    vocab= get_label_and_ids("replica")[0]
    if additional_vocab is not None:
        vocab.extend(additional_vocab)



    name_of_scene = pcd_path.split("/")[-1].split(".")[0]
    if pcd_path.endswith(".ply"):
        mesh = load_mesh(pcd_path)
        pcd_rgb = np.hstack([np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors) * 255.])
    elif pcd_path.endswith(".npy"):
        pcd_rgb = np.load(pcd_path)[:, :6]
    else:
        raise ValueError("Unsupported file format")

    # load model and generate masks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("third_party/scannet200_val.ckpt").to(device).eval()
    data, features, _, inverse_map = prepare_data(pcd_rgb, device)
    with torch.no_grad():
        mask_list = map_output_to_pointcloud(model(data, raw_coordinates=features), inverse_map, 0.5)

    image_size = [800, 800]
    adjust_camera = [2, 0.1, 1.0] # [lift_cam, zoomout, remove_lip]
    snap = Snap(image_size, adjust_camera, "output/snap_demo")
    lookup = Lookup(image_size, adjust_camera[2], "output/snap_demo", text_input=vocab, results_folder="output/results_demo")

    if detector =="odise":
        lookup.call_ODISE()
    elif detector == "yoloworld":
        lookup.call_YOLOWORLD()

    # snap and lookup
    snap.scene_image_rendering(torch.tensor(pcd_rgb).float(), name_of_scene, mode=["global", "wide", "corner" ])
    mask_classfication, score = lookup.lookup_pipelie(pcd_rgb, mask_list, name_of_scene, threshold = 0.5)

    results_txt = [vocab[i] for i in mask_classfication if i != -1]
    mask_final = mask_list[:, [i for i in range(len(mask_classfication)) if mask_classfication[i] != -1]]

    snap.scene_image_rendering(torch.tensor(pcd_rgb).float(), f"{name_of_scene}_vis", mode=["global"], mask=[mask_final, results_txt])

    print("\n" + "="*50)
    print(f"*** Finished processing the scene! Results are saved in 'output/snap_demo/{name_of_scene}_vis/image/' folder. ***")
    print("="*50 + "\n")