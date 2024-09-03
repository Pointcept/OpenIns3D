from snap import Snap
from lookup import Lookup
from glob import glob
import torch
import os
import numpy as np
from evaluation.evaluate import evaluate
from evaluation.evaluate_bbox import evaluate_bbox
from utils import get_label_and_ids, openworld_recognition, display_results
from collections import defaultdict
import argparse
from tqdm import tqdm

class OpenIns3D:

    """
    This is the main code for OpenIns3D that combined Mask, Snap and Lookup for open world scene understanding.
    """

    def __init__(self, dataset_name, dataset_path, snap_folder = "output/snap", save_folder = "output/results", image_detector = "yoloworld", mode = "OVIS", use_2d = False, vis = True): 

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.save_folder = save_folder
        self.CLASS_LABELS, self.VALID_CLASS_IDS = get_label_and_ids(dataset_name)
        self.initilize_snap_lookup_parameters(dataset_name, use_2d)
        self.use_2d = use_2d
        self.save_vis = vis
        if self.use_2d:
            snap_folder = f"{dataset_path}/rgbd"
        self.snap = Snap(self.image_size, self.adjust_camera, snap_folder)
        self.lookup = Lookup(self.image_size, self.adjust_camera[2], snap_folder, text_input=self.CLASS_LABELS, results_folder = save_folder)

        # get all the scene names for the dataset
        self.scene_name = [scene.split('/')[-1].split('.')[0] for scene in sorted(glob(f"{self.dataset_path}/scenes/*"))]
        if image_detector.lower() == "yoloworld":
            self.lookup.call_YOLOWORLD()
        elif image_detector.lower() == "odise":
            self.lookup.call_ODISE()

        print(f"Total number of scenes in the {self.dataset_name} dataset: {len(self.scene_name)}")

        self.finished_scene = [
            scene for scene in self.scene_name
            if os.path.exists(f"{self.save_folder}/{scene}/{scene}_mask_classfication.pt")
        ]  # this is designed to continue from the last scene if the code is interrupted

        if mode == "OVIS": # open vocabulary instance segmentation
            self.OV_instance_seg()
        elif mode == "OVOR": # open vocabulary object recognition
            self.OV_object_recognition()
        elif mode == "OVOD": # open vocabulary object detection
            self.OV_object_detection()
        else:
            raise ValueError("Invalid mode. Please choose from OVOR, OVOD, OVIS")

    def OV_instance_seg(self):
    
        predict_results = {}
        for scene in tqdm(self.scene_name):
            scene_path, scene_mask, pcd_rgb = self.get_files_for_scene(scene, gt_or_detected_mask='detected')  #  mesh, mask, pcd_rgb
            # mask
            mask_list = torch.load(scene_mask).to_dense()
            if mask_list.size(1) == 0: 
                predict_results[scene] = {
                'pred_masks': torch.zeros((1, 0)),
                'pred_scores': torch.zeros(0),
                'pred_classes': torch.ones(0)*255}
                continue
                
            if scene in self.finished_scene:
                print(f"Scene {scene} is already processed")
                mask_classfication = torch.load(f"{self.save_folder}/{scene}/{scene}_mask_classfication.pt")
                score = torch.load(f"{self.save_folder}/{scene}/{scene}_score.pt")
                if self.dataset_name == 'stpls3d':
                    pcd_rgb = pcd_rgb[0]
            else:
                print(f"Processing scene {scene}")
                # snap
                if self.dataset_name == 'stpls3d':
                    self.snap.scene_image_rendering(pcd_rgb[1], scene, mode=["global"])
                    pcd_rgb = pcd_rgb[0]
                elif self.dataset_name == 's3dis':
                    self.snap.scene_image_rendering(pcd_rgb, scene, mode=["global", "wide", "corner"])
                elif (self.dataset_name == 'replica' or self.dataset_name == 'scannet') and not self.use_2d:
                    self.snap.scene_image_rendering(scene_path, scene, mode=["global", "wide", "corner"])
                # lookup
                mask_classfication, score = self.lookup.lookup_pipelie(pcd_rgb, mask_list, scene, threshold = 0.3, use_2d = self.use_2d)
                # log the results
                torch.save(mask_classfication, f"{self.save_folder}/{scene}/{scene}_mask_classfication.pt")
                torch.save(score, f"{self.save_folder}/{scene}/{scene}_score.pt")

            # evaluate
            pred_masks, pred_scores, pred_class, pred_class_txt = self.filter_results(mask_classfication, score, mask_list)
             
            predict_results[scene] = {
                'pred_masks': pred_masks if pred_masks.shape[1] > 0 else torch.zeros((1, 0)),
                'pred_scores':pred_scores if len(pred_masks) > 0 else torch.zeros(1),
                'pred_classes': pred_class if len(pred_masks) > 0 else torch.ones(1)*255
            }

            if self.save_vis:
                if pred_masks.shape[1] == 0 : # no mask in this scene
                    continue
                if self.dataset_name == "replica" and self.use_2d:
                    self.snap = Snap([800, 800], [3, 0.1, 1.0], f"{self.save_folder}")
                if self.dataset_name == 's3dis' or self.dataset_name == 'stpls3d':
                    self.snap.scene_image_rendering(pcd_rgb, f"{scene}_vis", mode=["global"], mask=[pred_masks, pred_class_txt]) # Comment this out unless visualization is needed, as it is quite slow.
                else:
                    self.snap.scene_image_rendering(scene_path, f"{scene}_vis", mode=["global"], mask=[pred_masks, pred_class_txt])

        evaluate(predict_results, f"{dataset_path}/ground_truth/", self.CLASS_LABELS, self.VALID_CLASS_IDS,  f"{self.save_folder}/{self.dataset_name}_final_result.csv")

    def OV_object_detection(self):
        predict_results = {}
        scene_pcds = [] # this is used to calculate the bbox when evaluating
    
        for scene in tqdm(self.scene_name):
            scene_path, scene_mask, pcd_rgb = self.get_files_for_scene(scene, gt_or_detected_mask='detected')  #  mesh, mask, pcd_rgb
            # mask
            mask_list = torch.load(scene_mask).to_dense()
            if mask_list.size(1) == 0: 
                predict_results[scene] = {
                'pred_masks': torch.zeros((1, 0)),
                'pred_scores': torch.zeros(0),
                'pred_classes': torch.ones(0)*255}
                continue
                
            if scene in self.finished_scene:
                print(f"Scene {scene} is already processed")
                mask_classfication = torch.load(f"{self.save_folder}/{scene}/{scene}_mask_classfication.pt")
                score = torch.load(f"{self.save_folder}/{scene}/{scene}_score.pt")
                if self.dataset_name == 'stpls3d':
                    pcd_rgb = pcd_rgb[0]
            else:
                print(f"Processing scene {scene}")
                # snap
                if self.dataset_name == 'stpls3d':
                    self.snap.scene_image_rendering(pcd_rgb[1], scene, mode=["global"])
                    pcd_rgb = pcd_rgb[0]
                elif self.dataset_name == 's3dis':
                    self.snap.scene_image_rendering(pcd_rgb, scene, mode=["global", "wide", "corner"])
                elif self.dataset_name == 'replica' or self.dataset_name == 'scannet':
                    self.snap.scene_image_rendering(scene_path, scene, mode=["global", "wide", "corner"])
                # lookup
                mask_classfication, score = self.lookup.lookup_pipelie(pcd_rgb, mask_list, scene, threshold = 0.5 , use_2d = self.use_2d)
                # log the results
                torch.save(mask_classfication, f"{self.save_folder}/{scene}/{scene}_mask_classfication.pt")
                torch.save(score, f"{self.save_folder}/{scene}/{scene}_score.pt")

            # evaluate
            pred_masks, pred_scores, pred_class, pred_class_txt = self.filter_results(mask_classfication, score, mask_list)

            predict_results[scene] = {
                'pred_masks': pred_masks if pred_masks.shape[1] > 0 else torch.zeros((1, 0)),
                'pred_scores':pred_scores if len(pred_masks) > 0 else torch.zeros(0),
                'pred_classes': pred_class if len(pred_masks) > 0 else torch.ones(0)*255
            }

            scene_pcds.append(torch.from_numpy(pcd_rgb))

            if self.save_vis:
                if pred_masks.shape[1] == 0 : # no mask in this scene
                    continue
                if self.dataset_name == "replica" and self.use_2d:
                    self.snap = Snap([800, 800], [3, 0.1, 1.0], f"{self.save_folder}")
                if self.dataset_name == 's3dis' or self.dataset_name == 'stpls3d':
                    self.snap.scene_image_rendering(pcd_rgb, f"{scene}_vis", mode=["global"], mask=[pred_masks, pred_class_txt]) # Comment this out unless visualization is needed, as it is quite slow.
                else:
                    self.snap.scene_image_rendering(scene_path, f"{scene}_vis", mode=["global"], mask=[pred_masks, pred_class_txt])

        evaluate_bbox(predict_results, f"{dataset_path}/ground_truth/", scene_pcds, self.CLASS_LABELS, self.VALID_CLASS_IDS,  f"{self.save_folder}/{self.dataset_name}_final_result.csv")

    def OV_object_recognition(self):

        class_counts = defaultdict(lambda: {'correct': 0, 'total': 0}) # for top-1 accuracy

        for scene in tqdm(self.scene_name):
            scene_path, scene_mask, pcd_rgb = self.get_files_for_scene(scene, gt_or_detected_mask='detected')  #  mesh, mask, pcd_rgb
            # mask
            mask_list = torch.load(scene_mask).to_dense()
            if mask_list.size(1) == 0: 
                continue 
            if scene in self.finished_scene:
                print(f"Scene {scene} is already processed")
                mask_classfication = torch.load(f"{self.save_folder}/{scene}/{scene}_mask_classfication.pt")
                score = torch.load(f"{self.save_folder}/{scene}/{scene}_score.pt")
            else:
                print(f"Processing scene {scene}")
                # snap
                if self.dataset_name == 'stpls3d':
                    self.snap.scene_image_rendering(pcd_rgb[1], scene, mode=["global"])
                    pcd_rgb = pcd_rgb[0]
                elif self.dataset_name == 's3dis':
                    self.snap.scene_image_rendering(pcd_rgb, scene, mode=["global", "wide", "corner"])
                elif self.dataset_name == 'replica' or self.dataset_name == 'scannet':
                    self.snap.scene_image_rendering(scene_path, scene, mode=["global", "wide", "corner"])
                # lookup
                mask_classfication, score = self.lookup.lookup_pipelie(pcd_rgb, mask_list, scene, threshold = 0.5, use_2d = self.use_2d)
                # log the results
                torch.save(mask_classfication, f"{self.save_folder}/{scene}/{scene}_mask_classfication.pt")
                torch.save(score, f"{self.save_folder}/{scene}/{scene}_score.pt")
            # load the raw prediction

            predict_labels = [self.VALID_CLASS_IDS[i] if i != -1 else -1 for i in mask_classfication]
            class_counts = openworld_recognition(mask_list, predict_labels, f"{dataset_path}/ground_truth/{scene}.txt", class_counts, self.VALID_CLASS_IDS)

        display_results(class_counts, self.VALID_CLASS_IDS, self.CLASS_LABELS)

    def initilize_snap_lookup_parameters(self, dataset_name, use_2d = False):


        if dataset_name in ['scannet', 's3dis', 'replica'] and not (dataset_name == 'replica' and use_2d) and not (dataset_name == 'scannet' and use_2d):

            image_width, image_height = 800, 800
            lift_cam, zoomout, remove_lip = 3, 0.1, 1.0
        elif dataset_name == 'stpls3d':
            image_width, image_height = 1000, 1000
            lift_cam, zoomout, remove_lip = 20, 0.1, 0
        elif dataset_name == 'replica' and use_2d:
            image_width, image_height = 360, 640
            lift_cam, zoomout, remove_lip = 2, 0.1, 0

        elif dataset_name == 'scannet' and use_2d:
            image_width, image_height = 480, 640
            lift_cam, zoomout, remove_lip = 2, 0.1, 0


        self.image_size = [image_width, image_height]
        self.adjust_camera = [lift_cam, zoomout, remove_lip]            

    def filter_results(self, mask_classfication, score, scene_mask):

        all_valid_idx = [i for i in range(len(mask_classfication)) if mask_classfication[i] != -1]
        class_id_results = [self.VALID_CLASS_IDS[i] for i in mask_classfication if i != -1]
        score = [score[i] for i in range(len(mask_classfication)) if mask_classfication[i] != -1]
        txt_results = [self.CLASS_LABELS[i] for i in mask_classfication if i != -1]
        mask_final = scene_mask[:, all_valid_idx]

        return mask_final, torch.tensor(score), torch.tensor(class_id_results), txt_results

    def get_files_for_scene(self, scene_name, gt_or_detected_mask = 'detected'):

        if self.dataset_name == "s3dis":
            mesh_path = f"{self.dataset_path}/scenes/{scene_name}.npy"
            scene_mask = f"{self.dataset_path}/masks/detected/{scene_name}.pt" if gt_or_detected_mask == 'detected' else f"{self.dataset_path}/masks/ground_truth/{scene_name}.pt"
            pcd = np.load(mesh_path)
            pcd = torch.tensor(pcd)
        elif self.dataset_name == "stpls3d":
            mesh_path = f"{self.dataset_path}/scenes/{scene_name}.npy"
            mesh_path_raw = f"{self.dataset_path}/raw_pcd/{scene_name}.npy"
            scene_mask = f"{self.dataset_path}/masks/detected/{scene_name}.pt" if gt_or_detected_mask == 'detected' else f"{self.dataset_path}/masks/ground_truth/{scene_name}.pt"
            pcd = torch.from_numpy(np.load(mesh_path))
            pcd_raw = torch.from_numpy(np.load(mesh_path_raw))
            return mesh_path, scene_mask, [pcd, pcd_raw]
        else:
            mesh_path = f"{self.dataset_path}/scenes/{scene_name}.ply"
            scene_mask = f"{self.dataset_path}/masks/detected/{scene_name}.pt" if gt_or_detected_mask == 'detected' else f"{self.dataset_path}/masks/ground_truth/{scene_name}.pt"
            pcd = self.snap.read_plymesh(mesh_path)[0]

        return mesh_path, scene_mask, pcd


def get_args():
    
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='OpenIns3D')
    parser.add_argument('--dataset', default="Replica", type=str, help='dataset to test on')
    parser.add_argument('--task', default="OVIS", type=str, help='tasks to perform: could be OVOD, OVIS, OVOR')
    parser.add_argument('--detector', default="yoloworld", type=str, help='image detector to use: could be yoloworld, odise')
    parser.add_argument('--use_2d', default=False, type=lambda x: (str(x).lower() == 'true'), help='Use 2D images or not (for ScanNet and Replica).')
    # add visualization
    parser.add_argument('--save_vis', default=True, type=lambda x: (str(x).lower() == 'true'), help='Visualize the results or not.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    dataset_name = args.dataset
    dataset_path = f'./data/{dataset_name.lower()}'
    openins3d = OpenIns3D(dataset_name.lower(), dataset_path, image_detector = args.detector, mode = args.task, use_2d = args.use_2d, vis = args.save_vis)