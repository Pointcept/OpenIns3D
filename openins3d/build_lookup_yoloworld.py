"""
Script for Building Lookup Dict with Yoloworld for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)

reference:
YOLOWORLD: https://github.com/AILab-CVC/YOLO-World

"""

import os.path as osp
from torchvision.ops import nms
import torch
from mmengine.runner.amp import autocast
from tqdm import tqdm
from mmengine.dataset import Compose
from mmyolo.registry import RUNNERS
from mmengine.config import Config
from mmengine.runner import Runner
import glob
import cv2 
import os
import random
from utils import get_image_resolution, get_label_and_ids, plot_bounding_boxes

# add to sys path

import sys
sys.path.append("./third_party/YOLO-World")


class YOLOWORLD:
    """
    to call YOLO-World for mask understanding
    """
    def __init__(self, Snap_path, Save_path, vocab):
        self.texts = [[t] for t in vocab] + [[' ']]
        self.load_yolo()
        self.Save_path = Save_path
        self.Snap_path = Snap_path

    def load_yolo(self):
        # learnt from new work https://github.com/aminebdj/OpenYOLO3D for easy use of YOLO-World
        self.topk = 100
        self.th = 0.1
        self.nms = 0.2
        self.use_amp = False
        self.resolution = None  
        # Configuration and model paths
        config_path = "third_party/pretrained/configs/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        pre_trained_path = "third_party/pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"
        cfg = Config.fromfile(config_path)
        # Setup working directory and pre-trained model path
        work_dir_name = osp.splitext(osp.basename(config_path))[0]
        cfg.work_dir = osp.join(os.getcwd(), 'YOLO-World', 'yolo_world', 'work_dirs', work_dir_name)
        cfg.load_from = osp.join(os.getcwd(), pre_trained_path)
        # Initialize runner
        if 'runner_type' not in cfg:
            self.runner = Runner.from_cfg(cfg)
        else:
            self.runner = RUNNERS.build(cfg)
        # Prepare for model evaluation
        self.runner.call_hook('before_run')
        self.runner.load_or_resume()
        self.runner.pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
        self.runner.model.eval()

    def build_lookup_dict(self, scene, save = True): 

        num_images = len(glob.glob(f"{self.Snap_path}/{scene}/image/*"))
        scene_preds = {}
        all_images = []
        
        print(f"*****************Start to Build Lookup Dict for {scene}*****************")

        # Process each image
        for i in range(num_images):
            sys.stdout.write(f"\rLookup module: {i} out of {num_images} images are processed by YOLOWORLD")
            sys.stdout.flush()
            image_path = f"{self.Snap_path}/{scene}/image/image_rendered_angle_{i}.png"
            all_images.append(image_path)
            frame_prediction = self.inference_detector([image_path])
            scene_preds.update(frame_prediction)
        
        # Initialize templates for bounding boxes and labels
        bbox_template = torch.full((len(scene_preds), 40, 4), -1, dtype=torch.float32)
        label_template = torch.full((len(scene_preds), 40), -1, dtype=torch.int64)
                
        # Plot bounding boxes and update templates
        for i, (image_path, prediction) in enumerate(zip(all_images, scene_preds.values())):
            img_name = osp.basename(image_path)
            bboxes = prediction["bbox"]
            labels = prediction["labels"]
            # Store the results in the template
            bbox_template[i, :len(bboxes)] = bboxes
            label_template[i, :len(labels)] = labels
        # Filter out large bounding boxes
        bbox_template, label_template = self.filtering_predict_mask(bbox_template, label_template, threshold=0.5)

        if save:
            for i, (image_path, prediction) in enumerate(zip(all_images, scene_preds.values())):
                img_name = osp.basename(image_path)
                if not os.path.exists(f"{self.Save_path}/{scene}"): os.makedirs(f"{self.Save_path}/{scene}")
                plot_bounding_boxes(image_path, bbox_template[i], [self.texts[label][0] for label in label_template[i]], f"{self.Save_path}/{scene}/{img_name}")
            
        return bbox_template, label_template

    def inference_detector(self, images_batch):
        
        if self.resolution is None:
            self.resolution = get_image_resolution(images_batch[0])
        self.width, self.height = self.resolution
        inputs = []
        data_samples = []
        for img_id, image_path in enumerate(images_batch):
            data_info = dict(img_id=img_id, img_path=image_path, texts=self.texts)
            data_info = self.runner.pipeline(data_info)
            inputs.append(data_info['inputs'])
            data_samples.append(data_info['data_samples'])

        data_batch = dict(inputs=torch.stack(inputs),
                        data_samples=data_samples)
        
        with autocast(enabled=self.use_amp), torch.no_grad():
            output = self.runner.model.test_step(data_batch)
        frame_prediction = {}

        for img_id, image_path in enumerate(images_batch):
            with autocast(enabled=self.use_amp), torch.no_grad():
                pred_instances = output[img_id].pred_instances
            keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=self.nms)
            pred_instances = pred_instances[keep]
            pred_instances = pred_instances[pred_instances.scores.float() > self.th]
            
            if len(pred_instances.scores) > self.topk:
                indices = pred_instances.scores.float().topk(self.topk)[1]
                pred_instances = pred_instances[indices]
            mask = ~(((pred_instances['bboxes'][:,2]-pred_instances['bboxes'][:,0] > self.resolution[0]-50)*(pred_instances['bboxes'][:,3]-pred_instances['bboxes'][:,1] > self.resolution[1]-50)) == 1)
            bboxes_ = pred_instances['bboxes'][mask].cpu()
            labels_ = pred_instances['labels'][mask].cpu()
            scores_ = pred_instances['scores'][mask].cpu()
            frame_id = osp.basename(image_path).split(".")[0] 
            frame_prediction.update({frame_id:{"bbox":bboxes_, "labels":labels_, "scores":scores_}})
        
        return frame_prediction

    def filtering_predict_mask(self, pred_bbox, pred_label, threshold=0.5):
        """
        Filter out bounding boxes that are too large and cover the whole image.
        
        Parameters:

        pred_bbox (np.array): Array of shape [num_imgs, num_masks, 4] containing bounding boxes.
        pred_label (np.array): Array of shape [num_imgs, num_masks] containing labels.
        width (int): Width of the image.
        height (int): Height of the image.
        threshold (float): Threshold to filter out large bounding boxes.

        Returns:
        filtered_bbox (np.array): Filtered bounding boxes.
        filtered_label (np.array): Filtered labels.
        """

        area = ((pred_bbox[:, :, 2] - pred_bbox[:, :, 0]) *
                (pred_bbox[:, :, 3] - pred_bbox[:, :, 1])) / (self.width * self.height)

        valid_mask = area < threshold

        # Apply the mask to select the valid bounding boxes and labels
        filtered_bbox = torch.where(valid_mask[:, :, None], pred_bbox, torch.tensor(-1.0)).cuda()
        filtered_label = torch.where(valid_mask, pred_label, torch.tensor(-1)).cuda()

        return filtered_bbox, filtered_label

def main():

    vocab, _ = get_label_and_ids("Replica")
    Save_path = "Lookup_dict_rgb_new/"
    Snap_path = "/home/zelda/zh340/myzone/OpenIns3D_final_github/OpenIns3D/example_snap/"

    YOLOWORLD_scene = YOLOWORLD(Snap_path, Save_path, vocab)
    scene = "scannet_scene_mesh"
    bbox, label = YOLOWORLD_scene.build_lookup_dict(scene, save = True)

    print(bbox.shape, label.shape)
    
if __name__ == "__main__":
    main()



