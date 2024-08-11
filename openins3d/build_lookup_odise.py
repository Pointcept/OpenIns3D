"""
Script for Building Lookup Dict with ODISE for OpenIns3D
Author: Zhening Huang (zh340@cam.ac.uk)

reference: 
ODISE: https://github.com/NVlabs/ODISE
"""

import sys
sys.path.append("third_party/ODISE")

import glob
import os
from contextlib import ExitStack
import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.engine import create_ddp_model
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from torch import nn
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.engine.defaults import get_model_from_module
from demo.demo import VisualizationDemo
from utils import get_label_and_ids, get_image_resolution

class ODISE:
    """
    to called ODISE for mask understanding
    """

    def __init__(self, Snap_path, Save_path, vocab, ODISE_PATH = "./third_party/ODISE/"):
        """
        vocab: str, the vocabularies for the ODISE model, a list of strings, e.g. ["cabinet", "bed", "chair", ...]
        Snap_path: str, the path to the Snap
        Save_path: str, the path to save the lookup image for visulization, if necessary.
        """
        self.vocab = "; ".join(vocab)
        self.ODISE_folder = ODISE_PATH
        # call the ODISE model, this could take a while to initialize
        self.inference_model, self.demo_metadata, self.aug = self.load_odise(self.vocab)
        self.Snap_path = Snap_path # path to save the odise_results for debugging
        self.Save_path = Save_path # path to save the lookup image for visulization, if necessary.

    def load_odise(self, vocab):

        """
        taken from ODISE orginal codebabse: https://github.com/NVlabs/ODISE
        """
        config_file = f"{self.ODISE_folder}/configs/Panoptic/odise_label_coco_50e.py"
        init_from = "odise://Panoptic/odise_label_coco_50e"

        cfg = LazyConfig.load(config_file)
        cfg.model.overlap_threshold = 0
        cfg.model.clip_head.alpha = 0.35
        cfg.model.clip_head.beta = 0.65
        seed_all_rng(42)

        dataset_cfg = cfg.dataloader.test
        wrapper_cfg = cfg.dataloader.wrapper  
        extra_classes = []
        if vocab:
            for words in vocab.split(";"):
                extra_classes.append([word.strip() for word in words.split(",")])

        extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

        demo_thing_classes = extra_classes
        demo_stuff_classes = []
        demo_thing_colors = extra_colors
        demo_stuff_colors = []

        demo_metadata = MetadataCatalog.get("odise_demo_metadata")
        demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
        demo_metadata.stuff_classes = [
            *demo_metadata.thing_classes,
            *[c[0] for c in demo_stuff_classes],
        ]
        demo_metadata.thing_colors = demo_thing_colors
        demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
        demo_metadata.stuff_dataset_id_to_contiguous_id = {
            idx: idx for idx in range(len(demo_metadata.stuff_classes))
        }
        demo_metadata.thing_dataset_id_to_contiguous_id = {
            idx: idx for idx in range(len(demo_metadata.thing_classes))
        }

        wrapper_cfg.labels = demo_thing_classes + demo_stuff_classes
        wrapper_cfg.metadata = demo_metadata

        aug = instantiate(dataset_cfg.mapper).augmentations

        model = instantiate_odise(cfg.model)
        model.to(cfg.train.device)
        ODISECheckpointer(model).load(init_from)

        # look for the last wrapper
        while "model" in wrapper_cfg:
            wrapper_cfg = wrapper_cfg.model
        wrapper_cfg.model = get_model_from_module(model)

        inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
        return inference_model, demo_metadata, aug

    def build_lookup_dict(self, scene, save = False):

        num_images = len(glob.glob(f"{self.Snap_path}/{scene}/image/*"))
        width, heigh = get_image_resolution(glob.glob(f"{self.Snap_path}/{scene}/image/*")[0])

        panoptic_seg_list = torch.zeros((num_images, heigh, width), dtype=torch.int64)
        labels = torch.ones((num_images, 80), dtype=torch.int64) * -1
        
        with ExitStack() as stack:
            if isinstance(self.inference_model, nn.Module):
                stack.enter_context(inference_context(self.inference_model))
            stack.enter_context(torch.no_grad())
            demo = VisualizationDemo(self.inference_model, self.demo_metadata, self.aug)
            print(f"*****************Start to Build Lookup Dict for {scene}*****************")

            for i in range(num_images):
                sys.stdout.write(f"\rLookup module: {i} out of {num_images} images are processed by ODISE")
                sys.stdout.flush()
                path = f"{self.Snap_path}/{scene}/image/image_rendered_angle_{i}.png"
                img = utils.read_image(path, format="RGB")
                predictions, visualized_output = demo.run_on_image(img)
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                panoptic_seg_list[i, :, :] = panoptic_seg
                labels[i, :len(segments_info)] = torch.tensor([seg["category_id"] for seg in segments_info])
                if save:
                    if not os.path.exists(f"{self.Save_path}/{scene}"): os.makedirs(f"{self.Save_path}/{scene}")
                    out_filename = os.path.join(f"{self.Save_path}/{scene}", os.path.basename(path))
                    visualized_output.save(out_filename)
        return panoptic_seg_list, labels

def main():

    vocab, _ = get_label_and_ids("Replica")
    Save_path = "Lookup_dict_rgb/"
    Snap_path = "/home/zelda/zh340/myzone/OpenIns3D_final_github/OpenIns3D/example_snap/"

    ODISE_scene = ODISE(Snap_path, Save_path, vocab)
    scene = "scannet_scene_mesh"
    panoptic_seg_list, labels = ODISE_scene.build_lookup_dict(scene, save = True)

if __name__ == "__main__":
    main()

