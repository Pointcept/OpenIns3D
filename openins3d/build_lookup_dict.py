"""
Build Lookup Dict Script for OpenIns3D

Author: Zhening Huang (zh340@cam.ac.uk)
"""

import sys
sys.path.append("/home/zelda/zh340/myzone/lookup/ODISE")

import glob
import os
import time
from contextlib import ExitStack
import json
import torch
import tqdm
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


def load_odise(vocab):
    config_file = "ODISE/configs/Panoptic/odise_label_coco_50e.py"
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

def build_lookup_dict(vocab, scene_id, image_PATH, SAVED_PATH):
    
    inference_model, demo_metadata, aug = load_odise(vocab)

    with ExitStack() as stack:
        if isinstance(inference_model, nn.Module):
            stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())
        demo = VisualizationDemo(inference_model, demo_metadata, aug)
        get_scene_id = glob.glob(image_PATH + "/*")
        scene_id_list = sorted([scene.split("/")[-1] for scene in get_scene_id])

        # print("scene_id_list", scene_id_list)

        for scene in tqdm.tqdm(scene_id_list):
            print(f"-------------start to process {scene}--------------")
            label_folder = f"{SAVED_PATH}/{scene}/label/"
            map_folder = f"{SAVED_PATH}/{scene}/map/"
            image_folder = f"{SAVED_PATH}/{scene}/image/"
            
            if not os.path.exists(label_folder):os.makedirs(label_folder)
            if not os.path.exists(map_folder):os.makedirs(map_folder)
            if not os.path.exists(image_folder):os.makedirs(image_folder)
                    
            input = f"{image_PATH}/{scene}/image/*"
            input = glob.glob(input)
            for path in tqdm.tqdm(input):
                img = utils.read_image(path, format="RGB")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                torch.save(panoptic_seg.to_sparse(), f"{map_folder}/{os.path.basename(path).split('.')[0]}.pt")
                with open(f"{label_folder}/{os.path.basename(path).split('.')[0]}", 'w') as fout:
                    json.dump(segments_info, fout)
                out_filename = os.path.join(image_folder, os.path.basename(path))
                visualized_output.save(out_filename)

def load_2d_model(vocab):
    inference_model, demo_metadata, aug = load_odise(vocab)
    return inference_model, demo_metadata, aug

def build_lookup_dict_one_scene(odise_model, scene_id, image_PATH, SAVED_PATH):
    
    inference_model, demo_metadata, aug = odise_model

    with ExitStack() as stack:
        if isinstance(inference_model, nn.Module):
            stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())
        demo = VisualizationDemo(inference_model, demo_metadata, aug)

        scene = scene_id
        label_folder = f"{SAVED_PATH}/{scene}/label/"
        map_folder = f"{SAVED_PATH}/{scene}/map/"
        image_folder = f"{SAVED_PATH}/{scene}/image/"
        
        if not os.path.exists(label_folder):os.makedirs(label_folder)
        if not os.path.exists(map_folder):os.makedirs(map_folder)
        if not os.path.exists(image_folder):os.makedirs(image_folder)
                
        input = f"{image_PATH}/{scene}/image/*"
        input = glob.glob(input)
        for path in tqdm.tqdm(input):
            img = utils.read_image(path, format="RGB")
            predictions, visualized_output = demo.run_on_image(img)
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            torch.save(panoptic_seg.to_sparse(), f"{map_folder}/{os.path.basename(path).split('.')[0]}.pt")
            with open(f"{label_folder}/{os.path.basename(path).split('.')[0]}", 'w') as fout:
                json.dump(segments_info, fout)
            out_filename = os.path.join(image_folder, os.path.basename(path))
            visualized_output.save(out_filename)

if __name__ == "__main__":
    vocab = "cabinet; bed; chair; sofa; table; door; window; bookshelf; picture; counter; desk; curtain; refrigerator; showercurtain; toilet; sink; bathtub"
    image_PATH = "/home/zelda/zh340/myzone/lookup/scannet_2dimages/"
    SAVED_PATH = "Lookup_dict_rgb/"
    snap_save = "/home/zelda/zh340/myzone/lookup/export/Snap"
    Lookup_save = "Lookup_dict_new"
    final_results = "save_results_new"
    build_lookup_dict(vocab, snap_save, Lookup_save)