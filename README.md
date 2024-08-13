
<p align="center">

  <h1 align="center"> OpenIns3D: Snap and Lookup for 3D Open-vocabulary Instance Segmentation</h1>
  <p align="center">
    <a href="https://zheninghuang.github.io/"><strong>Zhening Huang</strong></a>
    ·
    <a href="https://xywu.me"><strong>Xiaoyang Wu</strong></a>
    ·
    <a href="https://xavierchen34.github.io/"><strong>Xi Chen</strong></a>
    ·
    <a href="https://hszhao.github.io"><strong>Hengshuang Zhao</strong></a>
    ·
    <a href="https://sites.google.com/site/indexlzhu/home"><strong>Lei Zhu</strong></a>
    ·
    <a href="http://sigproc.eng.cam.ac.uk/Main/JL"><strong>Joan Lasenby</strong></a>
  </p>
  
  <h3 align="center"><a href="https://arxiv.org/abs/2309.00616">Paper</a> | <a href="https://www.youtube.com/watch?v=kwlMJkEfTyY">Video</a> | <a href="https://zheninghuang.github.io/OpenIns3D/">Project Page</a></h3>
  <div align="center"></div>
</p>

   
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/openins3d-snap-and-lookup-for-3d-open/zero-shot-3d-point-cloud-classification-on-1)](https://paperswithcode.com/sota/zero-shot-3d-point-cloud-classification-on-1?p=openins3d-snap-and-lookup-for-3d-open) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/openins3d-snap-and-lookup-for-3d-open/3d-open-vocabulary-instance-segmentation-on-3)](https://paperswithcode.com/sota/3d-open-vocabulary-instance-segmentation-on-3?p=openins3d-snap-and-lookup-for-3d-open)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/openins3d-snap-and-lookup-for-3d-open/3d-open-vocabulary-object-detection-on-1)](https://paperswithcode.com/sota/3d-open-vocabulary-object-detection-on-1?p=openins3d-snap-and-lookup-for-3d-open)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/openins3d-snap-and-lookup-for-3d-open/3d-open-vocabulary-instance-segmentation-on-1)](https://paperswithcode.com/sota/3d-open-vocabulary-instance-segmentation-on-1?p=openins3d-snap-and-lookup-for-3d-open)
<p align="center">
<strong> TL;DR: OpenIns3D proposes a "mask-snap-lookup" scheme to achieve 2D-input-free 3D open-world scene understanding, which attains SOTA performance across datasets, even with fewer input prerequisites. 🚀✨
</p>


<table>
<tr>
    <td><img src="assets/demo_1.gif" width="100%"/></td>
    <td><img src="assets/demo_2.gif" width="100%"/></td>
    <td><img src="assets/demo_3.gif" width="100%"/></td>
</tr>
<tr>
    <td align='center' width='24%'>device to watch BBC news</td>
    <td align='center' width='24%'>furniture that is capable of producing music</td>
    <td align='center' width='24%'>Ma Long's domain of excellence</td>
<tr>
<tr>
    <td><img src="assets/demo_4.gif" width="100%"/></td>
    <td><img src="assets/demo_5.gif" width="100%"/></td>
    <td><img src="assets/demo_6.gif" width="100%"/></td>
</tr>
<tr>
    <td align='center' width='24%'>most comfortable area to sit in the room</td>
    <td align='center' width='24%'>penciling down ideas during brainstorming</td>
    <td align='center' width='24%'>furniture offers recreational enjoyment with friends</td>
<tr>
</table>


<br>

<!-- # OpenIns3D pipeline

<img src="assets/general_pipeline_updated.png" width="100%"/> -->


# Highlights
- *2 Aug, 2024*: Major update 🔥: We have released optimized and easy-to-use code for OpenIns3D to [reproduce all the results in the paper](#Reproducing-Results) and [demo](#Zero-Shot-Inference-with-Single-Vocabulary).
- *1 Jul, 2024*: OpenIns3D has been accepted at ECCV 2024 🎉. We will release more code on various experiments soon.
- *6 Jan, 2024*: We have released a major revision, incorporating S3DIS and ScanNet benchmark code. Try out the latest version.
- *31 Dec, 2023* We release the batch inference code on ScanNet.
- *31 Dec, 2023* We release the zero-shot inference code， test it on your own data!
- *Sep, 2023*: **OpenIns3D** is released on [arXiv](https://arxiv.org/abs/2309.00616), alongside with [explanatory video](https://www.youtube.com/watch?v=kwlMJkEfTyY), [project page](https://zheninghuang.github.io/OpenIns3D/). We will release the code at end of this year.

# Overview

- [Installation](#installation)
- [Reproducing All Benchmarks Results](#reproducing-results)
- [Replacing Snap with RGBD](#Replacing-Snap-with-RGBD)
- [Zero-Shot Inference with Single Vocabulary](#Zero-Shot-Inference-with-Single-Vocabulary)
- [Zero-Shot Inference with Multiple Vocabulary](#Zero-Shot-Inference-with-Multiple-Vocabulary)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

# Installation

Please check the [installation file](installation.md) to install OpenIns3D for:
1. [reproducing all results in the paper](#reproducing-results),
2. [testing on your own dataset](#Zero-Shot-Inference-with-Multiple-Vocabulary)

---

# Reproducing Results

### 🗂️ Replica

**🔧 Data Preparation**: 
1. Execute the following command to set up the Replica dataset, including scene `.ply` files, predicted masks, and ground truth:
```sh
sh scripts/prepare_replica.sh
```

**📊 Open Vocabulary Instance Segmentation**:
```sh
python openins3d/main.py --dataset replica --task OVIS --detector yoloworld
```
**📈 Results Log**: 
| Task                        |  AP  | AP50 | AP25 | Log |
|-----------------------------|:----:|:----:|:----:|:----:|
| Replica OVIS (in paper)      | 13.6 | 18.0 | 19.7 |      |
| Replica OVIS (this Code)     | 15.4 | 19.5 | 25.2 | [log](assets/logs/log_replica_ovis.txt)  |
---

### 🗂️ ScanNet

**🔧 Data Preparation**: 
1. Make sure you have completed the form on [ScanNet](http://www.scan-net.org/) to obtain access.
2. Place the `download-scannet.py` script into the `scripts` directory.
3. Run the following command to download all `_vh_clean_2.ply` files for validation sets, as well as instance ground truth, GT-masks, and detected masks:

```sh
sh scripts/prepare_scannet.sh
```

**📊 Open Vocabulary Object Recognition**: 
```sh
python openins3d/main.py --dataset scannet --task OVOR --detector odise
```

**📈 Results Log**: 
| Task                        | Top-1 Accuracy | Log |
|-----------------------------|:--------------:|:----:|
| ScanNet_OVOR (in paper)      |     60.4       |      |
| ScanNet_OVOR (this Code)     |     64.2       | [log](assets/logs/log_scannet_classfication.txt)  |

**📊 Open Vocabulary Object Detection**:
```sh
python openins3d/main.py --dataset scannet --task OVOD --detector odise
```

**📊 Open Vocabulary Instance Segmentation**:
```sh
python openins3d/main.py --dataset scannet --task OVIS --detector odise
```
**📈 Results Log**: 
| Task                        |  AP  | AP50 | AP25 | Log |
|-----------------------------|:----:|:----:|:----:|:----:|
| ScanNet_OVOD (in paper)      | 17.8 | 28.3 | 36.0 |      |
| ScanNet_OVOD (this Code)     | 20.7 | 29.9 | 39.7 | [log](assets/logs/log_scannet_ovod.txt)  |
| ScanNet_OVIS (in paper)      | 19.9 | 28.7 | 38.9 |      |
| ScanNet_OVIS (this Code)     | 23.3 | 34.6 | 42.6 | [log](assets/logs/log_scannet_ovis.txt)  |

---

### 🗂️ S3DIS

**🔧 Data Preparation**: 
1. Make sure you have completed the form on [S3DIS](https://redivis.com/datasets/9q3m-9w5pa1a2h/files) to obtain access. 
2. Then, run the following command to acquire scene `.ply` files, predicted masks, and ground truth:
```sh
sh scripts/prepare_s3dis.sh
```

**📊 Open Vocabulary Instance Segmentation**:
```sh
python openins3d/main.py --dataset s3dis --task OVIS --detector odise
```
**📈 Results Log**: 
| Task                        |  AP  | AP50 | AP25 | Log |
|-----------------------------|:----:|:----:|:----:|:----:|
| S3DIS OVIS (in paper)        | 21.1 | 28.3 | 29.5 |      |
| S3DIS OVIS (this Code)       | 22.9 | 29.0 | 31.4 | [log](assets/logs/log_s3dis_ovis.txt)  |

---

### 🗂️ STPLS3D

**🔧 Data Preparation**: 
1. Make sure you have completed the form [STPLS3D](https://www.stpls3d.com/data) to gain access. 
2. Then, run the following command to obtain scene `.ply` files, predicted masks, and ground truth:
```sh
sh scripts/prepare_stpls3d.sh
```

**📊 Open Vocabulary Instance Segmentation**:
```sh
python openins3d/main.py --dataset stpls3d --task OVIS --detector odise
```
**📈 Results Log**: 
| Task                        |   AP   | AP50  | AP25  | Log |
|-----------------------------|:------:|:-----:|:-----:|:----:|
| STPLS3D OVIS (in paper)      | 11.4  | 14.2 | 17.2 |      |
| STPLS3D OVIS (this Code)     |  15.3      | 17.3   | 17.4      | [log](assets/logs/log_stpls3d_ovis.txt)  |

---

# Replacing Snap with RGBD

We also evaluate the performance of OpenIns3D when the Snap module is replaced with original RGBD images while keeping the other design intact.

### 🗂️ Replica

**🔧 Data Preparation**  
1. Download the Replica dataset and RGBD images:

```sh
sh scripts/prepare_replica.sh
sh scripts/prepare_replica2d.sh
```

**📊 Open Vocabulary Instance Segmentation**

```sh
python openins3d/main.py --dataset replica --task OVIS --detector yoloworld --use_2d true
python openins3d/main.py --dataset replica --task OVIS --detector yoloworld --use_2d true
python openins3d/main.py --dataset scannet200 --task OVIS --detector yoloworld --use_2d true
```

**📈 Results Log**  
| Task           |  AP  | AP50 | AP25 | Log                                      |
|----------------|:----:|:----:|:----:|:----------------------------------------:|
| OpenMask3D     | 13.1 | 18.4 | 24.2 |                                          |
| Open3DIS       | 18.5 | 24.5 | 28.2 |                                          |
| OpenIns3D      | 21.1 | 26.2 | 30.6 | [log](assets/logs/log_replica_use2d_ovis.txt) |


# Zero-Shot Inference with Single Vocabulary

We demonstrate how to perform single-vocabulary instance segmentation similar to the teaser image in the paper. The key new feature is the introduction of a CLIP ranking and filtering module to reduce false-positive results. (Works best with RGBD but is also fine with SNAP.)

Quick Start: 

1. 📥 **Download the demo dataset** by running:

   ```sh
   sh scripts/prepare_demo_single.sh 
   ```

2. 🚀 **Run the model** by executing:

   ```sh
   python zero_shot_single_voc.py
   ```


You can now view results like teaser images in 2D or 3D.


---

# Zero-Shot Inference with Multiple Vocabulary

ℹ️ **Note**: Ensure you have installed the mask module according to the installation guide, as it is not required for reproducing results.

To perform zero-shot scene understanding:

1. 📥 **Download** the `scannet200_val.ckpt` checkpoint from [this link](https://drive.google.com/file/d/1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B/view) and place it in the `third_party/` directory.

2. 🚀 **Run the model** by executing `python zero_shot.py` and specify:
   - 🗂️ `pcd_path`: The path to the colored point cloud file.
   - 📝 `vocab`: A list of vocabulary terms to search for.

You can also use the following script to automatically set up the `scannet200_val.ckpt` checkpoint and download some sample 3D scans:

```bash
sh scripts/prepare_zero_shot.sh
```

### 🚀 Running a Zero-Shot Inference

To perform zero-shot inference using the sample dataset (default with Replica vocabulary), run:

```bash
python zero_shot_multi_vocs.py --pcd_path data/demo_scenes/demo_scene_1.ply
```

📂 **Results** are saved under `output/snap_demo/demo_scene_1_vis/image`.

To use a different 2D detector (🔍 **ODISE works better on pcd-rendered images**):

```bash
python zero_shot_multi_vocs.py --pcd_path data/demo_scenes/demo_scene_2.ply --detector yoloworld
```

📝 **Custom Vocabulary**: If you want to specify your own vocabulary list, add it with the `--vocab` flag as follows:

```bash
python zero_shot_multi_vocs.py \
--pcd_path 'data/demo_scenes/demo_scene_4.ply' \
--vocab "drawers" "lower table"
```


# Citation

If you find OpenIns3D and this codebase useful for your research, please cite our work as a form of encouragement. 😊

```
@article{huang2024openins3d,
      title={OpenIns3D: Snap and Lookup for 3D Open-vocabulary Instance Segmentation}, 
      author={Zhening Huang and Xiaoyang Wu and Xi Chen and Hengshuang Zhao and Lei Zhu and Joan Lasenby},
      journal={European Conference on Computer Vision},
      year={2024}
    }

```

# Acknowlegement

The mask proposal model is modified from [Mask3D](https://jonasschult.github.io/Mask3D/), and we heavily used the [easy setup](https://github.com/cvg/Mask3D) version of it for MPM. Thanks again for the great work! 🙌 We also drew inspiration from [LAR](https://github.com/eslambakr/LAR-Look-Around-and-Refer) and [ContrastiveSceneContexts](https://github.com/facebookresearch/ContrastiveSceneContexts) when developing the code. 🚀
