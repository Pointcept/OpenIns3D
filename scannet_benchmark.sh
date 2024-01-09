#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export PYTHONPATH="openins3d"

CURR_DBSCAN=0.95
CURR_QUERY=150

# please provide pathes for the following:
SCANNET_PROCESSED_DIR="data/processed/scannet" # copy the full path for the processed
MPM_CHECKPOINT='scannet_val.ckpt'
SCAN_PATH="data/scans"
# Corresponding paths; no need to change unless preprocessing was not followed completely
PROJECT_SAVE_FOLDER="scannet_saved"
FINAL_RESULT_FOLDER="scannet_results"
MASK_SAVE_DIR="${PROJECT_SAVE_FOLDER%/}/scannet_masks_sparse"
SCANNET_LABEL_DB_PATH="${SCANNET_PROCESSED_DIR%/}/label_database.yaml"
SCANNET_COLOR_MEAN_STD_PATH="${SCANNET_PROCESSED_DIR%/}/color_mean_std.yaml"
SCANNET_GT_PATH="${SCANNET_PROCESSED_DIR%/}/instance_gt/validation"

# OpenIns3D start

# step 1: obtained Class-agnostic mask from Mask proposal module
echo "[******OpenIns3D INFO*******] Mask Proposal"
python openins3d/mask3d/get_scannet.py \
    general.checkpoint=${MPM_CHECKPOINT}\
    data/datasets=scannet \
    data.num_labels=20 \
    general.area=${CURR_AREA}\
    model.num_queries=${CURR_QUERY} \
    general.use_dbscan=true \
    data.test_dataset.data_dir=${SCANNET_PROCESSED_DIR}  \
    data.validation_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
    data.train_dataset.data_dir=${SCANNET_PROCESSED_DIR} \
    data.train_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
    data.validation_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH} \
    data.test_dataset.label_db_filepath=${SCANNET_LABEL_DB_PATH}  \
    general.mask_save_dir=${MASK_SAVE_DIR} 

echo "[******OpenIns3D INFO*******] Mask Proposal is done and class-agonstic masks are saved"

# step 2: obtain and save mask classfication with Snap & Lookup module
echo "[******OpenIns3D INFO*******] Snap and Lookup"
python inference_openins3d.py \
    --processed_scene "${SCANNET_PROCESSED_DIR%/}/validation" \
    --img_size 1000 \
    --result_save ${FINAL_RESULT_FOLDER} \
    --byproduct_save ${PROJECT_SAVE_FOLDER} \
    --ca_mask_path ${MASK_SAVE_DIR} \
    --save_results_in_2d false \
    --scans_path ${SCAN_PATH} \
    --dataset scannet
echo "[******OpenIns3D INFO*******] Snap and Lookup is done and detection results are saved"

# step 3: evluate the results
python evaluate.py \
    --result_save ${FINAL_RESULT_FOLDER} \
    --gt_path ${SCANNET_GT_PATH} \
    --dataset scannet

echo "[******OpenIns3D INFO*******] ALL Finished"
