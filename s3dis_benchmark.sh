#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export PYTHONPATH="openins3d"

CURR_AREA=5  # please set this to Area 5

# please provide pathes for the following:
S3DIS_PROCESSED_DIR="~/OpenIns3d/data/processed/s3dis" # copy the full path for the processed
MPM_CHECKPOINT="checkpoints/download_models/s3dis_area5.ckpt"

# Corresponding paths; no need to change unless preprocessing was not followed completely
PROJECT_SAVE_FOLDER="s3dis_saved"
FINAL_RESULT_FOLDER="s3dis_results"
MASK_SAVE_DIR="${PROJECT_SAVE_FOLDER%/}/s3dis_masks_sparse"
S3DIS_LABEL_DB_PATH="${S3DIS_PROCESSED_DIR%/}/label_database.yaml"
S3DIS_COLOR_MEAN_STD_PATH="${S3DIS_PROCESSED_DIR%/}/color_mean_std.yaml"
S3DIS_GT_PATH="${S3DIS_PROCESSED_DIR%/}/instance_gt/Area_${CURR_AREA}"

# OpenIns3D start

# step 1: obtained Class-agnostic mask from Mask proposal module
echo "[******OpenIns3D INFO*******] Mask Proposal"

python openins3d/mask3d/get_s3dis.py \
    general.checkpoint=${MPM_CHECKPOINT} \
    data/datasets=s3dis \
    data.num_labels=13 \
    general.area=${CURR_AREA}\
    model.num_queries=100 \
    general.use_dbscan=true \
    data.test_dataset.data_dir=${S3DIS_PROCESSED_DIR}  \
    data.validation_dataset.data_dir=${S3DIS_PROCESSED_DIR} \
    data.train_dataset.data_dir=${S3DIS_PROCESSED_DIR} \
    data.train_dataset.label_db_filepath=${S3DIS_LABEL_DB_PATH} \
    data.validation_dataset.label_db_filepath=${S3DIS_LABEL_DB_PATH} \
    data.test_dataset.label_db_filepath=${S3DIS_LABEL_DB_PATH}  \
    general.mask_save_dir=${MASK_SAVE_DIR} 

# echo "[******OpenIns3D INFO*******] Mask Proposal is done and class-agonstic masks are saved"

# # step 2: obtain and save mask classfication with Snap & Lookup module
# echo "[******OpenIns3D INFO*******] Snap and Lookup"
# python inference_openins3d.py \
#     --processed_scene "${S3DIS_PROCESSED_DIR%/}/Area_${CURR_AREA}" \
#     --img_size 1000 \
#     --result_save ${FINAL_RESULT_FOLDER} \
#     --byproduct_save ${PROJECT_SAVE_FOLDER} \
#     --ca_mask_path ${MASK_SAVE_DIR} \
#     --save_results_in_2d false \
#     --dataset s3dis
# echo "[******OpenIns3D INFO*******] Snap and Lookup is done and detection results are saved"

# # step 3: evluate the results
# python evaluate.py \
#     --result_save ${FINAL_RESULT_FOLDER} \
#     --gt_path ${S3DIS_GT_PATH} \
#     --dataset s3dis

# echo "[******OpenIns3D INFO*******] ALL Finished"
