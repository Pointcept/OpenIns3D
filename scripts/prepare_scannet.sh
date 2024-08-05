#!/bin/bash
mkdir -p ./data/scannet
SCENES_FILE="scripts/scannetv2_val.txt"

if [ ! -f "$SCENES_FILE" ]; then
  echo "Error: The file $SCENES_FILE does not exist."
  exit 1
fi

echo "Starting to download all _vh_clean_2.ply files for the ScanNet validation set."

total_lines=$(wc -l < "$SCENES_FILE")
current_line=0
while IFS= read -r scene_name; do
    yes '' | python scripts/download-scannet.py --type _vh_clean_2.ply -o ./data/scannet --id "$scene_name" > /dev/null 2>&1
    current_line=$(expr $current_line + 1)
    progress=$(echo "scale=2; $current_line/$total_lines*100" | bc)
    printf "\rProgress: %.2f%% (%d/%d)" "$progress" "$current_line" "$total_lines"
done < "$SCENES_FILE"

echo -e "\nDownload complete."

SRC_BASE_DIR="data/scannet/scans"
TGT_DIR="data/scannet/scenes"
mkdir -p "$TGT_DIR"
find "$SRC_BASE_DIR" -type f -name "*.ply" -exec mv {} "$TGT_DIR" \;
for file in "$TGT_DIR"/*.ply; do
    base_name=$(basename "$file" .ply)
    new_name=$(echo "$base_name" | sed 's/_vh_clean_2//')
    mv "$file" "$TGT_DIR/$new_name.ply"
done

echo "All .ply files have been moved and renamed in $TGT_DIR"

rm -r "$SRC_BASE_DIR"

echo "Starting to download the ScanNet ground truth and ScanNet masks files."

gdown 1NfnNBTvapKHi_ZrIrBiKrW8VSKzYc-V_ -O data/scannet/scannet.zip
unzip data/scannet/scannet.zip -d data/scannet

rm data/scannet/scannet.zip

echo "Data preparation complete."