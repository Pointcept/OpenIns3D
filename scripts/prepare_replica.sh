# !/bin/bash

# download from scannet

mkdir -p ./data/replica
gdown 19AfS2eRDH-BmF9lo4LB0xg2PCTUsguKO -O data/replica/replica.zip
unzip data/replica/replica.zip -d data/replica
rm data/replica/replica.zip