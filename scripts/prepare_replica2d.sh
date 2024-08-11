# !/bin/bash

# download from scannet

mkdir -p ./data/replica
gdown 1j2rwrnAehniQHyfnW7eVqenG8TiqSWwW -O data/replica/2d.zip
unzip data/replica/2d.zip -d data/replica
rm data/replica/2d.zip