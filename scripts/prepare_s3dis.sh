# !/bin/bash

# download from s3dis

mkdir -p ./data/s3dis
gdown 1FLr2VMTYpxW5Xrl5iaTt1uWA9AvN2zK0 -O data/s3dis/s3dis.zip
unzip data/s3dis/s3dis.zip -d data/s3dis
rm data/s3dis/s3dis.zip