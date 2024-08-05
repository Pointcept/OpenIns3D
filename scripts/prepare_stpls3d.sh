# !/bin/bash

# download from stpls3d

mkdir -p ./data/stpls3d
gdown 1yrNxJ1cVEpwcURZsRGHQO_aZqGYeZGOZ -O data/stpls3d/stpls3d.zip
unzip data/stpls3d/stpls3d.zip -d data/stpls3d
rm data/stpls3d/stpls3d.zip