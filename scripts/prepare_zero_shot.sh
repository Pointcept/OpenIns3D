
gdown 1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B -O third_party/scannet200_val.ckpt

mkdir -p ./data/demo_scenes

gdown 1h5zJ7X5iVAa-aM9R5wBnl5lpTdi5yy99 -O ./data/demo_scenes/demo_scenes.zip
unzip ./data/demo_scenes/demo_scenes.zip -d ./data/demo_scenes
rm ./data/demo_scenes/demo_scenes.zip
