# download the pretrained model and config file for yoloworld detector

mkdir -p third_party/pretrained
gdown 1UF1vi19pwTvp8CT9sJ7Bg26aErvL-OgZ -O third_party/pretrained/yoloworld.zip
unzip third_party/pretrained/yoloworld.zip -d third_party/pretrained
rm third_party/pretrained/yoloworld.zip

echo "Download complete."