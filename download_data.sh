# download visual7w telling
cd visual7w-toolkit/datasets/visual7w-telling/
./download_dataset.sh
cd ../../..

# download visual7w pointing
cd visual7w-toolkit/datasets/visual7w-pointing/
./download_dataset.sh
cd ../../..

wget http://vision.stanford.edu/yukezhu/visual7w_images.zip
unzip visual7w_images.zip
rm visual7w_images.zip

mkdir -p cnn_models
cd cnn_models
wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
cd ..
