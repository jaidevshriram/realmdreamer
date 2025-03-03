cd nerfstudio
pip install -e .
cd ../
pip install -e pcd_generator/depth_estimators/extern/depth_anything
conda install pytorch=2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
cd realmdreamer
pip install -e .
cd ../
pip install ./realmdreamer/realmdreamer/gaussian_splatting/simple-knn
pip install ./realmdreamer/realmdreamer/gaussian_splatting/occlude
pip install ./realmdreamer/diff-gaussian-rasterization
pip uninstall -y tinycudann
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.7/download/linux-64/pytorch3d-0.7.7-py39_cu118_pyt201.tar.bz2
conda install ./pytorch3d-0.7.7-py39_cu118_pyt201.tar.bz2

wget --content-disposition -P ./pcd_generator/depth_estimators/checkpoints "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"; wget --content-disposition -P ./pcd_generator/depth_estimators/checkpoints "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt"
