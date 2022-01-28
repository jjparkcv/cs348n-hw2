# hw2-master

Install the pip requirements in requirements.txt

download weights.pth 
download data/

For running the codes for the homework:

python -i single_scene.py --epoch 2000 --name deepsdf --gpu 0 --trunc 0.1 --lr 2e-4

python train.py --input data/bunny.npy --epochs 2000

python marching_cubes.py --name output

python -i decoder_deepsdf.py --epoch 2000 --batch_size 16 --num_scenes 100 --name deepsdf --gpu 0 --trunc 0.1 --dim_latent 256 --lr 1e-3 --lr_z 2e-3

python -i shape_completion_deepsdf.py --load_model weights.pth
