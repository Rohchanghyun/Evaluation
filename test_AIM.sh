# For LTCC dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --root /workspace/dataset --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 --eval --resume ../checkpoint-60000/model.pt #
# For PRCC dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --root /workspace/dataset --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 --eval --resume ../checkpoint-60000/model.pt #