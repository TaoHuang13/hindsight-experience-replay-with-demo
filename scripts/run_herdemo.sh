export CUDA_VISIBLE_DEVICES=0

for seed in 1 2 3
do
    python train.py seed=$seed task_name=NeedlePick-v0
done