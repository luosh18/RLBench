python3 learn/test.py \
    --save_dir=$HOME/disk/train_noMDN \
    --dataset_root=$HOME/disk/dataset \
    --task_name=pick_and_place_test \
    --T=40 --simulation_timestep=0.1 \
    --state_size=3 --action_size=4 \
    --iteration=33500
