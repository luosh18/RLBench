python3 learn/train.py \
    --save_dir=$HOME/disk/train_ee_dr \
    --dataset_root=$HOME/disk/dataset_dr \
    --task_name=pick_and_place \
    --T=40 --simulation_timestep=0.1 \
    --state_size=3 --action_size=4 \
    --randomize=True --iteration=150000