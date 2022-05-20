python3 learn/test.py \
    --save_dir=$HOME/disk/train_ee \
    --dataset_root=$HOME/disk/dataset_dr \
    --task_name=pick_and_place_test \
    --T=40 --simulation_timestep=0.1 \
    --state_size=9 --action_size=7 \
    --randomize=True \
    --iteration=75000
