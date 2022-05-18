python3 learn/train.py \
    --save_dir=$HOME/disk/train_noMDN_gripper \
    --dataset_root=$HOME/disk/dataset_new \
    --task_name=pick_and_place \
    --T=40 --simulation_timestep=0.1 \
    --state_size=9 --action_size=7 \
    --gripper_action=True
