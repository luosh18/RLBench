python3 learn/test.py \
    --save_dir=$HOME/disk/train_ee_gripper \
    --dataset_root=$HOME/disk/dataset_new \
    --task_name=pick_and_place_test \
    --T=40 --simulation_timestep=0.1 \
    --state_size=3 --action_size=4 \
    --gripper_action=True \
    --iteration=75000
