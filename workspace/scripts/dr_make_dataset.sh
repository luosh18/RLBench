python3 make_dataset.py \
    --save_path=$HOME/disk/dataset_dr/ --tasks=pick_and_place \
    --processes=8 --variations=500 --episodes_per_task=5 \
    --simulation_timestep=0.1 \
    --randomize=True
sleep 10
python3 make_dataset.py \
    --save_path=$HOME/disk/dataset_dr/ --tasks=pick_and_place_test \
    --processes=8 --variations=100 --episodes_per_task=5 \
    --simulation_timestep=0.1 \
    --randomize=True