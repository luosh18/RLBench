python3 jaco_dataset_generator.py \
    --save_path=$HOME/disk/rlbench_data/ --tasks=pick_and_place \
    --processes=8 --episodes_per_task=10
python3 preprocess_dataset.py \
    --save_path=$HOME/disk/preprocessed_data/ \
    --data_path=$HOME/disk/rlbench_data/ 
