python3 make_dataset.py \
    --save_path=$HOME/disk/dataset/ --tasks=pick_and_place \
    --processes=8 --variations=500 --episodes_per_task=5
sleep 10
python3 make_dataset.py \
    --save_path=$HOME/disk/dataset/ --tasks=pick_and_place_test \
    --processes=8 --variations=100 --episodes_per_task=5

# for debug
# python3 make_dataset.py \
#     --save_path=$HOME/disk/dataset/ --tasks=pick_and_place \
#     --processes=2 --variations=2 --episodes_per_task=3
# sleep 10
# python3 make_dataset.py \
#     --save_path=$HOME/disk/dataset/ --tasks=pick_and_place_test \
#     --processes=2 --variations=2 --episodes_per_task=3
