uv run train.py \
    --world 1 \
    --stage 1 \
    --lr 1e-4 \
    --action_type jump \
    --save_interval 16

uv run train.py \
    --world 1 \
    --stage 2 \
    --lr 1e-4 \
    --action_type jump
