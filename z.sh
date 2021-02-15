python train.py models/best \
    --gpus=1 \
    --fast_dev_run 1 \
    --progress_bar_refresh_rate 1 \
    --gradient_clip_val 1 \
    --max_epochs 10 \
    --limit_train_batches 0.01 \
    --limit_val_batches 0.01 \
    --limit_test_batches 0.01 \
    --val_check_interval 0.2


python train.py models/best \
    --gpus=1 \
    --progress_bar_refresh_rate 20 \
    --gradient_clip_val 1 \
    --max_epochs 10 \
    --limit_train_batches 0.02 \
    --limit_val_batches 0.02 \
    --limit_test_batches 0.02 \
    --val_check_interval 0.2
