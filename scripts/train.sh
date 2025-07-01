SAVE_DIR="checkpoints/output"

torchrun --nproc_per_node=8 train.py \
    --train_dataset "5_000 @ Scannet(split='train', ROOT='data/scannet_processed', resolution=(224, 224)) + 5_000 @ Scannetpp(split='train', ROOT='data/scannetpp_processed', resolution=(224, 224))" \
    --test_dataset "100 @ Scannet(split='val', ROOT='data/scannet_processed', resolution=(224, 224), seed=777)" \
    --train_criterion "GaussianLoss()" \
    --test_criterion "GaussianLoss()" \
    --lr 0.0001 \
    --min_lr 1e-06 \
    --warmup_epochs 10 \
    --epochs 100 \
    --batch_size 4 \
    --accum_iter 1 \
    --save_freq 1 \
    --keep_freq 5 \
    --eval_freq 1 \
    --output_dir $SAVE_DIR
