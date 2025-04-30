python method/train.py \
    --max-epoch 50 \
    --mode clip \
    --semantic-size 512 \
    --text_type gpt \
    --shot 1  \
    --step-size 40 \
    --test-batch 600 \
    --batch-size 128 \
    --num-workers 8 \
    --lr 1e-4 \
    --dataset FC100 \
    --backbone resnet
# MiniImageNet/TiredImageNet/CIFAR-FS/FC100
# resnet/swin