for train_seed in 41 42 43; do
    for split_seed in 41 42; do
        python scripts/train.py name=0001_${split_seed}_${train_seed} trainer.seed=$train_seed datamodule.splitter.seed=$split_seed log_dir=logs/norm/
    done
done