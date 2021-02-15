# Seq2Seq - Polynomial Expansion
This repo contains an implementation of a transformer for seq2seq tasks (ie, translation) using PyTorch and PyTorch Lightning.

## The Problem
Implement a deep learning model that learns to expand single variable polynomials, where the model takes the factorized sequence as input and predict the expanded sequence. For example:

* `n*(n-11)=n**2-11*n`
* `n*(n-11)` is the factorized input
* `n**2-11*n`  is the expanded target

The expanded expressions are commutable, but only the form provided is considered correct.

## Model Accuracy
The model is evaluated against a strict string equality between the predicted target sequence and the groud truth target sequence. The model achieved an accuracy of`0.88`.


## Instructions
Split into train and test set (use `--help` for more options):
```shell
python data.py
```

Train the model (use `--help` for more options):
```shell
python train.py \
    "models/best" \
    --gpus 1 \
    --gradient_clip_val 1 \
    --max_epochs 10 \
    --val_check_interval 0.2
```

Evaluate model on test set (use `--help` for more options):
```shell
python evaluate.py models/best
```

Run unit tests:
```shell
python -m unittest tests.py
```
