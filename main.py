import os
import pickle
import argparse

from train import evaluate, load_model
from utils import load_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str)
    parser.add_argument("--model_ckpt", type=str, default="model.ckpt")
    parser.add_argument("--data_path", type=str, default="data/test_set.txt")
    args = parser.parse_args()

    pairs = load_file(args.data_path)
    model = load_model(args.dirpath, args.model_ckpt)
    evaluate(model, pairs)
