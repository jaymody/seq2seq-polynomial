"""Evaluate model."""
import argparse

from train import evaluate, load_model
from data import PolynomialLanguage
from utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str)
    parser.add_argument("--model_ckpt", type=str, default="model.ckpt")
    parser.add_argument("--data_path", type=str, default="data/test_set.txt")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    pairs = PolynomialLanguage.load_pairs(args.data_path)
    model = load_model(args.dirpath, args.model_ckpt)
    evaluate(model, pairs)
