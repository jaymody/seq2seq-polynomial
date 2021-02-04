import os
import sys
import pickle

import numpy as np
from tqdm import tqdm

from train import Seq2Seq, train
from data import tokenize
from utils import load_file, score

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

if __name__ == "__main__":
    if "-t" in sys.argv:
        factors, expansions = load_file("data/train.txt")
        train(factors, expansions)
    else:
        dirpath = "models/test_run"
        with open(os.path.join(dirpath, "src_field.pickle"), "rb") as fi:
            src_field = pickle.load(fi)
        with open(os.path.join(dirpath, "trg_field.pickle"), "rb") as fi:
            trg_field = pickle.load(fi)

        model = Seq2Seq.load_from_checkpoint(
            os.path.join(dirpath, "model-epoch=07-val_loss=0.3249-v1.ckpt"),
            input_dim=len(src_field.vocab),
            output_dim=len(trg_field.vocab),
            src_field=src_field,
            trg_field=trg_field,
        )

        factors, expansions = load_file("data/train.txt")
        factors = factors[:2000]
        expansions = expansions[:2000]

        pred = ["".join(model.predict(tokenize(f))[0]) for f in tqdm(factors)]
        scores = [score(te, pe) for te, pe in zip(expansions, pred)]
        print(np.mean(scores))
