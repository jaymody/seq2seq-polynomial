import os
import pickle
import random
import argparse

from tqdm import tqdm

from utils import get_device, score
from train import Seq2Seq

device = get_device()


def test(dirpath, model_ckpt, test_pairs=None):
    with open(os.path.join(dirpath, "src_lang.pickle"), "rb") as fi:
        src_lang = pickle.load(fi)
    with open(os.path.join(dirpath, "trg_lang.pickle"), "rb") as fi:
        trg_lang = pickle.load(fi)

    if test_pairs == None:
        with open(os.path.join(dirpath, "test_pairs.pickle"), "rb") as fi:
            test_pairs = pickle.load(fi)

    model = Seq2Seq.load_from_checkpoint(
        os.path.join(dirpath, model_ckpt),
        src_lang=src_lang,
        trg_lang=trg_lang,
    ).to(device)

    total_score = 0
    pred_pairs = random.sample(test_pairs, 1000)
    for i, (pred_src, pred_trg) in enumerate(tqdm(pred_pairs, desc="scoring")):
        translation, _, _ = model.predict(pred_src)
        pred_score = score(pred_trg, translation)
        total_score += pred_score

        if i < 10:
            print(f"\n\n\n---- Example {i} ----")
            print(f"src = {pred_src}")
            print(f"trg = {pred_trg}")
            print(f"prd = {translation}")
            print(f"score = {pred_score}")

    print(f"{total_score}/{len(pred_pairs)} = {total_score/len(pred_pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str)
    parser.add_argument("--model_ckpt", type=str, default="model.ckpt")
    args = parser.parse_args()
    test(args.dirpath, args.model_ckpt)
