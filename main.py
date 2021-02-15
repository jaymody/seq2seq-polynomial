import os
import pickle
import random
import argparse

from tqdm import tqdm

from utils import get_device, score, load_file
from train import Seq2Seq

device = get_device()


def evaluate(dirpath, model_ckpt="model.ckpt", test_pairs=None):
    if test_pairs == None:
        with open(os.path.join(dirpath, "test_pairs.pickle"), "rb") as fi:
            src_sentences, trg_sentences = zip(*pickle.load(fi))
    else:
        src_sentences, trg_sentences = zip(*test_pairs)

    with open(os.path.join(dirpath, "src_lang.pickle"), "rb") as fi:
        src_lang = pickle.load(fi)
    with open(os.path.join(dirpath, "trg_lang.pickle"), "rb") as fi:
        trg_lang = pickle.load(fi)

    model = Seq2Seq.load_from_checkpoint(
        os.path.join(dirpath, model_ckpt),
        src_lang=src_lang,
        trg_lang=trg_lang,
    ).to(device)
    prd_sentences, _, _ = model.predict(src_sentences, batch_size=512)
    assert len(prd_sentences) == len(src_sentences) == len(trg_sentences)

    total_score = 0
    for i, (src, trg, prd) in enumerate(
        tqdm(zip(src_sentences, trg_sentences, prd_sentences), desc="scoring")
    ):
        pred_score = score(trg, prd)
        total_score += pred_score
        if i < 10:
            print(f"\n\n\n---- Example {i} ----")
            print(f"src = {src}")
            print(f"trg = {trg}")
            print(f"prd = {prd}")
            print(f"score = {pred_score}")

    print(f"{total_score}/{len(prd_sentences)} = {total_score/len(prd_sentences)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str)
    parser.add_argument("--model_ckpt", type=str, default="model.ckpt")
    parser.add_argument("--data_path", type=str, default="data/test_set.txt")
    args = parser.parse_args()

    pairs = load_file(args.data_path)
    evaluate(args.dirpath, args.model_ckpt, test_pairs=pairs)
