import re
import random
import argparse


class Language:
    PAD_idx = 0
    SOS_idx = 1
    EOS_idx = 2
    UNK_idx = 3

    def __init__(self):
        self.word2count = {}
        self.word2index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = 4
        self.max_length = 0

    def sentence_to_words(self, sentence):
        raise NotImplementedError()

    def words_to_sentence(self, words):
        raise NotImplementedError()

    def add_sentence(self, sentence):
        words = self.sentence_to_words(sentence)

        if len(words) > self.max_length:
            self.max_length = len(words)

        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class PolynomialLanguage(Language):
    def sentence_to_words(self, sentence):
        return re.findall(r"sin|cos|tan|\d|\w|\(|\)|\+|-|\*+", sentence.strip().lower())

    def words_to_sentence(self, words):
        return "".join(words)

    @staticmethod
    def load_pairs(filepath, reverse=False):
        with open(filepath) as fi:
            pairs = [line.strip().split("=") for line in fi]

        if reverse:
            pairs = [(b, a) for a, b in pairs]

        return pairs


def train_test_split(pairs, train_test_split_ratio):
    random.shuffle(pairs)
    split = int(train_test_split_ratio * len(pairs))
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    return train_pairs, test_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates train test split.")
    parser.add_argument("--datapath", type=str, default="data/train.txt")
    parser.add_argument("--trainpath", type=str, default="data/train_set.txt")
    parser.add_argument("--testpath", type=str, default="data/test_set.txt")
    parser.add_argument("--ratio", type=int, default=0.98)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.datapath) as fi:
        pairs = fi.read().splitlines()
        train_pairs, test_pairs = train_test_split(pairs, args.ratio)

    with open(args.trainpath, "w") as fo:
        fo.write("\n".join(train_pairs) + "\n")

    with open(args.testpath, "w") as fo:
        fo.write("\n".join(test_pairs) + "\n")

    print(f"num pairs:          {len(pairs)}")
    print(f"num train pairs:    {len(train_pairs)}")
    print(f"num test pairs:     {len(test_pairs)}")
