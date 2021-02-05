import re


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
