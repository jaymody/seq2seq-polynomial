import os
import pickle
import random
import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from data import PolynomialLanguage
from utils import get_device, set_seed, load_file, score
from layers import Encoder, Decoder

device = get_device()


def create_vocabs(pairs):
    src_lang = PolynomialLanguage()
    trg_lang = PolynomialLanguage()

    for src, trg in tqdm(pairs, desc="creating vocabs"):
        src_lang.add_sentence(src)
        trg_lang.add_sentence(trg)

    return src_lang, trg_lang


def train_test_split(pairs, train_test_split_ratio):
    random.shuffle(pairs)
    split = int(train_test_split_ratio * len(pairs))
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    return train_pairs, test_pairs


class Collater:
    def __init__(self, src_lang, trg_lang):
        self.src_lang = src_lang
        self.trg_lang = trg_lang

    def __call__(self, batch):
        src_tensors, trg_tensors = zip(*batch)
        src_tensors = nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=self.src_lang.PAD_idx
        )
        trg_tensors = nn.utils.rnn.pad_sequence(
            trg_tensors, batch_first=True, padding_value=self.trg_lang.PAD_idx
        )
        return src_tensors, trg_tensors


class PolynomialDataset(Dataset):
    def __init__(self, pairs, src_lang, trg_lang):
        def sentence_to_tensor(sentence, lang):
            indexes = [lang.word2index[w] for w in lang.sentence_to_words(sentence)]
            indexes = [lang.SOS_idx] + indexes + [lang.EOS_idx]
            return torch.LongTensor(indexes)

        self.src_tensors, self.trg_tensors = zip(
            *[
                (sentence_to_tensor(src, src_lang), sentence_to_tensor(trg, trg_lang))
                for src, trg in tqdm(pairs, desc="creating dataset")
            ]
        )
        assert len(self.src_tensors) == len(self.trg_tensors)

    def __len__(self):
        return len(self.src_tensors)

    def __getitem__(self, index):
        return self.src_tensors[index], self.trg_tensors[index]


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        src_lang,
        trg_lang,
        hid_dim=256,
        enc_layers=3,
        dec_layers=3,
        enc_heads=8,
        dec_heads=8,
        enc_pf_dim=512,
        dec_pf_dim=512,
        enc_dropout=0.1,
        dec_dropout=0.1,
        lr=0.0005,
    ):
        super().__init__()

        self.input_dim = src_lang.n_words
        self.output_dim = trg_lang.n_words

        self.encoder = Encoder(
            self.input_dim,
            hid_dim,
            enc_layers,
            enc_heads,
            enc_pf_dim,
            enc_dropout,
            device,
        )

        self.decoder = Decoder(
            self.output_dim,
            hid_dim,
            dec_layers,
            dec_heads,
            dec_pf_dim,
            dec_dropout,
            device,
        )

        self.src_lang = src_lang
        self.trg_lang = trg_lang

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_lang.PAD_idx)
        self.lr = lr

        self.initialize_weights()
        self.to(device)

    def initialize_weights(self):
        def _initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.encoder.apply(_initialize_weights)
        self.decoder.apply(_initialize_weights)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_lang.PAD_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_lang.PAD_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len)).type_as(trg)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention

    def predict(self, sentence, max_len=31):
        src_indexes = [
            self.src_lang.word2index[word]
            for word in self.src_lang.sentence_to_words(sentence)
        ]
        src_indexes = [self.src_lang.SOS_idx] + src_indexes + [self.src_lang.EOS_idx]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)

        src_mask = self.make_src_mask(src_tensor)
        enc_src = self.encoder(src_tensor, src_mask)

        trg_indexes = [self.trg_lang.SOS_idx]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)

            trg_mask = self.make_trg_mask(trg_tensor)

            output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            pred = output.argmax(2)[:, -1].item()

            if pred == self.trg_lang.EOS_idx:
                break

            trg_indexes.append(pred)

        pred_words = [self.trg_lang.index2word[i] for i in trg_indexes[1:]]
        pred_sentence = self.trg_lang.words_to_sentence(pred_words)

        return pred_sentence, pred_words, attention

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        src, trg = batch

        output, _ = self(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = self.criterion(output, trg)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        src, trg = batch

        output, _ = self(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = self.criterion(output, trg)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


def train(
    pairs,
    dirpath,
    train_test_split_ratio=0.9,
    batch_size=128,
    num_workers=8,
    seed=1234,
):
    set_seed(seed)

    src_lang, trg_lang = create_vocabs(pairs)
    train_pairs, test_pairs = train_test_split(
        pairs, train_test_split_ratio=train_test_split_ratio
    )

    train_dataset = PolynomialDataset(train_pairs, src_lang, trg_lang)
    test_dataset = PolynomialDataset(test_pairs, src_lang, trg_lang)

    collate_fn = Collater(src_lang, trg_lang)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    save_to_pickle = {
        "src_lang.pickle": src_lang,
        "trg_lang.pickle": trg_lang,
        "train_pairs.pickle": train_pairs,
        "test_pairs.pickle": test_pairs,
    }
    for k, v in save_to_pickle.items():
        with open(os.path.join(dirpath, k), "wb") as fo:
            pickle.dump(v, fo)

    model = Seq2Seq(src_lang, trg_lang).to(device)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=dirpath,
        filename="model",
        save_top_k=1,
        mode="min",
    )

    lim_ratio = 0.03
    trainer = pl.Trainer(
        default_root_dir=dirpath,  # set directory to save weights, logs, etc ...
        gpus=1,  # num gpus to use if using gpu
        fast_dev_run=False,  # set to True to quickly verify your code works
        progress_bar_refresh_rate=20,  # change to 20 if using google colab
        gradient_clip_val=1,
        max_epochs=8,
        limit_train_batches=lim_ratio,  # percentage of train data to use
        limit_val_batches=lim_ratio,  # percentage of validation data to use
        limit_test_batches=1.00,  # percentage of test data to use
        val_check_interval=1.0,  # run validation after every n percent of an epoch
        precision=32,  # use 16 for half point precision
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(model, test_dataloader)

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

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the models.")
    parser.add_argument("--file_path", type=str, default="data/train.txt")
    args = parser.parse_args()

    dirpath = "models/another_test"
    pairs = load_file(args.file_path)
    train(pairs, dirpath)
