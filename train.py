import os
import pickle
import random
import argparse

import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from data import PolynomialLanguage, train_test_split
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


class Collater:
    def __init__(self, src_lang, trg_lang=None, predict=False):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.predict = predict

    def __call__(self, batch):
        # TODO: try pack_padded_sequence for faster processing
        if self.predict:
            # batch = src_tensors in predict mode
            return nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=self.src_lang.PAD_idx
            )

        src_tensors, trg_tensors = zip(*batch)
        src_tensors = nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=self.src_lang.PAD_idx
        )
        trg_tensors = nn.utils.rnn.pad_sequence(
            trg_tensors, batch_first=True, padding_value=self.trg_lang.PAD_idx
        )
        return src_tensors, trg_tensors


def sentence_to_tensor(sentence, lang):
    indexes = [lang.word2index[w] for w in lang.sentence_to_words(sentence)]
    indexes = [lang.SOS_idx] + indexes + [lang.EOS_idx]
    return torch.LongTensor(indexes)


def pairs_to_tensors(pairs, src_lang, trg_lang):
    tensors = [
        (sentence_to_tensor(src, src_lang), sentence_to_tensor(trg, trg_lang))
        for src, trg in tqdm(pairs, desc="creating tensors")
    ]
    return tensors


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


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

        self.save_hyperparameters()
        del self.hparams["src_lang"]
        del self.hparams["trg_lang"]

        self.src_lang = src_lang
        self.trg_lang = trg_lang

        self.encoder = Encoder(
            src_lang.n_words,
            hid_dim,
            enc_layers,
            enc_heads,
            enc_pf_dim,
            enc_dropout,
            device,
        )

        self.decoder = Decoder(
            trg_lang.n_words,
            hid_dim,
            dec_layers,
            dec_heads,
            dec_pf_dim,
            dec_dropout,
            device,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_lang.PAD_idx)
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

    def predict_sentence(self, sentence, max_len=31):
        """Predicts a single sentence."""
        src_indexes = [
            self.src_lang.word2index[word]
            for word in self.src_lang.sentence_to_words(sentence)
        ]
        src_indexes = [self.src_lang.SOS_idx] + src_indexes + [self.src_lang.EOS_idx]

        # src_indexes = [src len], where src len is len(<sos> sentence <eos>)

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)

        # src_tensor = [batch size = 1, src len]

        src_mask = self.make_src_mask(src_tensor)

        # src_mask = [batch size = 1, 1, 1, src len]

        enc_src = self.encoder(src_tensor, src_mask)

        # enc_src = [batch size = 1, src len, hid dim]

        trg_indexes = [self.trg_lang.SOS_idx]

        # trg_indexes = [cur trg len = 1]

        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)

            # trg_tensor = [1, cur trg len]

            trg_mask = self.make_trg_mask(trg_tensor)

            # trg_mask = [batch size = 1, 1, cur trg len, cur trg len]

            output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            # output = [batch size = 1, cur trg len, output dim]

            pred = output.argmax(2)[:, -1].item()

            if pred == self.trg_lang.EOS_idx:
                break

            trg_indexes.append(pred)

        pred_words = [self.trg_lang.index2word[i] for i in trg_indexes[1:]]
        pred_sentence = self.trg_lang.words_to_sentence(pred_words)

        return pred_sentence, pred_words, attention

    def predict(self, sentences, max_len=31, batch_size=128, num_workers=1):
        """Efficiently predict a list of sentences"""
        pred_tensors = [
            sentence_to_tensor(sentence, self.src_lang)
            for sentence in tqdm(sentences, desc="creating prediction tensors")
        ]

        collate_fn = Collater(self.src_lang, predict=True)
        pred_dataloader = DataLoader(
            SimpleDataset(pred_tensors),
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        sentences = []
        words = []
        attention = []
        for batch in tqdm(pred_dataloader, desc="predict batch num"):
            preds = self.predict_batch(batch.to(device), max_len)
            pred_sentences, pred_words, pred_attention = preds
            sentences.extend(pred_sentences)
            words.extend(pred_words)
            attention.extend(pred_attention)

        # sentences = [num pred sentences]
        # words = [num pred sentences, trg len]
        # attention = [num pred sentences, n heads, trg len, src len]

        return sentences, words, attention

    def predict_batch(self, batch, max_len=31):
        """Predicts on a batch of src_tensors."""
        # batch = src_tensor when predicting = [batch_size, src len]

        src_tensor = batch
        src_mask = self.make_src_mask(batch)

        # src_mask = [batch size, 1, 1, src len]

        enc_src = self.encoder(src_tensor, src_mask)

        # enc_src = [batch size, src len, hid dim]

        trg_indexes = [[self.trg_lang.SOS_idx] for _ in range(len(batch))]

        # trg_indexes = [batch_size, cur trg len = 1]

        trg_tensor = torch.LongTensor(trg_indexes).to(self.device)

        # trg_tensor = [batch_size, cur trg len = 1]
        # cur trg len increases during the for loop up to the max len

        for _ in range(max_len):

            trg_mask = self.make_trg_mask(trg_tensor)

            # trg_mask = [batch size, 1, cur trg len, cur trg len]

            output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            # output = [batch size, cur trg len, output dim]

            preds = output.argmax(2)[:, -1].reshape(-1, 1)

            # preds = [batch_size, 1]

            trg_tensor = torch.cat((trg_tensor, preds), dim=-1)

            # trg_tensor = [batch_size, cur trg len], cur trg len increased by 1

        src_tensor = src_tensor.detach().cpu().numpy()
        trg_tensor = trg_tensor.detach().cpu().numpy()
        attention = attention.detach().cpu().numpy()

        pred_words = []
        pred_sentences = []
        pred_attention = []
        for src_indexes, trg_indexes, attn in zip(src_tensor, trg_tensor, attention):
            # trg_indexes = [trg len = max len (filled with eos if max len not needed)]
            # src_indexes = [src len = len of longest sentence (padded if not longest)]

            # indexes where first eos tokens appear
            src_eosi = np.where(src_indexes == self.src_lang.EOS_idx)[0][0]
            trg_eosi = np.where(trg_indexes == self.trg_lang.EOS_idx)[0][0]

            # cut target indexes up to first eos token and also exclude sos token
            trg_indexes = trg_indexes[1:trg_eosi]

            # attn = [n heads, trg len=max len, src len=max len of sentence in batch]
            # we want to keep n heads, but we'll cut trg len and src len up to
            # their first eos token
            attn = attn[:, :trg_eosi, :src_eosi]  # cut attention for trg eos tokens

            words = [self.trg_lang.index2word[index] for index in trg_indexes]
            sentence = self.trg_lang.words_to_sentence(words)
            pred_words.append(words)
            pred_sentences.append(sentence)
            pred_attention.append(attn)

        # pred_sentences = [batch_size]
        # pred_words = [batch_size, trg len]
        # attention = [batch size, n heads, trg len (varies), src len (varies)]

        return pred_sentences, pred_words, pred_attention

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

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
    train_val_split_ratio=0.95,
    batch_size=128,
    num_workers=8,
    seed=1234,
):
    set_seed(seed)

    src_lang, trg_lang = create_vocabs(pairs)
    train_pairs, val_pairs = train_test_split(
        pairs, train_test_split_ratio=train_val_split_ratio
    )

    train_tensors = pairs_to_tensors(train_pairs, src_lang, trg_lang)
    val_tensors = pairs_to_tensors(train_pairs, src_lang, trg_lang)

    collate_fn = Collater(src_lang, trg_lang)
    train_dataloader = DataLoader(
        SimpleDataset(train_tensors),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        SimpleDataset(val_tensors),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    save_to_pickle = {
        "src_lang.pickle": src_lang,
        "trg_lang.pickle": trg_lang,
        "train_pairs.pickle": train_pairs,
        "val_pairs.pickle": val_pairs,
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

    lim_ratio = 1.00
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
        val_check_interval=0.2,  # run validation after every n percent of an epoch
        precision=32,  # use 16 for half point precision
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, val_dataloader)

    total_score = 0
    pred_pairs = random.sample(val_pairs, 1000)
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
    # TODO: add cli for model hparams and trainer hparams and program hparams
    parser = argparse.ArgumentParser("Train the models.")
    parser.add_argument("--file_path", type=str, default="data/train_set.txt")
    args = parser.parse_args()

    # TODO: seperate data file into train.txt and test.txt to ensure 0 overlap
    # and allows consitent scoring across models
    # (write assertions to make sure there is no overlap in the split files)
    _dirpath = "models/run1"
    _pairs = load_file(args.file_path)
    train(_pairs, _dirpath)
