import os
import pickle
import random
import argparse

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from data import create_examples, train_test_split, create_iterators
from utils import get_device, set_seed, load_file
from layers import Encoder, Decoder

device = get_device()


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        src_field,
        trg_field,
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

        self.encoder = Encoder(
            input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device
        )

        self.decoder = Decoder(
            output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, dec_dropout, device
        )

        self.src_field = src_field
        self.trg_field = trg_field

        self.src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
        self.trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]

        self.init_token = src_field.init_token
        self.init_idx = src_field.vocab.stoi[src_field.init_token]

        self.eos_token = trg_field.eos_token
        self.eos_idx = trg_field.vocab.stoi[trg_field.eos_token]

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        self.lr = lr
        self.to(device)

        self.initialize_weights()

    def initialize_weights(self):
        def _initialize_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.encoder.apply(_initialize_weights)
        self.decoder.apply(_initialize_weights)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

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

    def predict(self, tokens, max_len=31):
        tokens = [self.init_token] + tokens + [self.eos_token]
        src_indexes = [self.src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)

        src_mask = self.make_src_mask(src_tensor)
        enc_src = self.encoder(src_tensor, src_mask)

        trg_indexes = [self.src_field.vocab.stoi[self.init_token]]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)

            trg_mask = self.make_trg_mask(trg_tensor)

            output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()

            if pred_token == self.trg_field.vocab.stoi[self.trg_field.eos_token]:
                break

            trg_indexes.append(pred_token)

        trg_tokens = [self.trg_field.vocab.itos[i] for i in trg_indexes[1:]]

        return trg_tokens, attention

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        src = batch.src
        trg = batch.trg

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
        src = batch.src
        trg = batch.trg

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


def train(factors, expansions):
    dirpath = "models/test_run"
    set_seed(1234)

    examples, src_field, trg_field = create_examples(factors, expansions)
    train_examples, test_examples = train_test_split(
        examples, train_test_split_ratio=0.9
    )
    train_iterator, valid_iterator, test_iterator = create_iterators(
        train_examples, test_examples, src_field, trg_field, batch_size=128
    )

    save_to_pickle = {
        "src_field.pickle": src_field,
        "trg_field.pickle": trg_field,
        "train_examples.pickle": train_examples,
        "test_examples.pickle": test_examples,
    }
    for k, v in save_to_pickle.items():
        with open(os.path.join(dirpath, k), "wb") as fo:
            pickle.dump(v, fo)

    model = Seq2Seq(
        input_dim=len(src_field.vocab),
        output_dim=len(trg_field.vocab),
        src_field=src_field,
        trg_field=trg_field,
    ).to(device)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=dirpath,
        filename="model-{epoch:02d}-{val_loss:.4f}",
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

    trainer.fit(model, train_iterator, valid_iterator)

    trainer.test(test_dataloaders=test_iterator)

    # model = Seq2Seq.load_from_checkpoint(
    #     "models/first_run/model-epoch=07-val_loss=0.02.ckpt",
    #     input_dim=len(SRC.vocab),
    #     output_dim=len(TRG.vocab),
    #     src_pad_idx=SRC.vocab.stoi[SRC.pad_token],
    #     trg_pad_idx=TRG.vocab.stoi[TRG.pad_token],
    #     device=device
    # )

    example_idx = 11

    src = test_examples[example_idx].src
    trg = test_examples[example_idx].trg
    translation, _ = model.predict(src)

    print("\n\n\n---- Example ----")
    print(f'src = {"".join(src)}')
    print(f'trg = {"".join(trg)}')
    print(f'prd = {"".join(translation)}')

    pred_examples = random.sample(test_examples, 1000)
    correct = 0

    for example in tqdm(pred_examples):
        pred = model.predict(example.src)[0]
        if "".join(pred) == "".join(example.trg):
            correct += 1

    print(f"{correct}/{len(pred_examples)} = {correct/len(pred_examples)}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the models.")
    parser.add_argument("--file_path", type=str, default="data/train.txt")
    args = parser.parse_args()

    factors, expansions = load_file(args.file_path)
    train(factors, expansions)
