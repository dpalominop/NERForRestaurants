import pandas as pd
import numpy as np

import joblib
import torch
import os
import shutil

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
from dataset import BertDataLoader
import engine
from model import BertModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    enc_tags = preprocessing.LabelEncoder()
    
    df.loc[:, "Tag"] = enc_tags.fit_transform(df["Tag"])
    
    sentences = df.groupby("Id")["Word"].apply(list).values
    tags = df.groupby("Id")["Tag"].apply(list).values
    
    return sentences[0:141], tags[0:141], enc_tags


if __name__ == "__main__":
    RUN_PATH = config.BASE_MODEL_PATH+f"/{config.THIS_RUN}"
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.mkdir(RUN_PATH)

    sentences, tags, enc_tags = process_data(config.TRAINING_FILE)
    num_tags = len(list(enc_tags.classes_))
    
    (
        train_sentences,
        test_sentences,
        train_tags,
        test_tags
    ) = model_selection.train_test_split(sentences, tags, random_state=42, test_size=0.1)
    
    train_data_loader = BertDataLoader(texts=train_sentences, tags=train_tags, 
                                              batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
    valid_data_loader = BertDataLoader(texts=test_sentences, tags=test_tags, 
                                              batch_size=config.VALID_BATCH_SIZE, num_workers=1)
    print("Loaded training and validation data into DataLoaders.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel(config.BASE_MODEL_PATH, num_labels=num_tags)
    model.to(device)
    print(f"Initialized model and moved it to {device}.")
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        }
    ]
    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    schedule = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    for epoch in range(config.EPOCHS):
        tr_loss, tr_acc = engine.train_fn(train_data_loader, model, optimizer, device, schedule)
        val_loss, val_acc, cl_report = engine.eval_fn(valid_data_loader, enc_tags, model, device)
        print(f"[{epoch}]: Loss = {tr_loss} Acc = {tr_acc} / Val Loss = {val_loss} Val Acc = {val_acc}")
        print(f"Classification Report:\n {cl_report}")

        if epoch == config.EPOCHS-1:
            # Save model and config
            model.set_config(enc_tags)
            model.bert.save_pretrained(RUN_PATH)
